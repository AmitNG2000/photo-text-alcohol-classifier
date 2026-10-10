"""Run CPU alcohol classification on an unlabeled directory of images.

Usage:
    python src/inference.py /path/to/images --output results/results.csv
"""

import argparse
import json
import re
import unicodedata
from collections import Counter, defaultdict
from difflib import get_close_matches
from pathlib import Path

import cmudict
import easyocr
import joblib
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageOps
from sklearn.preprocessing import normalize
from tqdm.auto import tqdm

if __package__:
    from .alcohol_lists import ALCOHOL_BRANDS, ALCOHOL_DRINKS
else:
    from alcohol_lists import ALCOHOL_BRANDS, ALCOHOL_DRINKS


PROJECT_ROOT = Path(__file__).resolve().parent.parent

MIN_ENGLISH_RATIO = 0.80
MIN_OCR_CONFIDENCE = 0.80
MIN_SCORE_GAIN = 0.05
MIN_TEXT_SCORE = 0.20
OCR_PIPELINE_VERSION = 3
OCR_OPTIONS = {
    "decoder": "greedy",
    "min_size": 10,
    "width_ths": 0.30,
    "contrast_ths": 0.30,
    "adjust_contrast": 0.70,
    "rotation_info": [180],
}
OCR_MAGNIFICATION = 1.5
OCR_BATCH_SIZE = 8
BEVERAGE_WORDS = {
    "absolut", "bacardi", "budweiser", "heineken", "jagermeister", "smirnoff",
    "matcha", "kombucha", "yerba", "espresso", "cappuccino", "latte",
    "coffee", "tea", "juice", "lemonade", "milk", "water", "chocolate",
}
ALCOHOL_TERMS = set(ALCOHOL_BRANDS) | set(ALCOHOL_DRINKS)
OCR_RETRY_COLUMNS = {
    "english_word_ratio", "flip_attempted", "selected_orientation",
    "original_text", "original_mean_confidence", "original_english_word_ratio",
    "flipped_text", "flipped_mean_confidence", "flipped_english_word_ratio",
    "quality_score", "selected_preprocessing", "enhancement_attempted",
    "enhanced_text", "enhanced_mean_confidence", "enhanced_english_word_ratio",
    "ocr_cache_key",
}

MIN_WORD_SIMILARITY = 0.90
MIN_CORRECTION_MARGIN = 0.03
NUMBER_TO_LETTER = {
    "0": "o", "1": "il", "2": "z", "3": "e", "4": "a",
    "5": "s", "6": "g", "7": "t", "8": "bs", "9": "g",
}
LETTER_CONFUSIONS = {"i": "l", "o": "q"}
COMMON_PHRASES = {"get ready", "don't miss", "limited edition", "check out"}
COMMON_TEXT_WORDS = {
    "dont", "don't", "miss", "limited", "edition", "nightlife", "event",
    "get", "ready", "for", "check", "out", "grab", "now", "try", "enjoy",
    "smooth", "premium", "imported", "menu", "special", "offer", "party",
    "reserve", "flavor", "taste", "celebration", "explore", "experience",
    "discount", "bar", "new",
}


def load_english_vocabulary(extra_words):
    """Load a shared vocabulary for OCR scoring and text cleaning.

    Args:
        extra_words (set[str]): Additional beverage and brand names.

    Returns:
        set[str]: Normalized dictionary words and domain-term components.
    """
    vocabulary = {word.casefold() for word in cmudict.words()}
    for term in set(extra_words) | ALCOHOL_TERMS:
        vocabulary.update(re.findall(r"[^\W_]+(?:'[^\W_]+)*", term.casefold()))
    return vocabulary


def calculate_english_word_ratio(text, english_words):
    """Measure dictionary coverage without changing the transcription.

    Args:
        text (str): OCR text to evaluate.
        english_words (set[str]): Recognized lowercase words.

    Returns:
        float: Recognized-word fraction, or zero for text without words.
    """
    text = text.casefold().replace("’", "'").replace("‘", "'")
    tokens = re.findall(r"[^\W_]+(?:'[^\W_]+)*", text)
    words = [token for token in tokens if not token.isnumeric()]
    return sum(word in english_words for word in words) / len(words) if words else 0.0


def create_ocr_reader(model_dir):
    """Initialize one English EasyOCR reader on the CPU.

    Args:
        model_dir (Path): Directory containing downloaded model weights.

    Returns:
        easyocr.Reader: Reader shared across all images.
    """
    return easyocr.Reader(
        ["en"], gpu=False, model_storage_directory=str(model_dir),
        user_network_directory=str(model_dir / "user_network"), verbose=True,
    )


def summarize_ocr_attempt(detections, english_words):
    """Order text boxes by line and summarize an OCR attempt.

    Args:
        detections (list): Bounding-box, text, and confidence tuples.
        english_words (set[str]): Recognized lowercase words.

    Returns:
        dict: Transcription, coverage, character-weighted confidence, and boxes.
    """
    detections = [item for item in detections if item[1].strip()]
    lines = []
    by_height = sorted(detections, key=lambda item: np.mean(item[0], axis=0)[1])
    for box, text, confidence in by_height:
        points = np.asarray(box, dtype=float)
        center_x, center_y = points.mean(axis=0)
        height = max(np.ptp(points[:, 1]), 1.0)
        item = (center_x, center_y, height, box, text.strip(), float(confidence))
        same_line = lines and (
            abs(center_y - np.mean([entry[1] for entry in lines[-1]]))
            <= height * 0.6
        )
        if same_line:
            lines[-1].append(item)
        else:
            lines.append([item])
    ordered = [
        item for line in lines for item in sorted(line, key=lambda item: item[0])
    ]
    text = " ".join(item[4] for item in ordered)
    confidences = [item[5] for item in ordered]
    lengths = [max(len(item[4].replace(" ", "")), 1) for item in ordered]
    angles = [
        np.degrees(np.arctan2(box[1][1] - box[0][1], box[1][0] - box[0][0]))
        for box, _, _ in detections
    ]
    angles = [angle for angle in angles if abs(angle) <= 45]
    return {
        "extracted_text": text,
        "word_count": len(text.split()),
        "recognized_word_count": sum(
            token in english_words
            for token in re.findall(r"[^\W_]+(?:'[^\W_]+)*", text.casefold())
            if not token.isnumeric()
        ),
        "text_box_count": len(ordered),
        "mean_confidence": (
            np.average(confidences, weights=lengths) if ordered else np.nan
        ),
        "english_word_ratio": calculate_english_word_ratio(text, english_words),
        "text_angle": float(np.median(angles)) if angles else 0.0,
        "text_boxes": json.dumps([np.asarray(item[3]).tolist() for item in ordered]),
    }


def score_ocr_attempt(attempt):
    """Combine confidence and vocabulary coverage for attempt selection.

    Args:
        attempt (dict): Text, confidence, and English-word ratio for an attempt.

    Returns:
        float: Quality score, or -1 for an empty transcription.
    """
    if not attempt["extracted_text"].strip():
        return -1.0
    confidence = attempt["mean_confidence"]
    confidence = 0.0 if pd.isna(confidence) else confidence
    return 0.45 * confidence + 0.55 * attempt["english_word_ratio"]


def is_better_ocr_attempt(candidate, current, require_language=False):
    """Require a meaningful quality gain without losing recognized words.

    Args:
        candidate (dict): Alternative OCR attempt.
        current (dict): Currently selected OCR attempt.
        require_language (bool): Require vocabulary support for mirror selection.

    Returns:
        bool: Whether the alternative should replace the current result.
    """
    if not current["extracted_text"] and score_ocr_attempt(candidate) < MIN_TEXT_SCORE:
        return False
    current_known = current["recognized_word_count"]
    candidate_known = candidate["recognized_word_count"]
    language_supported = (
        candidate["english_word_ratio"] > 0
        and candidate["english_word_ratio"] >= current["english_word_ratio"]
    )
    return (
        (not require_language or language_supported)
        and score_ocr_attempt(candidate) > score_ocr_attempt(current) + MIN_SCORE_GAIN
        and candidate_known >= current_known * 0.8
    )


def read_ocr_attempt(image, reader, english_words, magnification=1.0):
    """Keep native color OCR as a baseline and enhance grayscale on retries.

    Args:
        image (PIL.Image.Image): EXIF-oriented image or a retry variant.
        reader (easyocr.Reader): Shared OCR reader.
        english_words (set[str]): Vocabulary for scoring.
        magnification (float): Detection magnification factor.

    Returns:
        dict: Summarized OCR attempt.
    """
    if magnification == 1.0:
        detections = reader.readtext(
            np.array(image.convert("RGB")), detail=1, paragraph=False,
            batch_size=OCR_BATCH_SIZE,
        )
    else:
        detections = reader.readtext(
            np.array(image.convert("L")), detail=1, paragraph=False,
            mag_ratio=magnification, batch_size=OCR_BATCH_SIZE, **OCR_OPTIONS,
        )
    return summarize_ocr_attempt(detections, english_words)


def needs_ocr_retry(attempt, min_english_ratio=0.80, min_confidence=0.80):
    """Retry only when both vocabulary coverage and confidence are low.

    Args:
        attempt (dict): OCR attempt with coverage and mean confidence.
        min_english_ratio (float): Vocabulary coverage threshold.
        min_confidence (float): Mean confidence threshold.

    Returns:
        bool: Whether both signals are below their thresholds. Missing
            confidence is treated as zero so empty detections can retry.
    """
    confidence = attempt["mean_confidence"]
    confidence = 0.0 if pd.isna(confidence) else confidence
    return (
        attempt["english_word_ratio"] < min_english_ratio
        and confidence < min_confidence
    )


def extract_image_text(
    dataset, reader, english_words, min_english_ratio=0.80, min_confidence=0.80
):
    """Read images and retry only while both OCR quality signals are low.

    Args:
        dataset (pd.DataFrame): Image metadata containing image_path.
        reader (easyocr.Reader): Shared OCR reader.
        english_words (set[str]): Shared scoring vocabulary.
        min_english_ratio (float): Coverage threshold for skipping retries.
        min_confidence (float): Confidence threshold for skipping retries.

    Returns:
        pd.DataFrame: Metadata, selected text, and original/retry diagnostics.
    """
    records = []
    for row in tqdm(dataset.itertuples(index=False), total=len(dataset), desc="OCR"):
        with Image.open(row.image_path) as image:
            image = ImageOps.exif_transpose(image).convert("RGB")
        original = read_ocr_attempt(image, reader, english_words)
        selected = original
        orientation = "original"
        preprocessing = "native"
        flipped = None
        enhanced = None
        flip_attempted = needs_ocr_retry(
            original, min_english_ratio, min_confidence
        )
        if flip_attempted:
            flipped = read_ocr_attempt(ImageOps.mirror(image), reader, english_words)
            if is_better_ocr_attempt(flipped, selected, require_language=True):
                selected, orientation = flipped, "flipped"
        enhancement_attempted = needs_ocr_retry(
            selected, min_english_ratio, min_confidence
        )
        if enhancement_attempted:
            retry_image = ImageOps.mirror(image) if orientation == "flipped" else image
            enhanced = read_ocr_attempt(
                ImageOps.autocontrast(retry_image.convert("L")), reader, english_words,
                OCR_MAGNIFICATION,
            )
            if is_better_ocr_attempt(enhanced, selected):
                selected, preprocessing = enhanced, "contrast_scaled"
            angle = selected["text_angle"]
            angles = [angle] if abs(angle) >= 5 else []
            if not selected["extracted_text"]:
                angles = [-30, 30]
            for angle in angles:
                if not needs_ocr_retry(selected, min_english_ratio, min_confidence):
                    break
                deskewed = retry_image.rotate(angle, expand=True, fillcolor="white")
                attempt = read_ocr_attempt(
                    deskewed, reader, english_words, OCR_MAGNIFICATION
                )
                if is_better_ocr_attempt(attempt, selected):
                    selected, preprocessing = attempt, f"deskew_{angle:.1f}"
                if is_better_ocr_attempt(attempt, enhanced):
                    enhanced = attempt
        records.append({
            **selected,
            "quality_score": score_ocr_attempt(selected),
            "flip_attempted": flip_attempted,
            "selected_orientation": orientation,
            "selected_preprocessing": preprocessing,
            "enhancement_attempted": enhancement_attempted,
            **ocr_attempt_diagnostics(original, "original"),
            **ocr_attempt_diagnostics(flipped, "flipped"),
            **ocr_attempt_diagnostics(enhanced, "enhanced"),
        })
    return pd.concat([dataset.reset_index(drop=True), pd.DataFrame(records)], axis=1)


def ocr_attempt_diagnostics(attempt, prefix):
    """Name the same diagnostic fields for an original or retry attempt.

    Args:
        attempt (dict | None): OCR attempt, or None if it was not run.
        prefix (str): Original, flipped, or enhanced column prefix.

    Returns:
        dict: Text, confidence, and vocabulary coverage with prefixed names.
    """
    attempt = attempt or {}
    return {
        f"{prefix}_text": attempt.get("extracted_text", ""),
        f"{prefix}_mean_confidence": attempt.get("mean_confidence", np.nan),
        f"{prefix}_english_word_ratio": attempt.get("english_word_ratio", np.nan),
    }


def normalize_ocr_text(text):
    """Normalize Unicode, case, apostrophes, and spacing in OCR text.

    Args:
        text (str): Raw OCR text, possibly missing.

    Returns:
        str: Normalized text, or an empty string for missing values.
    """
    if pd.isna(text):
        return ""
    text = unicodedata.normalize("NFKC", str(text)).lower()
    text = text.replace("’", "'").replace("‘", "'")
    return re.sub(r"\s+", " ", text).strip()


def build_cleaning_vocabulary(english_words, alcohol_terms):
    """Combine English words and component words from alcohol terms.

    Args:
        english_words (set[str]): Vocabulary loaded in the OCR section.
        alcohol_terms (set[str]): Imported alcohol-brand and drink terms.

    Returns:
        tuple: Combined vocabulary and candidate words grouped by length.
    """
    vocabulary = set(english_words)
    for term in alcohol_terms:
        vocabulary.update(
            re.findall(r"[^\W_]+(?:'[^\W_]+)*", normalize_ocr_text(term))
        )
    word_groups = defaultdict(list)
    for word in sorted(vocabulary):
        word_groups[len(word)].append(word)
    return vocabulary, word_groups


def calculate_ocr_edit_distance(word, target, confusions):
    """Measure edit distance with smaller costs for common OCR substitutions.

    Args:
        word (str): OCR token.
        target (str): Candidate vocabulary word.
        confusions (dict[str, str]): Characters that are easily confused.

    Returns:
        float: Edit cost; insertions/deletions cost one and confusions cost 0.25.
    """
    previous = list(range(len(target) + 1))
    for row, character in enumerate(word, start=1):
        current = [row]
        for column, expected in enumerate(target, start=1):
            substitution = 0.0 if character == expected else 1.0
            if (
                character != expected
                and (expected in confusions.get(character, "")
                     or character in confusions.get(expected, ""))
            ):
                substitution = 0.25
            current.append(min(
                previous[column] + 1, current[column - 1] + 1,
                previous[column - 1] + substitution,
            ))
        previous = current
    return previous[-1]


def correct_ocr_word(
    word, vocabulary, word_groups, confusions, min_similarity,
    previous_word="", next_word="", phrase_pairs=frozenset(),
):
    """Correct an unknown token only when one OCR-aware match clearly wins.

    Args:
        word (str): Normalized OCR token.
        vocabulary (set[str]): Shared English and beverage vocabulary.
        word_groups (dict): Candidate words grouped by length.
        confusions (dict[str, str]): Possible digit and letter substitutions.
        min_similarity (float): Minimum edit-based similarity.
        previous_word (str): Previous token, used as phrase evidence.
        next_word (str): Next token, used as phrase evidence.
        phrase_pairs (set[tuple]): Known neighboring words in domain phrases.

    Returns:
        str: Reliable correction, or the unchanged token if evidence is weak.
    """
    if word in vocabulary or not any(character.isalpha() for character in word):
        return word
    possible_words = [
        target for length in range(max(1, len(word) - 1), len(word) + 2)
        for target in word_groups.get(length, [])
    ]
    candidates = set(get_close_matches(word, possible_words, n=8, cutoff=0.75))
    candidates.update(
        target for before, target in phrase_pairs if before == previous_word
    )
    candidates.update(
        target for target, after in phrase_pairs if after == next_word
    )
    # Direct substitutions also recover short words that fuzzy matching misses.
    for index, character in enumerate(word):
        alternatives = set(confusions.get(character, ""))
        alternatives.update(
            source for source, values in confusions.items() if character in values
        )
        for alternative in alternatives:
            candidate = word[:index] + alternative + word[index + 1:]
            if candidate in vocabulary:
                candidates.add(candidate)
    scores = []
    for target in candidates:
        distance = calculate_ocr_edit_distance(word, target, confusions)
        similarity = 1 - distance / max(len(word), len(target))
        phrase_match = (
            (previous_word, target) in phrase_pairs
            or (target, next_word) in phrase_pairs
        )
        if phrase_match:
            similarity += 0.15
        scores.append((min(similarity, 1.0), target))
    scores.sort()
    if not scores or scores[-1][0] < min_similarity:
        return word
    if len(scores) > 1 and scores[-1][0] - scores[-2][0] < MIN_CORRECTION_MARGIN:
        return word
    return scores[-1][1]


def split_merged_ocr_word(word, vocabulary):
    """Recover spaces in long tokens while preserving all original characters.

    Args:
        word (str): Normalized OCR token.
        vocabulary (set[str]): Common text and beverage words of length >= 3.

    Returns:
        str: Segmented token when at least three words cover 75% of its letters.
    """
    if len(word) < 16 or word in vocabulary:
        return word
    # Unknown characters cost more than a recognized word, favoring long matches.
    best = [(0.0, [])] + [(float("inf"), []) for _ in word]
    max_length = max(map(len, vocabulary), default=0)
    for end in range(1, len(word) + 1):
        cost, parts = best[end - 1]
        best[end] = (cost + 1, parts + [(word[end - 1], False)])
        for start in range(max(0, end - max_length), end - 2):
            token = word[start:end]
            if token in vocabulary and best[start][0] + 0.5 < best[end][0]:
                best[end] = (best[start][0] + 0.5, best[start][1] + [(token, True)])
    parts = best[-1][1]
    recognized = [token for token, known in parts if known]
    if len(recognized) < 3 or sum(map(len, recognized)) < len(word) * 0.75:
        return word
    grouped = []
    for token, known in parts:
        if not known and grouped and not grouped[-1][1]:
            grouped[-1] = (grouped[-1][0] + token, False)
        else:
            grouped.append((token, known))
    return " ".join(token for token, _ in grouped)


def clean_ocr_results(
    results, english_words, alcohol_terms, number_to_letter, min_similarity=0.90
):
    """Add cleaned OCR text and count each applied token correction.

    Args:
        results (pd.DataFrame): OCR results containing extracted_text.
        english_words (set[str]): Vocabulary loaded once in the OCR section.
        alcohol_terms (set[str]): Additional alcohol-brand and drink terms.
        number_to_letter (dict[str, str]): Possible digit-to-letter substitutions.
        min_similarity (float): Minimum OCR-aware correction similarity.

    Returns:
        tuple: Copy of OCR results with cleaned_text, and a correction-count table.
    """
    vocabulary, word_groups = build_cleaning_vocabulary(english_words, alcohol_terms)
    confusions = {**LETTER_CONFUSIONS, **number_to_letter}
    segmentation_words = {
        word for term in alcohol_terms | BEVERAGE_WORDS | COMMON_TEXT_WORDS
        for word in re.findall(r"[^\W_]+(?:'[^\W_]+)*", normalize_ocr_text(term))
        if len(word) >= 3
    }
    phrase_pairs = set()
    for phrase in alcohol_terms | COMMON_PHRASES:
        tokens = normalize_ocr_text(phrase).split()
        phrase_pairs.update(zip(tokens, tokens[1:]))
    correction_cache = {}
    correction_counts = Counter()

    def replace_word(match):
        """Replace one token while recording every corrected occurrence.

        Args:
            match (re.Match): Token matched within normalized OCR text.

        Returns:
            str: Corrected token or its unchanged normalized form.
        """
        word = match.group()
        before = word_pattern.findall(match.string[:match.start()])
        after = word_pattern.findall(match.string[match.end():])
        previous_word = before[-1] if before else ""
        next_word = after[0] if after else ""
        cache_key = (word, previous_word, next_word)
        if cache_key not in correction_cache:
            corrected = correct_ocr_word(
                word, vocabulary, word_groups, confusions, min_similarity,
                previous_word, next_word, phrase_pairs,
            )
            correction_cache[cache_key] = split_merged_ocr_word(
                corrected, segmentation_words
            )
        corrected = correction_cache[cache_key]
        if corrected != word:
            correction_counts[(word, corrected)] += 1
        return corrected

    cleaned_results = results.copy()
    normalized_texts = results["extracted_text"].map(normalize_ocr_text)
    word_pattern = re.compile(r"[^\W_]+(?:'[^\W_]+)*")
    cleaned_results["cleaned_text"] = normalized_texts.map(
        lambda text: word_pattern.sub(replace_word, text)
    )
    corrections = pd.DataFrame(
        [
            {
                "original_token": original,
                "corrected_token": corrected,
                "occurrences": count,
            }
            for (original, corrected), count in sorted(
                correction_counts.items(), key=lambda item: (-item[1], item[0])
            )
        ],
        columns=["original_token", "corrected_token", "occurrences"],
    )
    return cleaned_results, corrections


def create_bert_embeddings(texts, model, batch_size=32):
    """Encode nonempty OCR text and preserve zero vectors for empty text.

    Args:
        texts (pd.Series): OCR text with missing values replaced by strings.
        model (SentenceTransformer): Initialized CPU sentence encoder.
        batch_size (int): Number of texts to encode in each batch.

    Returns:
        np.ndarray: Float32 feature matrix in input row order. Nonempty
            vectors have unit L2 norm; empty text has an all-zero vector.
    """
    embeddings = np.zeros(
        (len(texts), model.get_sentence_embedding_dimension()), dtype=np.float32
    )
    has_text = texts.str.strip().ne("").to_numpy()
    if has_text.any():
        embeddings[has_text] = model.encode(
            texts.loc[has_text].tolist(),
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
    return embeddings


def create_glove_embeddings(tokenized_texts, word_vectors, dimensions=100):
    """Average known GloVe words and normalize one vector per image.

    Args:
        tokenized_texts (pd.Series): Token lists in image row order.
        word_vectors (dict[str, np.ndarray]): Pretrained vectors for known words.
        dimensions (int): Number of features in each pretrained vector.

    Returns:
        tuple: Float32 embedding matrix and per-image token coverage DataFrame.
            Text with no known tokens retains an all-zero vector.
    """
    embeddings = np.zeros((len(tokenized_texts), dimensions), dtype=np.float32)
    token_counts = []
    matched_counts = []
    for row, tokens in enumerate(tokenized_texts):
        known_vectors = [word_vectors[token] for token in tokens if token in word_vectors]
        token_counts.append(len(tokens))
        matched_counts.append(len(known_vectors))
        if known_vectors:
            embeddings[row] = np.mean(known_vectors, axis=0)

    embeddings = normalize(embeddings, norm="l2", copy=False)
    coverage = pd.DataFrame(
        {"token_count": token_counts, "matched_token_count": matched_counts}
    )
    coverage["unknown_token_count"] = (
        coverage["token_count"] - coverage["matched_token_count"]
    )
    coverage["coverage_percent"] = (
        coverage["matched_token_count"]
        .div(coverage["token_count"].replace(0, np.nan))
        .fillna(0) * 100
    )
    coverage["vector_norm"] = np.linalg.norm(embeddings, axis=1)
    return embeddings, coverage


def load_input_images(input_dir):
    """List images directly inside an unlabeled input directory.

    Args:
        input_dir (Path): Directory containing images without class subfolders.

    Returns:
        pd.DataFrame: Sorted filenames and image paths.
    """
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    return pd.DataFrame(
        [
            {"file_name": path.name, "image_path": str(path)}
            for path in sorted(input_dir.iterdir())
            if path.is_file() and path.suffix.lower() in image_extensions
        ],
        columns=["file_name", "image_path"],
    )


def transform_texts(texts, vectorizer):
    """Transform cleaned text using the representation exported by training.

    Args:
        texts (pd.Series): Cleaned OCR text in image order.
        vectorizer: Fitted TF-IDF transformer, MiniLM encoder, or GloVe bundle.

    Returns:
        scipy.sparse.csr_matrix | np.ndarray: Features for the saved classifier.
    """
    if isinstance(vectorizer, dict) and vectorizer["embedding"] == "glove":
        tokens = texts.str.lower().str.findall(r"[a-z]+(?:'[a-z]+)*|[0-9]+")
        embeddings, _ = create_glove_embeddings(
            tokens, vectorizer["word_vectors"], vectorizer["dimensions"]
        )
        return embeddings
    if hasattr(vectorizer, "encode"):
        vectorizer.to("cpu")
        return create_bert_embeddings(texts, vectorizer)
    return vectorizer.transform(texts.tolist())


def predict_directory(input_dir, model_dir, output_path):
    """Load saved artifacts and classify images without training any models.

    Args:
        input_dir (Path): Unlabeled directory containing input images.
        model_dir (Path): Directory containing best_cls and best_vectorizer files.
        output_path (Path): Destination for the file_name,prediction CSV.

    Returns:
        pd.DataFrame: One filename and prediction per input image.
    """
    dataset = load_input_images(input_dir)
    results = dataset[["file_name"]].copy()
    if not dataset.empty:
        torch.set_num_threads(8)
        classifier = joblib.load(model_dir / "best_cls.joblib")
        vectorizer = joblib.load(model_dir / "best_vectorizer.joblib")
        english_words = load_english_vocabulary(BEVERAGE_WORDS)
        reader = create_ocr_reader(PROJECT_ROOT / ".cache" / "easyocr")
        ocr_results = extract_image_text(
            dataset, reader, english_words,
            min_english_ratio=MIN_ENGLISH_RATIO,
            min_confidence=MIN_OCR_CONFIDENCE,
        )
        cleaned_results, _ = clean_ocr_results(
            ocr_results, english_words, ALCOHOL_TERMS, NUMBER_TO_LETTER,
            min_similarity=MIN_WORD_SIMILARITY,
        )
        features = transform_texts(cleaned_results["cleaned_text"], vectorizer)
        predictions = classifier.predict(features)
        results["prediction"] = np.where(predictions == 1, "alcohol", "non_alcohol")
    else:
        results["prediction"] = pd.Series(dtype=str)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_path, index=False)
    return results


def main():
    """Parse paths, run inference, and print the saved prediction counts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_dir", type=Path, help="Directory of unlabeled images.")
    parser.add_argument(
        "--model-dir", type=Path, default=PROJECT_ROOT / "results",
        help="Directory containing best_cls.joblib and best_vectorizer.joblib.",
    )
    parser.add_argument(
        "--output", type=Path, default=PROJECT_ROOT / "results" / "results.csv",
        help="Output CSV path (default: results/results.csv).",
    )
    args = parser.parse_args()
    if not args.input_dir.is_dir():
        parser.error(f"Input directory does not exist: {args.input_dir}")
    results = predict_directory(args.input_dir, args.model_dir, args.output)
    print(f"Saved {len(results):,} predictions to {args.output}")
    if not results.empty:
        print(results["prediction"].value_counts().rename("images").to_frame())


if __name__ == "__main__":
    main()
