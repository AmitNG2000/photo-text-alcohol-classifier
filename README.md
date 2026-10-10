# Photo Text Alcohol Classifier

Initial exploration for the DV DS CV + NLP assignment. The notebook covers
image loading, basic statistics, English OCR extraction using EasyOCR, and
TF-IDF, MiniLM, and GloVe text embeddings on the CPU. Labels come from the
`images/alcohol/` and `images/non_alcohol/` folders.

## Run the notebook

Use Python 3.10 on Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
jupyter notebook data_exploration_ocr.ipynb
```

Select the virtual environment's Python kernel and run the cells from top to
bottom, starting in the repository root. The requirements include CPU-only
PyTorch wheels from the official PyTorch package index.

The OCR section applies EXIF orientation and keeps native-size color OCR as its
baseline. Enhanced retries use grayscale with English EasyOCR on the CPU. It orders detected boxes by
line and horizontal position and saves the selected box coordinates as JSON.
Coordinates refer to the selected preprocessing variant. Confidence is weighted
by each box's non-space character count, so short incidental detections do not
contribute as much as the main text.

If vocabulary coverage or confidence is below 0.80, OCR also tries a horizontal
mirror. Selection combines 45% confidence with 55% dictionary coverage. A retry
must improve this score by more than 0.05 while retaining at least 80% of the
recognized words. A mirror also needs positive dictionary coverage at least as
high as the current result. This protects readable original text from slightly more
confident mirrored gibberish while retaining support for genuinely mirrored text.
An empty result accepts newly detected text only when its quality score is at
least 0.20, limiting faint-noise false positives.

If the winner remains below either threshold, OCR tries contrast enhancement
with 2× detection magnification. Tilted detected boxes trigger an image deskew
retry using their median angle. Empty detections trigger -30° and +30° image
rotation retries. Enhanced recognition also checks 180° text-box rotations. Enhancements
must pass the same selection rule; they do not automatically replace native OCR.

Retry scoring and cleaning share CMUdict, the alcohol vocabulary from
`alcohol_lists.py`, and additional non-alcohol beverage words. Vocabulary coverage
and confidence describe extraction behavior; neither measures transcription
accuracy.

The notebook reuses `ocr/ocr_text.csv` only when a SHA-256 fingerprint matches
image contents, paths, labels, vocabulary, pipeline version, and OCR settings.
Older CSVs refresh automatically. Change `OCR_PIPELINE_VERSION` when changing
OCR logic rather than its configuration. The first extraction downloads models
into `.cache/easyocr/`; later extractions reuse them. No additional packages are
needed for the OCR improvements.

Dataset metadata and figures are saved under `results/data/`. OCR text,
summaries, retry diagnostics, and plots are saved under `results/ocr/`. Test mode
uses the same subfolders inside `results_test/`. The CSV preserves original,
mirrored, and enhanced attempt texts and scores, alongside the selected
orientation, preprocessing, and quality score.

The Clean Text section preserves `extracted_text` and adds `cleaned_text`.
Cleaning applies Unicode NFKC, lowercase, apostrophe normalization, and single
spaces. Recognized words, punctuation, and standalone numbers are preserved.
Unknown tokens are compared with nearby vocabulary words using OCR-aware edit
distance. Ordinary edits cost 1; common character confusions cost 0.25. Digit
mappings can have several alternatives, such as `1 → i/l` and `8 → b/s`.
A matching neighboring word in a common or beverage phrase adds 0.15 to a
candidate's score. Corrections require similarity of at least 0.90 and a lead of
at least 0.03 over the next candidate. Weak or ambiguous corrections stay unchanged.

Long unknown tokens can also recover spaces using common text and beverage
words. Segmentation requires at least three recognized words covering 75% of
a token of at least 16 characters. It preserves every original character and
does not guess missing letters.

Cleaning runs whenever its cells execute, independently of the OCR cache. It
saves `ocr_text_cleaned.csv`, cleaning summaries, correction counts, and a plot.
Changed-image totals include normalization; a separate count tracks spelling or
spacing changes. Correction counts include both replacements and segmented
tokens. Rerun downstream embedding and classification cells after changing OCR
or cleaning.

The OCR improvements were checked on 11 images, including eight manually
transcribed main captions. Results are saved in
`results/ocr/ocr_improvement_validation.csv`, with per-image character and word
error rates, a summary CSV, and a comparison plot. These selected examples are
regression checks, not a dataset-wide accuracy estimate. The validation files
are separate from the full-dataset CSVs; the next OCR run refreshes the older
full-dataset cache automatically.

The embedding section consumes `cleaned_text` from the cleaning cells. Its saved
metadata contains both original and cleaned text, with `has_text` based on the
cleaned text. Rerun embedding and classification cells to regenerate their
outputs after cleaning changes. The embedding section saves sparse word and
character TF-IDF features, fitted vectorizers, dense MiniLM and GloVe vectors, image
row mappings, summary tables, and plots under `results/embeddings/` (or
`results_test/embeddings/` when `test_mode=True`). Word TF-IDF outputs go in
`word_tfidf/`, character TF-IDF outputs go in `tfidf/`, and MiniLM outputs go in
`bert/`; shared image metadata stays
in the parent embeddings folder. MiniLM downloads its pinned
pretrained model into `.cache/sentence_transformers/` on first use.
GloVe uses `gensim.downloader` with `api.load("glove-wiki-gigaword-100")`.
Its download cache defaults to `.cache/glove/` and respects an existing
`GENSIM_DATA_DIR`. It averages known word vectors per image and L2-normalizes
nonzero rows; empty text or text without known words stays zero. Vectors,
coverage summaries, vocabulary tables, and plots are saved in `glove/`.
The silhouette comparison and all four classifiers include these vectors.

Combined word + character TF-IDF outputs go in `combined_tfidf/`. Its matrix
concatenates word columns first and character columns second, then L2-normalizes
each row. Both nonzero components contribute equally; an empty component stays
zero. The representation has up to 10,000 features, with 5,000 per component.
It saves both component vectorizers, a sparse matrix, feature and summary tables,
statistics plots, and PCA outputs in the same style as the separate TF-IDF types.

Word TF-IDF uses unigrams and bigrams from cleaned text, with the same
5,000-feature limit, sublinear term frequency, and L2 normalization as character
TF-IDF. Its default tokenizer keeps tokens of at least two word characters and
retains stop words and numbers. Both TF-IDF sections include feature statistics,
top-feature plots, and 2D PCA plots colored by label, with coordinates and
explained-variance statistics saved alongside them. Shared PCA helpers live in
Text Embedding's Utils subsection.

Classification compares character TF-IDF, word TF-IDF, combined word + character
TF-IDF, MiniLM, and GloVe using Logistic Regression, XGBoost, Random Forest, and MLPClassifier:
twenty runs on one shared stratified 70/30 train/test split. Both TF-IDF vectorizers are refitted on training text only;
held-out text is transformed with those fitted vectorizers. Combined features
are built from these same training-fitted components and normalized per row,
with word columns first in both training and test data. MiniLM and GloVe vectors
come from fixed pretrained models. Each classifier runs once per embedding type.
Reports include
per-class precision, recall, F1, support, accuracy, ROC AUC, and trapezoidal
PR AUC, using alcohol as the positive class.

Classification reports, models, predictions, curves, and the combined
comparison are saved under `RESULTS_DIR/classification/`, with separate
`tfidf/` (character), `word_tfidf/` (word), `combined_tfidf/` (word + character),
`bert/` (MiniLM), and `glove/` (GloVe) subfolders for each
classifier. Both training-fitted TF-IDF vectorizers and split metadata are saved
in the classification folder, including a combined-vectorizer dictionary with
`word` and `character` keys. Restore combined features by transforming with
those vectorizers, concatenating word first, and applying L2 normalization.
Word TF-IDF uses the same feature settings as its
embedding section; classification does not use the exploratory full-data matrix.
The 20-image test dataset provides only six held-out images, so its metrics
serve as workflow checks rather than reliable performance estimates.

The saved `ocr_smoke_test*` outputs cover six validation images (three from each
class). Running the notebook produces OCR results for the entire dataset.

MLPClassifier uses a 64-unit ReLU hidden layer with Adam and fixed settings. It
estimates performance with five stratified folds on the training partition,
refitting TF-IDF inside every fold. Pretrained MiniLM and GloVe vectors stay
fixed. Per-fold scores and mean/std summaries are saved under
`RESULTS_DIR/classification/mlp/`, along with final held-out reports, models,
predictions, and curves. The performance comparison includes all twenty
classifier/embedding combinations; MLP rows also include CV estimates.
