# Photo Text Alcohol Classifier

Initial exploration for the DV DS CV + NLP assignment. The notebook covers
image loading, basic statistics, English OCR extraction using EasyOCR, and
TF-IDF and MiniLM text embeddings on the CPU. Labels come from the
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

The OCR section runs on each EXIF-oriented image first. If fewer than 80% of
word occurrences are recognized English words, or mean text-box confidence is
below 0.80, it retries OCR on a horizontal mirror. The attempt with higher mean
confidence wins; ties retain the original. Missing confidence ranks below valid
confidence. The selected text is kept even if it remains below either threshold.

English-word recognition uses the bundled `cmudict` dictionary plus an editable
beverage/brand vocabulary in the OCR section. Lookup ignores case, punctuation,
and purely numeric tokens, and normalizes apostrophes. Mixed letter/number tokens
and words outside the vocabulary count as unrecognized; empty text scores zero.
This dictionary check approximates English quality and does not correct spelling
or split merged words. OCR confidence remains the mean over nonempty text boxes.

The notebook reuses `ocr/ocr_text.csv` in the selected results folder only when
all retry fields exist. Older CSVs are automatically refreshed on the next run;
updated CSVs skip OCR initialization and extraction on later runs. Remove the
saved CSV to rerun OCR after changing images, thresholds, or vocabulary.
The first extraction downloads pretrained models into `.cache/easyocr/` and
needs internet access; later extractions reuse those models.

Dataset metadata, statistics, and sample figures are saved in `results/data/`.
Extracted text (`ocr_text.csv`), OCR summaries, and OCR plots are saved in
`results/ocr/`. Test mode uses the same subfolders inside `results_test/`.
The OCR CSV includes the selected English ratio and orientation, a retry flag,
and each attempt's text, English ratio, and mean confidence. Unattempted flipped
metrics are missing. Retry totals, comparison tables, and confidence plots are
saved alongside the existing summaries. Rerun the downstream embedding and
classification cells to use the selected text.
OCR confidence and text coverage are descriptive statistics, not accuracy
measurements. The notebook also compares Logistic Regression, CPU XGBoost,
and Random Forest with each embedding type.

The Clean Text section adds `cleaned_text` while preserving `extracted_text`.
Its separate configuration cell contains the 90% similarity threshold and
digit-to-letter mapping, and imports the union of `ALCOHOL_BRANDS` and
`ALCOHOL_DRINKS` from `alcohol_lists.py`. Edit that module to change the alcohol
vocabulary; the lists are not copied into the notebook. Cleaning applies Unicode NFKC,
lowercase, apostrophe normalization, and single spaces. It preserves recognized
words, punctuation, and standalone numbers. Unknown words with digits first try
the configured letter substitutions; an exact English or brand match is accepted.
Otherwise, `SequenceMatcher` compares the original and converted candidates
against English words and brand tokens. A unique highest-scoring match of at
least 90% replaces the token; tied or unmatched tokens stay unchanged. Multiword
brands contribute their component words; cleaning does not split merged words.

Cleaning runs again whenever its cells are executed so configuration changes
take effect without rerunning OCR. It saves `ocr_text_cleaned.csv` (all OCR
columns plus `cleaned_text`), `text_cleaning_summary.csv`, correction counts, and
a correction plot in the selected OCR results folder. Changed-image totals
include normalization-only changes; corrected-token totals count word replacements.

The embedding section consumes `cleaned_text` from the cleaning cells. Its saved
metadata contains both original and cleaned text, with `has_text` based on the
cleaned text. Rerun embedding and classification cells to regenerate their
outputs after cleaning changes. The embedding section saves sparse
character TF-IDF features, the fitted vectorizer, dense MiniLM vectors, image
row mappings, summary tables, and plots under `results/embeddings/` (or
`results_test/embeddings/` when `test_mode=True`). TF-IDF outputs go in the
`tfidf/` subfolder and MiniLM outputs go in `bert/`; shared image metadata stays
in the parent embeddings folder. MiniLM downloads its pinned
pretrained model into `.cache/sentence_transformers/` on first use.
TF-IDF also includes a 2D PCA scatter plot colored by label, saved as
`tfidf/tfidf_pca.png`, with coordinates and explained-variance statistics
saved alongside it.

Classification uses a shared stratified 70/30 train/test split. TF-IDF is
refitted on training text only; MiniLM vectors come from the fixed pretrained
encoder. Each classifier runs once per embedding type. Reports include
per-class precision, recall, F1, support, accuracy, ROC AUC, and trapezoidal
PR AUC, using alcohol as the positive class.

Classification reports, models, predictions, curves, and the combined
comparison are saved under `RESULTS_DIR/classification/`, with separate
`tfidf/` and `bert/` subfolders for each classifier. The training-fitted
TF-IDF vectorizer and split metadata are saved in the classification folder.
The 20-image test dataset provides only six held-out images, so its metrics
serve as workflow checks rather than reliable performance estimates.

The saved `ocr_smoke_test*` outputs cover six validation images (three from each
class). Running the notebook produces OCR results for the entire dataset.
