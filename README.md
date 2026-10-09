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

The OCR section first checks for `ocr/ocr_text.csv` in the selected results folder.
If it exists, the notebook loads it and skips OCR model initialization and
extraction. Otherwise, OCR runs on all images, shows a progress bar, and saves
the CSV. Remove the saved CSV to rerun OCR after changing images or settings.
The first extraction downloads pretrained models into `.cache/easyocr/` and
needs internet access; later extractions reuse those models.

Dataset metadata, statistics, and sample figures are saved in `results/data/`.
Extracted text (`ocr_text.csv`), OCR summaries, and OCR plots are saved in
`results/ocr/`. Test mode uses the same subfolders inside `results_test/`.
OCR confidence and text coverage are descriptive statistics, not accuracy
measurements. The notebook also compares Logistic Regression, CPU XGBoost,
and Random Forest with each embedding type.

The embedding section consumes `ocr_results` from the OCR cells. It saves sparse
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
