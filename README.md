# Photo Text Alcohol Classifier

Initial exploration for the DV DS CV + NLP assignment. The notebook covers
image loading, basic statistics, and English OCR extraction using EasyOCR on
the CPU. Labels come from the `images/alcohol/` and `images/non_alcohol/` folders.

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

The first OCR run downloads pretrained models into `.cache/easyocr/` and needs
internet access. Subsequent runs reuse these models. OCR runs on all images and
shows a progress bar.

Tables, extracted text (`ocr_text.csv`), and figures are saved in `results/`.
OCR confidence and text coverage are descriptive statistics, not accuracy
measurements. This initial notebook stops after text extraction.

The saved `ocr_smoke_test*` outputs cover six validation images (three from each
class). Running the notebook produces OCR results for the entire dataset.
