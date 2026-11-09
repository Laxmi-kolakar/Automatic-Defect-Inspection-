# Defect Doctor AI — Brake-disc inspection prototype

A local Streamlit hackathon prototype for traceable brake-disc image review. It includes inspection forms, batch uploads, a SQLite log, dataset explorer, CSV reports, and a small PyTorch CNN training baseline.

> **Safety and honesty:** without a trained checkpoint, all images return **REVIEW REQUIRED**. The app is not certified, is not a substitute for trained quality inspectors, and does not drive PLCs or automated rejection equipment. Do not claim accuracy until you have trained and evaluated on representative, held-out brake-disc images.

## 1. Requirements
- Windows 10/11, Python 3.11 recommended
- VS Code with Python extension
- Internet only for installing packages / downloading a dataset

## 2. Setup (Windows PowerShell)
Open this folder in VS Code, then open Terminal → New Terminal:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
Copy-Item .env.example .env
```

If PowerShell blocks activation, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`, then activate again. PyTorch/torchvision installation may need the matching command from https://pytorch.org/get-started/locally/ if pip reports a version conflict.

Run tests and app:

```powershell
pytest -q
streamlit run app.py
```

## 3. Folder structure

- `app.py`: all Streamlit navigation and UI pages, input validation, single/batch inspection flow, visual analytics, CSV export.
- `config/settings.py`: reads `.env`, resolves project/data/model paths, sets image size and confidence threshold.
- `core/image_utils.py`: safe image decoding and RGB/array conversion.
- `core/decisions.py`: conservative PASS/FAIL/REVIEW decision policy.
- `core/database.py`: SQLite schema and read/write helpers.
- `core/inference.py`: CNN architecture, checkpoint loading, inference. Missing/broken checkpoint means REVIEW REQUIRED.
- `scripts/download_dataset.py`: downloads a user-supplied direct ZIP URL and safely extracts it; you must verify source/license/labels yourself.
- `scripts/inspect_dataset.py`: reports image counts and unreadable files.
- `scripts/split_dataset.py`: splits a class-folder dataset into train/val/test.
- `scripts/train_classifier.py`: trains and evaluates the CNN, saves `models/brake_disc_cnn.pt` and `models/evaluation_metrics.json`.
- `pages/`: reserved for future Streamlit-native pages; this version uses a single navigation selector in `app.py` so every page works from one entry point.
- `data/dataset/`: prepared labelled images; excluded from Git to avoid committing large datasets.
- `models/`: generated model checkpoints and metrics; checkpoints are excluded from Git.
- `tests/`: small unit tests for image loading, decision rules, and model status.

## 4. Dataset preparation

Only use a dataset whose terms allow your intended use. Keep a note of the source URL, author, license, download date, class meanings, and any attribution requirements. A repository being public does **not** automatically mean its images are licensed for reuse.

Recommended folder layout:

```text
data/dataset/
  train/
    defective/
    normal/
  val/
    defective/
    normal/
  test/
    defective/
    normal/
```

Every image must be inside the folder matching its verified label. If your downloaded dataset has another layout, manually map labels after checking its documentation; do not guess label meaning.

The generic ZIP downloader accepts a direct ZIP file URL, not a GitHub repository webpage URL:

```powershell
python -m scripts.download_dataset "https://example.org/your-authorized-dataset.zip" --source-name brake-disc-source
```

That command is an example placeholder, not a real dataset URL. For a GitHub repository archive, use the repository's official archive URL after reviewing its license, then reorganize verified labelled images into the expected split/class folders. Inspect:

```powershell
python -m scripts.inspect_dataset
```

To split from `data/raw/<class-name>/*` into the required folders:

```powershell
python -m scripts.split_dataset --source data/raw --destination data/dataset --train 0.70 --val 0.15 --seed 42
```

This script performs a random image-level split. If images are near-duplicates, frames from the same video, or photos of the same physical part, group by part/source first and split groups to avoid leakage.

## 5. Train and evaluate

```powershell
python -m scripts.train_classifier --epochs 8 --batch-size 16 --lr 0.001
```

Training requires `torchvision`, which is part of the PyTorch installation. Checkpoint class names are taken from the dataset folder names. The current decision policy recognizes `normal`/`defective` and common aliases. If you use other class labels, update `core/decisions.py` deliberately after reviewing label semantics.

The script writes a test-set report to `models/evaluation_metrics.json`. Do not report these metrics without also stating dataset size, class balance, split method, and limitations. Tune the confidence threshold using validation data and calibration before any serious use; the default 0.80 is a prototype setting, not a validated threshold.

## 6. Dataset leads to investigate

These are leads, not endorsements; verify availability, exact labels, suitability for automotive brake discs, and license before use:
- GitHub project: https://github.com/susie1214/Vision_Inspection
- GitHub project: https://github.com/Palakurthy-Bharadwaj/Condition-Monitoring-of-brake-disc-images-using-CNN
- Roboflow Universe project: https://universe.roboflow.com/final-year-project-foz7e/discs-defect-detection-win2v
- KolektorSDD2 industrial surface defect dataset: https://www2.vicos.si/resources/kolektorsdd2/ (not brake-disc-specific; check license and permitted use)
- Thermal brake-disc fissure dataset: https://data.mendeley.com/datasets/gbffch5dmv/1 (different imaging context; do not mix with RGB images without a clear rationale)

Do not combine datasets with different class meanings or imaging conditions blindly. A model trained on generic metal defects may not generalize to brake discs.

## 7. Troubleshooting
- `ModuleNotFoundError`: activate `.venv`, then `pip install -r requirements.txt`.
- Streamlit command not found: use `python -m streamlit run app.py`.
- No model found: expected before training; app correctly shows REVIEW REQUIRED.
- Training says split missing: confirm `data/dataset/train`, `val`, and `test` each contain class folders and images.
- CUDA errors: use CPU build / CPU device; this small baseline can run slowly on CPU.
- Dataset download fails: confirm URL is a direct ZIP link and the source permits downloads.
- Model class mismatch: make class folder names match intended semantics and update the decision mapping if necessary.

## 8. Demo flow
1. Start app with `streamlit run app.py`.
2. Show Overview and explain model status.
3. Before training, upload a sample image and show the honest REVIEW REQUIRED behavior.
4. Show dataset counts and explain how labels were verified.
5. Train and evaluate only if the dataset is ready; show the generated evaluation report with limitations.
6. Inspect images, demonstrate traceability metadata, batch CSV export, history and analytics.
7. Explain that live camera integration, object localization, calibrated thresholds and safe automatic rejection are future validated engineering stages.
