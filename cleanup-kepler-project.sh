#!/usr/bin/env bash
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

mkdir -p \
  docs \
  data/raw \
  data/metadata \
  data/interim \
  data/processed \
  notebooks \
  tests \
  src/kepler_vetting/data

touch src/kepler_vetting/__init__.py
touch src/kepler_vetting/data/__init__.py

if [ -d src/data ]; then
  for f in src/data/*.py; do
    [ -e "$f" ] || continue
    git mv "$f" "src/kepler_vetting/data/$(basename "$f")"
  done
  rmdir src/data 2>/dev/null || true
fi

python - <<'PY'
from pathlib import Path

Path("README.md").write_text("""# Kepler Vetting with PyTorch

This project uses public NASA Kepler data to build a PyTorch model for exoplanet candidate vetting.

The goal is to classify Kepler Objects of Interest as either planet-like signals or false positives. The labels and tabular candidate metadata come from the NASA Exoplanet Archive Q1-Q17 DR25 KOI table. The time-series inputs come from public MAST Kepler long-cadence FITS light curves.

The project starts with a small reproducible sample so the data path, FITS parsing, preprocessing, and training code can be developed without downloading a large fraction of the Kepler archive.

## Data

This repo does not commit downloaded data.

Local data is expected under:

- data/raw/
- data/metadata/
- data/interim/
- data/processed/

The current data path is:

NASA Exoplanet Archive KOI table -> KIC / kepid identifiers -> MAST public Kepler light-curve directories -> long-cadence FITS files -> cleaned and phase-folded tensors -> PyTorch models.

See docs/data_setup.md for the commands used to download and verify the initial sample.

## Current status

Working:

- Download KOI labels and metadata from NASA Exoplanet Archive TAP.
- Resolve KOI kepid values to public MAST Kepler light-curve directories.
- Download Kepler long-cadence FITS files.
- Open FITS files with astropy.
- Read TIME, SAP_FLUX, PDCSAP_FLUX, and SAP_QUALITY.

Next:

- Build preprocessing for stitched, cleaned, phase-folded light curves.
- Create train/validation/test splits grouped by kepid.
- Train a tabular baseline.
- Train a 1D CNN on folded light curves.
- Compare tabular, light-curve, and fused models.
""", encoding="utf-8")

Path("docs/data_setup.md").write_text("""# Data setup

This project uses two public NASA data sources:

1. NASA Exoplanet Archive KOI table: q1_q17_dr25_koi
2. MAST Kepler long-cadence FITS light curves

The FITS files are downloaded locally and are not committed to git.

## 1. Download KOI metadata

Run this from the repo root:

    mkdir -p data/raw

    python -c 'from urllib.parse import quote_plus; from urllib.request import urlretrieve; query = \"\"\"select kepid, kepoi_name, koi_disposition, koi_pdisposition, koi_period, koi_time0bk, koi_duration, koi_depth, koi_prad, koi_teq, koi_insol, koi_model_snr, koi_steff, koi_slogg, koi_srad, koi_kepmag from q1_q17_dr25_koi where koi_disposition in (\\\"CONFIRMED\\\", \\\"CANDIDATE\\\", \\\"FALSE POSITIVE\\\")\"\"\"; url = \"https://exoplanetarchive.ipac.caltech.edu/TAP/sync?query=\" + quote_plus(query) + \"&format=csv\"; urlretrieve(url, \"data/raw/koi_q1_q17_dr25.csv\"); print(\"wrote data/raw/koi_q1_q17_dr25.csv\")'

## 2. Smoke test one light curve

    python -m src.kepler_vetting.data.download_lightcurve_smoke_test

Expected result: one public Kepler FITS file downloads and opens with astropy.

## 3. Build a small manifest

    python -m src.kepler_vetting.data.build_lightcurve_manifest

The initial settings select 10 planet-like KOIs and 10 false-positive KOIs, with up to four FITS files per target.

## 4. Download the manifest FITS files

    python -m src.kepler_vetting.data.download_lightcurves_from_manifest

Some transient SSL/download failures can happen. Re-running the command skips existing files and retries missing files.

## 5. Verify local files

    python -c 'from pathlib import Path; import pandas as pd; manifest = pd.read_csv(\"data/metadata/lightcurve_manifest.csv\"); print(\"manifest shape:\", manifest.shape); print(); print(\"label counts:\"); print(manifest[\"binary_label\"].value_counts().sort_index()); print(); print(\"disposition counts:\"); print(manifest[\"koi_disposition\"].value_counts()); fits_files = sorted(Path(\"data/raw/lightcurves\").glob(\"*/*_llc.fits\")); print(); print(\"downloaded FITS files:\", len(fits_files)); print(\"first 5:\"); [print(\" \", path) for path in fits_files[:5]]'
""", encoding="utf-8")

Path("data/README.md").write_text("""# Local data directory

Downloaded and generated data lives here during local development.

The contents of raw/, metadata/, interim/, and processed/ are ignored by git. Recreate them with the scripts in src/kepler_vetting/data/ and the commands in docs/data_setup.md.
""", encoding="utf-8")

Path(".gitignore").write_text("""# Python
__pycache__/
*.py[cod]
.ipynb_checkpoints/
.pytest_cache/
.mypy_cache/
.ruff_cache/

# Virtual environments
.venv/
venv/
env/
.env

# Local data
data/raw/
data/interim/
data/processed/
data/metadata/*.csv
data/metadata/*.json

# Large/generated astronomy files
*.fits
*.fits.gz

# Model artifacts
artifacts/
checkpoints/
models/
outputs/
runs/
wandb/

# OS/editor
.DS_Store
.vscode/
.idea/
""", encoding="utf-8")

Path("requirements.txt").write_text("""astropy
matplotlib
numpy
pandas
scikit-learn
torch
tqdm
""", encoding="utf-8")
PY

git rm --cached data/metadata/lightcurve_manifest.csv 2>/dev/null || true

git status --short
