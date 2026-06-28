# Data setup

This project uses two public NASA data sources:

1. NASA Exoplanet Archive KOI table: q1_q17_dr25_koi
2. MAST Kepler long-cadence FITS light curves

The FITS files are downloaded locally and are not committed to git.

## 1. Download KOI metadata

Run this from the repo root:

    mkdir -p data/raw

    python -c 'from urllib.parse import quote_plus; from urllib.request import urlretrieve; query = """select kepid, kepoi_name, koi_disposition, koi_pdisposition, koi_period, koi_time0bk, koi_duration, koi_depth, koi_prad, koi_teq, koi_insol, koi_model_snr, koi_steff, koi_slogg, koi_srad, koi_kepmag from q1_q17_dr25_koi where koi_disposition in (\"CONFIRMED\", \"CANDIDATE\", \"FALSE POSITIVE\")"""; url = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync?query=" + quote_plus(query) + "&format=csv"; urlretrieve(url, "data/raw/koi_q1_q17_dr25.csv"); print("wrote data/raw/koi_q1_q17_dr25.csv")'

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

    python -c 'from pathlib import Path; import pandas as pd; manifest = pd.read_csv("data/metadata/lightcurve_manifest.csv"); print("manifest shape:", manifest.shape); print(); print("label counts:"); print(manifest["binary_label"].value_counts().sort_index()); print(); print("disposition counts:"); print(manifest["koi_disposition"].value_counts()); fits_files = sorted(Path("data/raw/lightcurves").glob("*/*_llc.fits")); print(); print("downloaded FITS files:", len(fits_files)); print("first 5:"); [print(" ", path) for path in fits_files[:5]]'
