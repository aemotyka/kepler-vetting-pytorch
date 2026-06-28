# Data setup

This project uses two public NASA data sources:

1. NASA Exoplanet Archive KOI table: q1_q17_dr25_koi
2. MAST Kepler long-cadence FITS light curves

The FITS files are downloaded locally and are not committed to git.

## 1. Download KOI metadata

Run this from the repo root:

    PYTHONPATH=src python -m kepler_vetting.data.download_koi_table

## 2. Smoke test one light curve

    PYTHONPATH=src python -m kepler_vetting.data.download_lightcurve_smoke_test

Expected result: one public Kepler FITS file downloads and opens with astropy.

## Estimate scale before larger downloads

Before increasing the manifest target count, estimate the number of targets, selected FITS files, and projected disk use:

    PYTHONPATH=src python -m kepler_vetting.data.estimate_lightcurve_scale --limit-per-class all --max-files-per-target 4

This does not download FITS files.

It writes:

    outputs/data/lightcurve_scale_estimate.csv
    outputs/data/lightcurve_scale_summary.csv

Use `--skip-file-sizes` for a faster directory-count-only estimate.

## 3. Build a small manifest

    PYTHONPATH=src python -m kepler_vetting.data.build_lightcurve_manifest

The initial settings select 10 planet-like KOIs and 10 false-positive KOIs, with up to four FITS files per target.

## 4. Download the manifest FITS files

    PYTHONPATH=src python -m kepler_vetting.data.download_lightcurves_from_manifest

Some transient SSL/download failures can happen. Re-running the command skips existing files and retries missing files.

## 5. Verify local files

    python - <<'EOF'
    from pathlib import Path
    import pandas as pd

    manifest = pd.read_csv("data/metadata/lightcurve_manifest.csv")

    print("manifest shape:", manifest.shape)
    print()
    print("label counts:")
    print(manifest["binary_label"].value_counts().sort_index())
    print()
    print("disposition counts:")
    print(manifest["koi_disposition"].value_counts())

    fits_files = sorted(Path("data/raw/lightcurves").glob("*/*_llc.fits"))
    print()
    print("downloaded FITS files:", len(fits_files))
    print("first 5:")
    for path in fits_files[:5]:
        print(" ", path)
    EOF
