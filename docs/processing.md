# Processing

This stage converts downloaded Kepler FITS light curves into fixed-size arrays for PyTorch.

## Inputs

- `data/metadata/lightcurve_manifest.csv`
- `data/raw/lightcurves/<zero-padded-kepid>/*_llc.fits`

## Outputs

- `data/processed/kepler_q1_q17_dr25_sample.npz`
- `data/processed/processed_manifest.csv`
- `outputs/figures/examples/*.png`
- `data/processed/model_readiness_report.csv`
- `data/processed/model_ready_manifest.csv`
- `data/processed/kepler_q1_q17_dr25_model_ready.npz`

These generated outputs are local artifacts and are not committed to git.

## Commands

Inspect the manifest and local FITS availability:

    PYTHONPATH=src python -m kepler_vetting.processing.inspect_manifest

Build the processed dataset:

    PYTHONPATH=src python -m kepler_vetting.processing.build_processed_dataset

Validate the processed dataset:

    PYTHONPATH=src python -m kepler_vetting.processing.validate_processed_dataset

Diagnose model readiness:

    PYTHONPATH=src python -m kepler_vetting.processing.diagnose_model_readiness

Create the filtered model-ready dataset:

    PYTHONPATH=src python -m kepler_vetting.processing.filter_model_ready_dataset

Plot example folded light curves:

    PYTHONPATH=src python -m kepler_vetting.processing.plot_processed_examples

## Processing logic

For each KOI:

1. Load selected local Kepler long-cadence FITS files.
2. Read `TIME`, `PDCSAP_FLUX`, and `SAP_QUALITY`.
3. Keep finite points with `SAP_QUALITY == 0`.
4. Normalize each FITS segment by median flux.
5. Stitch the clean segments together.
6. Phase-fold by `koi_period` and `koi_time0bk`.
7. Create a full-orbit global view with 2001 bins.
8. Create a transit-centered local view with 201 bins.
9. Save light-curve tensors, tabular features, labels, identifiers, and processing metadata.

The initial dataset is intentionally small: 20 KOIs, balanced between planet-like and false-positive labels.