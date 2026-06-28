# Kepler Vetting with PyTorch

`kepler-vetting-pytorch` is a PyTorch project for exoplanet candidate vetting using public NASA Kepler data.

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
- Train and evaluate a tabular baseline.
- Train a 1D CNN on folded light curves.
- Compare tabular, light-curve, and fused models.
