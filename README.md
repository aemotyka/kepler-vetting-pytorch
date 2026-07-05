# Kepler Vetting with PyTorch

`kepler-vetting-pytorch` is a PyTorch project for exoplanet candidate vetting using public NASA Kepler data.

The project classifies Kepler Objects of Interest as either planet-like signals or false positives. Labels and tabular candidate metadata come from the NASA Exoplanet Archive Q1-Q17 DR25 KOI table. Time-series inputs come from public MAST Kepler long-cadence FITS light curves.

## Final result

The final selected model is:

    fused_tabular_local_cnn

Evaluation setup:

    split: grouped_by_kepid_test0.10_val0.10
    threshold: fixed 0.5

Final test metrics:

| Model | Accuracy | F1 | ROC AUC | Precision | Recall |
|---|---:|---:|---:|---:|---:|
| `fused_tabular_local_cnn` | 0.868 | 0.872 | 0.936 | 0.859 | 0.885 |

The fused tabular + local phase-view CNN is the final model because it produced the best practical balance of accuracy, F1, ROC AUC, precision, and recall. Later soft-label, candidate-weighted, three-class, rescue, and learned-gate experiments were useful diagnostics, but none beat the fixed-threshold fused model cleanly.

See:

- `docs/results.md` for the full final result write-up.
- `notebooks/final_report.ipynb` for the report-only notebook that reads existing CSV outputs.
- `docs/data_setup.md` for data download and setup notes.

## Data

This repo does not commit downloaded data.

Local data is expected under:

- `data/raw/`
- `data/metadata/`
- `data/interim/`
- `data/processed/`

Primary model-ready dataset:

    data/processed/kepler_q1_q17_dr25_model_ready.npz

Current data path:

    NASA Exoplanet Archive KOI table
    -> KIC / kepid identifiers
    -> MAST public Kepler light-curve directories
    -> long-cadence FITS files
    -> cleaned and phase-folded tensors
    -> PyTorch models

## Final reproduction path

The commands below reproduce the final selected model path and comparison outputs.

### 1. Train the final model

    caffeinate -dimsu env \
      KEPLER_VETTING_TEST_SIZE=0.10 \
      KEPLER_VETTING_VAL_SIZE=0.10 \
      KEPLER_VETTING_METRICS_DIR=outputs/metrics/split801010 \
      KEPLER_VETTING_MODEL_DIR=artifacts/models/split801010 \
      PYTHONPATH=src \
      python -m kepler_vetting.modeling.train_fused_local_model

### 2. Rebuild final comparison tables

    env \
      KEPLER_VETTING_TEST_SIZE=0.10 \
      KEPLER_VETTING_VAL_SIZE=0.10 \
      KEPLER_VETTING_METRICS_DIR=outputs/metrics/split801010 \
      KEPLER_VETTING_MODEL_DIR=artifacts/models/split801010 \
      PYTHONPATH=src \
      python -m kepler_vetting.modeling.compare_model_metrics

    env \
      KEPLER_VETTING_TEST_SIZE=0.10 \
      KEPLER_VETTING_VAL_SIZE=0.10 \
      KEPLER_VETTING_METRICS_DIR=outputs/metrics/split801010 \
      KEPLER_VETTING_MODEL_DIR=artifacts/models/split801010 \
      PYTHONPATH=src \
      python -m kepler_vetting.modeling.compare_pairwise_model_errors --top-n 12

### 3. Open the final report notebook

    notebooks/final_report.ipynb

The notebook is report-only. It reads existing CSV outputs from `outputs/metrics/split801010/` and does not train models.

## Key output files

| File | Purpose |
|---|---|
| `outputs/metrics/split801010/model_comparison.csv` | Final model comparison summary |
| `outputs/metrics/split801010/model_comparison_by_seed.csv` | Per-seed model comparison |
| `outputs/metrics/split801010/pairwise_model_error_summary.csv` | Pairwise error/net-gain summary |
| `outputs/metrics/split801010/pairwise_model_changed_predictions.csv` | Changed prediction details |
| `outputs/metrics/split801010/fused_local_model_metrics_summary.csv` | Final model metrics summary |
| `outputs/metrics/split801010/fused_local_model_predictions.csv` | Final model predictions |

## Archived experiments

Failed or secondary trainer scripts are archived under:

    src/kepler_vetting/modeling/experiments/

Those modules are kept for reproducibility and experiment history, but they are not part of the final recommended training path.

Archived branches include soft-label targets, candidate weighting, three-class modeling, rescue stackers, selective rescue rules, and learned rescue/veto gates.

## Project status

Closeout state:

- Final model selected.
- Final results documented in `docs/results.md`.
- Final report notebook added.
- Failed and secondary experiment trainers archived.
- README now points to the final path only.
