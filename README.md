# Kepler Vetting with PyTorch

`kepler-vetting-pytorch` is a PyTorch project for exoplanet candidate vetting using public NASA Kepler data.

The project classifies Kepler Objects of Interest as either planet-like signals or false positives. Labels and tabular candidate metadata come from the NASA Exoplanet Archive Q1-Q17 DR25 KOI table. Time-series inputs come from public MAST Kepler long-cadence FITS light curves.

## Result

Final model:

    fused_tabular_local_cnn

Evaluation setup:

    split: grouped_by_kepid_test0.10_val0.10
    threshold: fixed 0.5

Final test metrics:

| Model | Accuracy | F1 | ROC AUC | Precision | Recall |
|---|---:|---:|---:|---:|---:|
| `fused_tabular_local_cnn` | 0.868 | 0.872 | 0.936 | 0.859 | 0.885 |

This is the model I would use. It had the best overall mix of accuracy, F1, ROC AUC, precision, and recall. I also tried soft labels, candidate weighting, a three-class target, rescue rules, and a learned gate. Those runs were useful for understanding the mistakes, but none of them clearly beat the fused model at the fixed threshold.

See:

- `docs/results.md` for the full results write-up.
- `notebooks/final_report.ipynb` for the notebook that rebuilds the final tables from saved CSVs.
- `docs/data_setup.md` for the data download/setup notes.

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

## Reproducing the final run

These are the commands for the final path only.

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

The notebook just reads the saved CSVs in `outputs/metrics/split801010/`. It does not train anything.

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

Old experiment trainers live here:

    src/kepler_vetting/modeling/experiments/

They are kept so the experiment history is still there, but they are not part of the final path.

That folder includes the soft-label, candidate-weighted, three-class, rescue, and learned-gate runs.

## Project status

Current state:

- Final model selected.
- Final results documented in `docs/results.md`.
- Final report notebook added.
- Failed and secondary experiment trainers archived.
- README points to the final path only.
