# Results

Final notes from the Kepler vetting modeling pass.

Final pick: `fused_tabular_local_cnn`, using the 80/10/10 grouped split and a fixed 0.5 threshold.

## Bottom line

The winner is the fused tabular + local phase-view CNN.

| Model | Threshold | Accuracy | F1 | ROC AUC | Precision | Recall |
|---|---:|---:|---:|---:|---:|---:|
| `fused_tabular_local_cnn` | 0.5 | 0.868 | 0.872 | 0.936 | 0.859 | 0.885 |

This is the model I would report. It has the best overall mix of accuracy, F1, AUC, precision, and recall. A few later runs improved one number, usually recall or threshold-tuned F1, but none of them clearly beat this model at the fixed threshold.

## Dataset and split

Dataset:

    data/processed/kepler_q1_q17_dr25_model_ready.npz

Dataset size:

| Item | Count |
|---|---:|
| Total rows | 5,799 |
| False positive | 2,862 |
| Candidate | 1,107 |
| Confirmed | 1,830 |
| Planet-like total, candidate + confirmed | 2,937 |

Binary label mapping:

| Disposition | Binary label |
|---|---:|
| `FALSE POSITIVE` | 0 |
| `CANDIDATE` | 1 |
| `CONFIRMED` | 1 |

Split:

    grouped_by_kepid_test0.10_val0.10

The split is grouped by `kepid`: about 80% train, 10% validation, and 10% test.

Split summary from the final run:

| Split | Rows | Groups | Negative | Positive | Positive rate |
|---|---:|---:|---:|---:|---:|
| Train | 4,639 | 4,639 | 2,287 | 2,352 | 0.507 |
| Val | 580 | 580 | 286 | 294 | 0.507 |
| Test | 580 | 580 | 289 | 291 | 0.502 |

## Main fixed-threshold comparison

All rows below use the same 80/10/10 grouped split and fixed threshold 0.5.

| Model | Accuracy | F1 | ROC AUC | Precision | Recall | Verdict |
|---|---:|---:|---:|---:|---:|---|
| `dummy_most_frequent` | 0.506 | 0.672 | 0.500 | 0.506 | 1.000 | Dummy baseline |
| `tabular_logistic_regression` | 0.837 | 0.849 | 0.886 | 0.799 | 0.905 | Strong tabular baseline |
| `tabular_local_features_logistic_regression` | 0.843 | 0.854 | 0.908 | 0.805 | 0.910 | Best non-neural baseline |
| `local_view_cnn` | 0.684 | 0.736 | 0.748 | 0.639 | 0.874 | Weak alone |
| `global_view_cnn` | 0.670 | 0.742 | 0.719 | 0.614 | 0.938 | Weak alone, high recall |
| `fused_tabular_local_cnn` | 0.868 | 0.872 | 0.936 | 0.859 | 0.885 | Final winner |
| `soft_label_fused_tabular_local_cnn` | 0.868 | 0.869 | 0.935 | 0.875 | 0.863 | More conservative; not better |
| `candidate_weighted_fused_tabular_local_cnn` | 0.867 | 0.869 | 0.936 | 0.867 | 0.871 | Did not beat fused |
| `three_class_fused_tabular_local_cnn` | 0.863 | 0.873 | 0.934 | 0.825 | 0.926 | High recall, too many false positives |
| `two_stage_rescue_gate` | 0.866 | 0.870 | 0.935 | 0.857 | 0.882 | Gate did not generalize |
| `fused_tabular_transit_set_cnn` | 0.865 | 0.867 | 0.935 | 0.864 | 0.871 | Did not beat fused local |
| `fused_tabular_local_transit_set_cnn` | 0.868 | 0.870 | 0.935 | 0.866 | 0.875 | Similar accuracy, lower F1/AUC |
| `rescue_stacked_logistic_regression` | 0.865 | 0.868 | 0.928 | 0.861 | 0.875 | Did not beat fused |
| `selective_rescue_rule_model` | 0.866 | 0.870 | 0.934 | 0.854 | 0.887 | Did not beat fused |

## Threshold-tuned observations

Threshold tuning produced a few rows worth keeping in the report, but I would not treat them as the final model. They depend on the validation-picked threshold and do not clearly beat the fixed-threshold fused model.

| Model | Tuned threshold | Accuracy | F1 | ROC AUC | Precision | Recall | Note |
|---|---:|---:|---:|---:|---:|---:|---|
| `soft_label_fused_tabular_local_cnn` | 0.409 | 0.869 | 0.874 | 0.935 | 0.851 | 0.899 | Best threshold-tuned F1 row |
| `fused_tabular_local_transit_set_cnn` | 0.451 | 0.869 | 0.873 | 0.935 | 0.858 | 0.889 | Slight threshold-tuned gain, not a fixed-threshold win |
| `rescue_stacked_logistic_regression` | 0.437 | 0.868 | 0.872 | 0.928 | 0.852 | 0.895 | F1 tied-ish, AUC much lower |
| `two_stage_rescue_gate` | 0.494 | 0.867 | 0.870 | 0.935 | 0.857 | 0.884 | Did not beat final fused model |

The threshold-tuned soft-label model is worth mentioning. It gets better F1 by lowering the threshold and taking on more recall. I would still use the fixed-threshold fused model as the main result.

## Pairwise test-set error comparison

The headline metrics are close, so the pairwise tables matter. The question is simple: does the new model fix more test examples than it breaks compared with the fused model?

Positive net gain means the right-hand model fixed more rows than it broke. Negative means it lost ground against the fused model.

| Pair | Variant | Left accuracy | Right accuracy | Net right-correct gain | Verdict |
|---|---|---:|---:|---:|---|
| `tabular_vs_fused` | fixed 0.5 | 0.837 | 0.868 | +182 | Fused clearly improves over tabular |
| `tabular_local_features_vs_fused` | fixed 0.5 | 0.843 | 0.868 | +149 | Fused clearly improves over tabular local features |
| `fused_vs_soft_label_fused_local` | fixed 0.5 | 0.868 | 0.868 | -1 | No fixed-threshold gain |
| `fused_vs_candidate_weighted_fused_local` | fixed 0.5 | 0.868 | 0.867 | -6 | Worse than fused |
| `fused_vs_three_class_fused_local` | fixed 0.5 | 0.868 | 0.863 | -28 | Worse than fused |
| `fused_vs_two_stage_rescue_gate` | fixed 0.5 | 0.868 | 0.866 | -12 | Worse than fused |
| `fused_vs_fused_transit_set` | fixed 0.5 | 0.868 | 0.865 | -18 | Worse than fused |
| `fused_vs_fused_local_transit_set` | fixed 0.5 | 0.868 | 0.868 | 0 | Similar, not better |
| `fused_vs_rescue_stacked` | fixed 0.5 | 0.868 | 0.865 | -17 | Worse than fused |
| `fused_vs_selective_rescue_rule` | fixed 0.5 | 0.868 | 0.866 | -12 | Worse than fused |

This backs up the final call: the extra modeling branches were useful, but they do not replace the fused tabular + local CNN.

## Experiment notes

### Tabular baselines

The tabular models are strong. The KOI table carries a lot of signal on its own.

`tabular_logistic_regression` achieved accuracy 0.837, F1 0.849, and ROC AUC 0.886.

Adding local light-curve summary features helped the tabular baseline. `tabular_local_features_logistic_regression` achieved accuracy 0.843, F1 0.854, and ROC AUC 0.908.

### CNN-only models

The CNN-only models were weaker than the tabular models.

The local-view CNN alone reached accuracy 0.684, F1 0.736, and ROC AUC 0.748.

The global-view CNN alone reached accuracy 0.670, F1 0.742, and ROC AUC 0.719.

They picked up some signal, especially on recall, but they were not competitive by themselves.

### Fused tabular + local model

The fused tabular + local CNN is the best model in this run.

It combines the KOI table with the local phase-folded light-curve view, and it beats both the tabular-only and CNN-only models.

Final fixed-threshold result:

| Metric | Value |
|---|---:|
| Accuracy | 0.868 |
| F1 | 0.872 |
| ROC AUC | 0.936 |
| Precision | 0.859 |
| Recall | 0.885 |

This is the final model.

### Transit-set and local+transit variants

The transit-set variants did not really improve things.

| Model | Accuracy | F1 | ROC AUC |
|---|---:|---:|---:|
| `fused_tabular_transit_set_cnn` | 0.865 | 0.867 | 0.935 |
| `fused_tabular_local_transit_set_cnn` | 0.868 | 0.870 | 0.935 |

The local+transit model tied the fused model on fixed-threshold accuracy, but had lower F1 and AUC. Pairwise net gain was 0, so there was no reason to switch.

### Soft-label candidate experiment

The soft-label experiment used this target mapping:

| Disposition | Training target |
|---|---:|
| `FALSE POSITIVE` | 0.0 |
| `CANDIDATE` | 0.7 |
| `CONFIRMED` | 1.0 |

Fixed-threshold result:

| Metric | Value |
|---|---:|
| Accuracy | 0.868 |
| F1 | 0.869 |
| ROC AUC | 0.935 |
| Precision | 0.875 |
| Recall | 0.863 |

This made the model more conservative. Precision went up, recall went down, and it did not beat the fused model.

The val-tuned result had the best F1 row:

| Metric | Value |
|---|---:|
| Threshold | 0.409 |
| Accuracy | 0.869 |
| F1 | 0.874 |
| ROC AUC | 0.935 |
| Precision | 0.851 |
| Recall | 0.899 |

This is useful to note, but the improvement depends on the tuned threshold. It is not a fixed-threshold win.

### Candidate-weighted hard-label experiment

The candidate-weighted experiment kept the binary target but downweighted candidate examples:

| Disposition | Target | Weight |
|---|---:|---:|
| `FALSE POSITIVE` | 0 | 1.0 |
| `CANDIDATE` | 1 | 0.7 |
| `CONFIRMED` | 1 | 1.0 |

Fixed-threshold result:

| Metric | Value |
|---|---:|
| Accuracy | 0.867 |
| F1 | 0.869 |
| ROC AUC | 0.936 |
| Precision | 0.867 |
| Recall | 0.871 |

This did not beat the final fused model.

### Three-class experiment

The three-class model used this target mapping:

| Disposition | Class |
|---|---:|
| `FALSE POSITIVE` | 0 |
| `CANDIDATE` | 1 |
| `CONFIRMED` | 2 |

For binary evaluation, candidate and confirmed were collapsed back into the positive class.

Fixed-threshold result:

| Metric | Value |
|---|---:|
| Accuracy | 0.863 |
| F1 | 0.873 |
| ROC AUC | 0.934 |
| Precision | 0.825 |
| Recall | 0.926 |

This model rescued a lot of positives, especially candidates, but it also created too many false positives. Good experiment, not the final model.

Pairwise against the final fused model:

| Pair | Fused accuracy | Three-class accuracy | Net gain |
|---|---:|---:|---:|
| `fused_vs_three_class_fused_local` | 0.868 | 0.863 | -28 |

### Rescue stacker and selective rescue rule

The rescue/meta-model path did not help.

| Model | Accuracy | F1 | ROC AUC |
|---|---:|---:|---:|
| `rescue_stacked_logistic_regression` | 0.865 | 0.868 | 0.928 |
| `selective_rescue_rule_model` | 0.866 | 0.870 | 0.934 |

Both are worse than the final fused model.

### Two-stage learned rescue/veto gate

The last rescue experiment trained a small logistic gate on validation disagreements. The fused model stayed as the base model, and the gate could rescue or veto using the other model scores.

Fixed-threshold result:

| Metric | Value |
|---|---:|
| Accuracy | 0.866 |
| F1 | 0.870 |
| ROC AUC | 0.935 |
| Precision | 0.857 |
| Recall | 0.882 |

Pairwise against the final fused model:

| Pair | Fused accuracy | Gate accuracy | Net gain |
|---|---:|---:|---:|
| `fused_vs_two_stage_rescue_gate` | 0.868 | 0.866 | -12 |

The gate looked better on validation than it did on test. Per-seed test gains were mixed:

| Seed | Test net gain vs fused |
|---:|---:|
| 0 | +3 |
| 1 | -13 |
| 2 | +13 |
| 3 | -3 |
| 4 | -1 |
| 5 | 0 |
| 6 | +2 |
| 7 | 0 |
| 8 | -12 |
| 9 | -1 |

So the learned gate is not part of the final model.

## Takeaway

The KOI table is strong, and the local light-curve shape adds useful extra signal. The best result comes from fusing those two inputs directly.

The later variants still told us something about the errors:

- Soft labels and candidate weighting moved the precision/recall tradeoff around, but did not improve the fixed-threshold model.
- The three-class model rescued many candidates and confirmed planets, but overpredicted positives.
- Rescue and gate models found some real fixes, but they did not generalize well enough to beat the fused model.

Final recommendation:

    Use fused_tabular_local_cnn with the 80/10/10 grouped split and fixed 0.5 threshold.

## Commands

Train the final model:

    caffeinate -dimsu env \
      KEPLER_VETTING_TEST_SIZE=0.10 \
      KEPLER_VETTING_VAL_SIZE=0.10 \
      KEPLER_VETTING_METRICS_DIR=outputs/metrics/split801010 \
      KEPLER_VETTING_MODEL_DIR=artifacts/models/split801010 \
      PYTHONPATH=src \
      python -m kepler_vetting.modeling.train_fused_local_model

Rebuild the comparison tables:

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

Primary result files:

| File | Purpose |
|---|---|
| `outputs/metrics/split801010/model_comparison.csv` | Summary comparison table |
| `outputs/metrics/split801010/model_comparison_by_seed.csv` | Per-seed comparison table |
| `outputs/metrics/split801010/pairwise_model_error_summary.csv` | Pairwise net-gain/error table |
| `outputs/metrics/split801010/pairwise_model_changed_predictions.csv` | Changed predictions from pairwise comparisons |
| `outputs/metrics/split801010/fused_local_model_metrics_summary.csv` | Final fused model summary |
| `outputs/metrics/split801010/fused_local_model_predictions.csv` | Final fused model predictions |