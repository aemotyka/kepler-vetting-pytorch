from __future__ import annotations

import numpy as np
import pandas as pd

from kepler_vetting.processing.common import (
    DATASET_TAG,
    MODEL_READINESS_REPORT_PATH,
    MODEL_READY_MANIFEST_PATH,
    MODEL_READY_NPZ_PATH,
    PROCESSED_MANIFEST_PATH,
    PROCESSED_NPZ_PATH,
    PROCESSED_SUCCESSFUL_MANIFEST_PATH,
    RUN_METRICS_DIR,
)


MODEL_SUMMARY_FILES = [
    RUN_METRICS_DIR / "tabular_local_features_metrics_summary.csv",
    RUN_METRICS_DIR / "fused_local_model_metrics_summary.csv",
]


OUTPUT_PATH = RUN_METRICS_DIR / "dataset_variant_model_summary.csv"


def require_file(path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"missing required file: {path}")


def load_test_summaries() -> pd.DataFrame:
    frames = []

    for path in MODEL_SUMMARY_FILES:
        require_file(path)

        frame = pd.read_csv(path)
        frame = frame[frame["split"] == "test"].copy()
        frame["source_file"] = str(path)

        frames.append(frame)

    combined = pd.concat(frames, ignore_index=True)

    preferred_columns = [
        "model",
        "split",
        "accuracy_mean",
        "accuracy_std",
        "f1_mean",
        "f1_std",
        "roc_auc_mean",
        "roc_auc_std",
        "precision_mean",
        "recall_mean",
        "tn_mean",
        "fp_mean",
        "fn_mean",
        "tp_mean",
        "source_file",
    ]

    return combined[
        [
            column
            for column in preferred_columns
            if column in combined.columns
        ]
    ]


def main() -> None:
    require_file(PROCESSED_NPZ_PATH)
    require_file(PROCESSED_MANIFEST_PATH)
    require_file(PROCESSED_SUCCESSFUL_MANIFEST_PATH)
    require_file(MODEL_READINESS_REPORT_PATH)
    require_file(MODEL_READY_MANIFEST_PATH)
    require_file(MODEL_READY_NPZ_PATH)

    processed = np.load(PROCESSED_NPZ_PATH)
    ready = np.load(MODEL_READY_NPZ_PATH)

    processed_manifest = pd.read_csv(PROCESSED_MANIFEST_PATH)
    successful_manifest = pd.read_csv(PROCESSED_SUCCESSFUL_MANIFEST_PATH)
    readiness_report = pd.read_csv(MODEL_READINESS_REPORT_PATH)
    ready_manifest = pd.read_csv(MODEL_READY_MANIFEST_PATH)

    test_summary = load_test_summaries()

    RUN_METRICS_DIR.mkdir(parents=True, exist_ok=True)
    test_summary.to_csv(OUTPUT_PATH, index=False)

    print("dataset_tag:", DATASET_TAG or "(default)")
    print("metrics_dir:", RUN_METRICS_DIR)
    print("processed_dataset:", PROCESSED_NPZ_PATH)
    print("model_ready_dataset:", MODEL_READY_NPZ_PATH)
    print()
    print("processed_rows:", int(processed["labels"].shape[0]))
    print("processed_manifest_rows:", int(processed_manifest.shape[0]))
    print("successful_processed_manifest_rows:", int(successful_manifest.shape[0]))
    print("model_ready_rows:", int(ready["labels"].shape[0]))
    print("model_ready_manifest_rows:", int(ready_manifest.shape[0]))
    print("excluded_from_model_ready_rows:", int((~readiness_report["model_ready"]).sum()))
    print("processed_global_shape:", processed["global_view"].shape)
    print("processed_local_shape:", processed["local_view"].shape)
    print("model_ready_global_shape:", ready["global_view"].shape)
    print("model_ready_local_shape:", ready["local_view"].shape)
    print()
    print("model-ready label counts:")
    print(pd.Series(ready["labels"]).value_counts().sort_index())
    print()
    print("test-set model summary:")
    print(
        test_summary.to_string(
            index=False,
            float_format=lambda value: f"{value:.3f}",
        )
    )
    print()
    print("wrote:", OUTPUT_PATH)
    print("SUMMARIZE_DATASET_VARIANT_EXPERIMENT_OK")


if __name__ == "__main__":
    main()