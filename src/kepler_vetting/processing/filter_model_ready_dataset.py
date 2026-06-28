from __future__ import annotations

import numpy as np
import pandas as pd

from kepler_vetting.processing.common import (
    MODEL_READY_MANIFEST_PATH,
    MODEL_READY_NPZ_PATH,
    MODEL_READINESS_REPORT_PATH,
    PROCESSED_NPZ_PATH,
)
from kepler_vetting.processing.model_readiness import build_model_readiness_report


def subset_array(name: str, array: np.ndarray, keep_mask: np.ndarray, n_rows: int) -> np.ndarray:
    if array.ndim >= 1 and array.shape[0] == n_rows:
        return array[keep_mask]

    return array


def main() -> None:
    if not PROCESSED_NPZ_PATH.exists():
        raise FileNotFoundError(f"missing processed dataset: {PROCESSED_NPZ_PATH}")

    report = build_model_readiness_report()
    report.to_csv(MODEL_READINESS_REPORT_PATH, index=False)

    keep_mask = report["model_ready"].to_numpy(dtype=bool)
    n_rows = len(report)
    kept_rows = int(keep_mask.sum())
    dropped_rows = int((~keep_mask).sum())

    if kept_rows == 0:
        raise RuntimeError("model-ready filter removed every row")

    data = np.load(PROCESSED_NPZ_PATH)
    output = {}

    for name in data.files:
        output[name] = subset_array(
            name=name,
            array=data[name],
            keep_mask=keep_mask,
            n_rows=n_rows,
        )

    labels = output["labels"]
    if set(labels.astype(int).tolist()) - {0, 1}:
        raise ValueError("filtered labels contain values outside 0/1")

    MODEL_READY_NPZ_PATH.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(MODEL_READY_NPZ_PATH, **output)

    model_ready_manifest = report[report["model_ready"]].reset_index(drop=True)
    model_ready_manifest.to_csv(MODEL_READY_MANIFEST_PATH, index=False)

    print("input_dataset:", PROCESSED_NPZ_PATH)
    print("output_dataset:", MODEL_READY_NPZ_PATH)
    print("readiness_report:", MODEL_READINESS_REPORT_PATH)
    print("model_ready_manifest:", MODEL_READY_MANIFEST_PATH)
    print("input_rows:", n_rows)
    print("kept_rows:", kept_rows)
    print("dropped_rows:", dropped_rows)
    print("global_view_shape:", output["global_view"].shape)
    print("local_view_shape:", output["local_view"].shape)
    print("tabular_features_shape:", output["tabular_features"].shape)
    print()
    print("label counts:")
    print(pd.Series(labels).value_counts().sort_index())

    print()
    print("disposition counts:")
    print(model_ready_manifest["koi_disposition"].value_counts())

    if dropped_rows:
        print()
        print("dropped rows:")
        print(
            report[~report["model_ready"]][
                [
                    "kepid",
                    "kepoi_name",
                    "koi_disposition",
                    "binary_label",
                    "n_fits_files",
                    "n_clean_points",
                    "global_missing_bin_fraction_before_interp",
                    "local_missing_bin_fraction_before_interp",
                    "model_ready_failure_reasons",
                ]
            ].to_string(index=False)
        )

    print()
    print("FILTER_MODEL_READY_DATASET_OK")


if __name__ == "__main__":
    main()