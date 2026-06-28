from __future__ import annotations

import numpy as np
import pandas as pd

from kepler_vetting.processing.common import (
    GLOBAL_BINS,
    LOCAL_BINS,
    PROCESSED_MANIFEST_PATH,
    PROCESSED_NPZ_PATH,
)


REQUIRED_ARRAYS = [
    "global_phase",
    "global_view",
    "local_phase",
    "local_view",
    "tabular_features",
    "labels",
    "kepid",
    "kepoi_name",
    "disposition",
    "local_window_half_width",
    "feature_names",
    "feature_medians",
    "feature_means",
    "feature_stds",
]


def assert_finite(name: str, array: np.ndarray) -> None:
    if not np.isfinite(array).all():
        bad_count = int((~np.isfinite(array)).sum())
        raise ValueError(f"{name} contains {bad_count} non-finite values")


def main() -> None:
    if not PROCESSED_NPZ_PATH.exists():
        raise FileNotFoundError(f"missing processed dataset: {PROCESSED_NPZ_PATH}")

    if not PROCESSED_MANIFEST_PATH.exists():
        raise FileNotFoundError(f"missing processed manifest: {PROCESSED_MANIFEST_PATH}")

    data = np.load(PROCESSED_NPZ_PATH)
    manifest = pd.read_csv(PROCESSED_MANIFEST_PATH)

    missing = [name for name in REQUIRED_ARRAYS if name not in data.files]
    if missing:
        raise ValueError(f"processed .npz missing arrays: {missing}")

    global_view = data["global_view"]
    local_view = data["local_view"]
    tabular_features = data["tabular_features"]
    labels = data["labels"]

    n = labels.shape[0]

    if global_view.shape != (n, GLOBAL_BINS):
        raise ValueError(f"global_view has bad shape: {global_view.shape}")

    if local_view.shape != (n, LOCAL_BINS):
        raise ValueError(f"local_view has bad shape: {local_view.shape}")

    if data["local_phase"].shape != (n, LOCAL_BINS):
        raise ValueError(f"local_phase has bad shape: {data['local_phase'].shape}")

    if tabular_features.shape[0] != n:
        raise ValueError(
            "tabular_features row count does not match labels: "
            f"{tabular_features.shape[0]} vs {n}"
        )

    label_values = set(labels.astype(int).tolist())
    if not label_values.issubset({0, 1}):
        raise ValueError(f"labels must only contain 0/1 values; got {label_values}")

    assert_finite("global_view", global_view)
    assert_finite("local_view", local_view)
    assert_finite("local_phase", data["local_phase"])
    assert_finite("tabular_features", tabular_features)
    assert_finite("local_window_half_width", data["local_window_half_width"])
    assert_finite("feature_medians", data["feature_medians"])
    assert_finite("feature_means", data["feature_means"])
    assert_finite("feature_stds", data["feature_stds"])

    successful_manifest_rows = manifest[manifest["processed_ok"] == True]
    if len(successful_manifest_rows) != n:
        raise ValueError(
            "successful processed_manifest rows do not match .npz rows: "
            f"{len(successful_manifest_rows)} vs {n}"
        )

    print("processed_dataset:", PROCESSED_NPZ_PATH)
    print("processed_manifest:", PROCESSED_MANIFEST_PATH)
    print("N:", n)
    print("global_view_shape:", global_view.shape)
    print("local_view_shape:", local_view.shape)
    print("tabular_features_shape:", tabular_features.shape)
    print("feature_names:", data["feature_names"].tolist())
    print()
    print("label counts:")
    print(pd.Series(labels).value_counts().sort_index())
    print()
    print("clean point summary:")
    print(
        successful_manifest_rows[
            [
                "n_fits_files",
                "n_raw_points",
                "n_clean_points",
                "clean_fraction",
                "time_span_days",
                "global_missing_bin_fraction_before_interp",
                "local_missing_bin_fraction_before_interp",
            ]
        ].describe()
    )
    print()
    print("VALIDATE_PROCESSED_DATASET_OK")


if __name__ == "__main__":
    main()