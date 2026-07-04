from __future__ import annotations

import numpy as np
import pandas as pd

from kepler_vetting.processing.common import (
    GLOBAL_BINS,
    LOCAL_BINS,
    MAX_TRANSIT_WINDOWS,
    PROCESSED_MANIFEST_PATH,
    PROCESSED_NPZ_PATH,
    PROCESSED_SUCCESSFUL_MANIFEST_PATH,
    TRANSIT_BINS,
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
    "local_phase_narrow",
    "local_view_narrow",
    "local_phase_wide",
    "local_view_wide",
    "local_window_half_width_narrow",
    "local_window_half_width_wide",
    "transit_view",
    "transit_mask",
    "transit_count",
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

    if not PROCESSED_SUCCESSFUL_MANIFEST_PATH.exists():
        raise FileNotFoundError(
            f"missing successful processed manifest: {PROCESSED_SUCCESSFUL_MANIFEST_PATH}"
        )

    data = np.load(PROCESSED_NPZ_PATH)
    attempted_manifest = pd.read_csv(PROCESSED_MANIFEST_PATH)
    manifest = pd.read_csv(PROCESSED_SUCCESSFUL_MANIFEST_PATH)

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
    
    if data["local_view_narrow"].shape != (n, LOCAL_BINS):
        raise ValueError(
            f"local_view_narrow has bad shape: {data['local_view_narrow'].shape}"
        )

    if data["local_view_wide"].shape != (n, LOCAL_BINS):
        raise ValueError(
            f"local_view_wide has bad shape: {data['local_view_wide'].shape}"
        )

    if data["local_phase_narrow"].shape != (n, LOCAL_BINS):
        raise ValueError(
            f"local_phase_narrow has bad shape: {data['local_phase_narrow'].shape}"
        )

    if data["local_phase_wide"].shape != (n, LOCAL_BINS):
        raise ValueError(
            f"local_phase_wide has bad shape: {data['local_phase_wide'].shape}"
        )

    if data["transit_view"].shape != (n, MAX_TRANSIT_WINDOWS, TRANSIT_BINS):
        raise ValueError(f"transit_view has bad shape: {data['transit_view'].shape}")

    if data["transit_mask"].shape != (n, MAX_TRANSIT_WINDOWS):
        raise ValueError(f"transit_mask has bad shape: {data['transit_mask'].shape}")

    if data["transit_count"].shape != (n,):
        raise ValueError(f"transit_count has bad shape: {data['transit_count'].shape}")

    transit_mask_counts = data["transit_mask"].sum(axis=1).astype(np.int64)

    if not np.array_equal(transit_mask_counts, data["transit_count"].astype(np.int64)):
        raise ValueError("transit_count does not match transit_mask row sums")

    if int(data["transit_count"].min()) <= 0:
        raise ValueError("every processed row must have at least one transit window")

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
    assert_finite("local_view_narrow", data["local_view_narrow"])
    assert_finite("local_view_wide", data["local_view_wide"])
    assert_finite("local_phase_narrow", data["local_phase_narrow"])
    assert_finite("local_phase_wide", data["local_phase_wide"])
    assert_finite("local_window_half_width_narrow", data["local_window_half_width_narrow"])
    assert_finite("local_window_half_width_wide", data["local_window_half_width_wide"])
    assert_finite("transit_view", data["transit_view"])
    assert_finite("transit_count", data["transit_count"])
    assert_finite("tabular_features", tabular_features)
    assert_finite("local_window_half_width", data["local_window_half_width"])
    assert_finite("feature_medians", data["feature_medians"])
    assert_finite("feature_means", data["feature_means"])
    assert_finite("feature_stds", data["feature_stds"])

    if "processed_ok" in manifest.columns:
        successful_mask = manifest["processed_ok"].astype(str).str.lower().isin(
            ["true", "1", "yes"]
        )

        if not successful_mask.all():
            raise ValueError(
                "processed_successful_manifest contains rows where processed_ok is false"
            )

    if len(manifest) != n:
        raise ValueError(
            "processed_successful_manifest rows do not match .npz rows: "
            f"{len(manifest)} vs {n}"
        )

    print("processed_dataset:", PROCESSED_NPZ_PATH)
    print("processed_attempted_manifest:", PROCESSED_MANIFEST_PATH)
    print("processed_successful_manifest:", PROCESSED_SUCCESSFUL_MANIFEST_PATH)
    print("attempted_manifest_rows:", len(attempted_manifest))
    print("successful_manifest_rows:", len(manifest))
    print("N:", n)
    print("global_view_shape:", global_view.shape)
    print("local_view_shape:", local_view.shape)
    print("local_view_narrow_shape:", data["local_view_narrow"].shape)
    print("local_view_wide_shape:", data["local_view_wide"].shape)
    print("transit_view_shape:", data["transit_view"].shape)
    print("transit_mask_shape:", data["transit_mask"].shape)
    print("transit_count_summary:")
    print(pd.Series(data["transit_count"]).describe())
    print("tabular_features_shape:", tabular_features.shape)
    print("feature_names:", data["feature_names"].tolist())
    print()
    print("label counts:")
    print(pd.Series(labels).value_counts().sort_index())
    print()
    print("clean point summary:")
    print(
        manifest[
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