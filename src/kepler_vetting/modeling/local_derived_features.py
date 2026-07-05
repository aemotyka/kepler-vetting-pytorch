from __future__ import annotations

import numpy as np
import pandas as pd


LOCAL_DERIVED_FEATURE_NAMES = [
    "local_min_flux",
    "local_argmin_phase",
    "local_argmin_abs_phase",
    "local_center_flux",
    "local_center_mean_11",
    "local_std",
    "local_flux_range",
    "local_left_mean",
    "local_right_mean",
    "local_left_right_asymmetry",
    "local_abs_left_right_asymmetry",
    "local_edge_mean",
    "local_center_vs_edge_depth",
    "local_min_vs_edge_depth",
]

QUALITY_FEATURE_NAMES = [
    "n_fits_files",
    "n_clean_points",
    "clean_fraction",
    "time_span_days",
    "global_missing_bin_fraction_before_interp",
    "local_missing_bin_fraction_before_interp",
    "local_window_half_width",
]


def ensure_phase_2d(
    local_phase: np.ndarray,
    n_rows: int,
) -> np.ndarray:
    local_phase = np.asarray(local_phase, dtype=np.float64)

    if local_phase.ndim == 1:
        return np.tile(local_phase[None, :], (n_rows, 1))

    if local_phase.ndim == 2:
        if local_phase.shape[0] != n_rows:
            raise ValueError(
                "local_phase row count does not match local_view: "
                f"{local_phase.shape[0]} vs {n_rows}"
            )

        return local_phase

    raise ValueError(f"local_phase must be 1D or 2D; got shape {local_phase.shape}")


def finite_mean(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]

    if values.size == 0:
        return float("nan")

    return float(values.mean())


def local_shape_features_for_row(
    phase: np.ndarray,
    flux: np.ndarray,
) -> list[float]:
    phase = np.asarray(phase, dtype=np.float64)
    flux = np.asarray(flux, dtype=np.float64)

    if phase.shape[0] != flux.shape[0]:
        raise ValueError(
            f"phase/flux length mismatch: {phase.shape[0]} vs {flux.shape[0]}"
        )

    finite_mask = np.isfinite(phase) & np.isfinite(flux)

    if finite_mask.sum() == 0:
        return [float("nan")] * len(LOCAL_DERIVED_FEATURE_NAMES)

    clean_phase = phase[finite_mask]
    clean_flux = flux[finite_mask]

    min_idx = int(np.argmin(clean_flux))
    min_flux = float(clean_flux[min_idx])
    argmin_phase = float(clean_phase[min_idx])
    argmin_abs_phase = abs(argmin_phase)

    center_idx = int(np.argmin(np.abs(clean_phase)))
    center_flux = float(clean_flux[center_idx])

    center_radius = max(2, int(round(clean_flux.shape[0] * 0.025)))
    center_start = max(0, center_idx - center_radius)
    center_end = min(clean_flux.shape[0], center_idx + center_radius + 1)
    center_values = clean_flux[center_start:center_end]
    center_mean = finite_mean(center_values)

    local_std = float(np.std(clean_flux))
    local_flux_range = float(np.max(clean_flux) - np.min(clean_flux))

    left_values = clean_flux[clean_phase < 0]
    right_values = clean_flux[clean_phase > 0]

    left_mean = finite_mean(left_values)
    right_mean = finite_mean(right_values)

    left_right_asymmetry = left_mean - right_mean
    abs_left_right_asymmetry = abs(left_right_asymmetry)

    edge_count = max(3, int(round(clean_flux.shape[0] * 0.10)))
    edge_values = np.concatenate(
        [
            clean_flux[:edge_count],
            clean_flux[-edge_count:],
        ]
    )
    edge_mean = finite_mean(edge_values)

    center_vs_edge_depth = edge_mean - center_mean
    min_vs_edge_depth = edge_mean - min_flux

    return [
        min_flux,
        argmin_phase,
        argmin_abs_phase,
        center_flux,
        center_mean,
        local_std,
        local_flux_range,
        left_mean,
        right_mean,
        left_right_asymmetry,
        abs_left_right_asymmetry,
        edge_mean,
        center_vs_edge_depth,
        min_vs_edge_depth,
    ]


def build_local_derived_feature_matrix(
    data: np.lib.npyio.NpzFile,
) -> tuple[np.ndarray, np.ndarray]:
    local_view = data["local_view"].astype(np.float64)
    local_phase = ensure_phase_2d(
        local_phase=data["local_phase"],
        n_rows=local_view.shape[0],
    )

    if local_view.ndim != 2:
        raise ValueError(f"local_view must be 2D; got shape {local_view.shape}")

    if local_phase.shape != local_view.shape:
        raise ValueError(
            "local_phase and local_view shapes differ: "
            f"{local_phase.shape} vs {local_view.shape}"
        )

    rows = [
        local_shape_features_for_row(
            phase=local_phase[row_idx],
            flux=local_view[row_idx],
        )
        for row_idx in range(local_view.shape[0])
    ]

    matrix = np.asarray(rows, dtype=np.float64)
    feature_names = np.asarray(LOCAL_DERIVED_FEATURE_NAMES, dtype=str)

    return matrix, feature_names


def build_quality_feature_matrix(
    manifest: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    missing = [
        feature for feature in QUALITY_FEATURE_NAMES if feature not in manifest.columns
    ]

    if missing:
        raise ValueError(f"model-ready manifest is missing quality fields: {missing}")

    matrix = (
        manifest[QUALITY_FEATURE_NAMES]
        .apply(pd.to_numeric, errors="coerce")
        .to_numpy(dtype=np.float64)
    )

    feature_names = np.asarray(QUALITY_FEATURE_NAMES, dtype=str)

    return matrix, feature_names


def combine_feature_blocks(
    blocks: list[tuple[np.ndarray, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray]:
    matrices = []
    feature_names = []

    n_rows = None

    for matrix, names in blocks:
        matrix = np.asarray(matrix, dtype=np.float64)
        names = np.asarray(names, dtype=str)

        if matrix.ndim != 2:
            raise ValueError(f"feature block must be 2D; got shape {matrix.shape}")

        if matrix.shape[1] != names.shape[0]:
            raise ValueError(
                "feature block column count does not match feature-name count: "
                f"{matrix.shape[1]} vs {names.shape[0]}"
            )

        if n_rows is None:
            n_rows = matrix.shape[0]
        elif matrix.shape[0] != n_rows:
            raise ValueError(
                f"feature block row count mismatch: {matrix.shape[0]} vs {n_rows}"
            )

        matrices.append(matrix)
        feature_names.extend(names.tolist())

    combined_matrix = np.column_stack(matrices)
    combined_feature_names = np.asarray(feature_names, dtype=str)

    return combined_matrix, combined_feature_names
