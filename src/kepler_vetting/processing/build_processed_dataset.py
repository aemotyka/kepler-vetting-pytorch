from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from kepler_vetting.processing.common import (
    GLOBAL_BINS,
    MAX_TRANSIT_WINDOWS,
    LOCAL_BINS,
    MANIFEST_PATH,
    PROCESSED_DIR,
    PROCESSED_MANIFEST_PATH,
    PROCESSED_NPZ_PATH,
    PROCESSED_SUCCESSFUL_MANIFEST_PATH,
    TABULAR_FEATURES,
    TRANSIT_BINS,
    local_fits_paths_for_row,
    local_window_half_width,
    median_bin,
    phase_fold,
    stitch_lightcurves,
    transform_tabular_features,
    validate_manifest_columns,
)


def row_float(row: pd.Series, column: str) -> float:
    value = pd.to_numeric(row.get(column), errors="coerce")
    return float(value) if np.isfinite(value) else float("nan")

AUX_LOCAL_VIEW_SCALES = {
    "narrow": 0.5,
    "wide": 2.0,
}

MIN_AUX_LOCAL_HALF_WIDTH = 0.01


def scaled_local_half_width(base_half_width: float, scale: float) -> float:
    if not np.isfinite(base_half_width) or base_half_width <= 0:
        base_half_width = 0.05

    return float(min(0.5, max(MIN_AUX_LOCAL_HALF_WIDTH, base_half_width * scale)))


def empty_phase_bin(
    half_width: float,
    n_bins: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    edges = np.linspace(
        -half_width,
        half_width,
        n_bins + 1,
        dtype=np.float64,
    )
    centers = 0.5 * (edges[:-1] + edges[1:])
    values = np.zeros(n_bins, dtype=np.float32)

    return centers.astype(np.float32), values, 1.0


def median_bin_or_empty(
    phase: np.ndarray,
    flux: np.ndarray,
    half_width: float,
    n_bins: int = LOCAL_BINS,
) -> tuple[np.ndarray, np.ndarray, float]:
    try:
        return median_bin(
            phase,
            flux,
            x_min=-half_width,
            x_max=half_width,
            n_bins=n_bins,
        )
    except ValueError as exc:
        message = str(exc)

        if "no points in bin range" not in message and "all bins are empty" not in message:
            raise

        return empty_phase_bin(
            half_width=half_width,
            n_bins=n_bins,
        )


def evenly_spaced_indices(n_items: int, max_items: int) -> np.ndarray:
    if n_items <= max_items:
        return np.arange(n_items, dtype=int)

    return np.unique(
        np.linspace(
            0,
            n_items - 1,
            max_items,
            dtype=int,
        )
    )


def transit_centers_in_observed_window(
    time: np.ndarray,
    period: float,
    epoch: float,
    half_width_phase: float,
) -> np.ndarray:
    if not np.isfinite(period) or period <= 0:
        raise ValueError(f"invalid period for transit windows: {period}")

    if not np.isfinite(epoch):
        raise ValueError(f"invalid epoch for transit windows: {epoch}")

    if time.size == 0:
        return np.asarray([], dtype=np.float64)

    time_min = float(np.nanmin(time))
    time_max = float(np.nanmax(time))
    half_width_days = half_width_phase * period

    k_start = int(np.ceil((time_min - half_width_days - epoch) / period))
    k_end = int(np.floor((time_max + half_width_days - epoch) / period))

    if k_end < k_start:
        return np.asarray([], dtype=np.float64)

    k_values = np.arange(k_start, k_end + 1, dtype=np.int64)

    return epoch + k_values.astype(np.float64) * period


def build_transit_windows(
    time: np.ndarray,
    flux: np.ndarray,
    period: float,
    epoch: float,
    half_width_phase: float,
) -> dict[str, Any]:
    centers = transit_centers_in_observed_window(
        time=time,
        period=period,
        epoch=epoch,
        half_width_phase=half_width_phase,
    )

    candidate_count = int(centers.size)

    selected_indices = evenly_spaced_indices(
        n_items=candidate_count,
        max_items=MAX_TRANSIT_WINDOWS,
    )
    selected_centers = centers[selected_indices]

    transit_view = np.zeros(
        (
            MAX_TRANSIT_WINDOWS,
            TRANSIT_BINS,
        ),
        dtype=np.float32,
    )
    transit_mask = np.zeros(MAX_TRANSIT_WINDOWS, dtype=bool)
    transit_missing_fractions = []

    stored_count = 0

    for center in selected_centers:
        relative_phase = (time - center) / period

        in_window = (
            np.isfinite(relative_phase)
            & np.isfinite(flux)
            & (relative_phase >= -half_width_phase)
            & (relative_phase <= half_width_phase)
        )

        if not in_window.any():
            continue

        _, window, missing_fraction = median_bin_or_empty(
            phase=relative_phase,
            flux=flux,
            half_width=half_width_phase,
            n_bins=TRANSIT_BINS,
        )

        transit_view[stored_count] = window
        transit_mask[stored_count] = True
        transit_missing_fractions.append(float(missing_fraction))
        stored_count += 1

        if stored_count >= MAX_TRANSIT_WINDOWS:
            break

    if stored_count == 0:
        raise ValueError("no per-transit windows with clean points")

    missing_values = np.asarray(transit_missing_fractions, dtype=np.float64)

    return {
        "transit_view": transit_view,
        "transit_mask": transit_mask,
        "transit_count": stored_count,
        "transit_candidate_count": candidate_count,
        "transit_selected_center_count": int(selected_centers.size),
        "transit_missing_bin_fraction_mean": float(np.mean(missing_values)),
        "transit_missing_bin_fraction_max": float(np.max(missing_values)),
    }

def process_row(row: pd.Series) -> tuple[dict[str, Any], dict[str, Any] | None]:
    kepid = int(row["kepid"])
    kepoi_name = str(row["kepoi_name"])

    output_record = {
        "kepid": kepid,
        "kepoi_name": kepoi_name,
        "binary_label": int(row["binary_label"]),
        "koi_disposition": row["koi_disposition"],
        "n_local_fits_files": 0,
        "n_missing_fits_files": 0,
        "n_fits_files": 0,
        "n_failed_fits_files": 0,
        "n_raw_points": 0,
        "n_clean_points": 0,
        "clean_fraction": 0.0,
        "time_min": np.nan,
        "time_max": np.nan,
        "time_span_days": 0.0,
        "global_missing_bin_fraction_before_interp": np.nan,
        "local_missing_bin_fraction_before_interp": np.nan,
        "local_window_half_width": np.nan,
        "local_window_half_width_narrow": np.nan,
        "local_window_half_width_wide": np.nan,
        "local_missing_bin_fraction_before_interp_narrow": np.nan,
        "local_missing_bin_fraction_before_interp_wide": np.nan,
        "transit_candidate_count": 0,
        "transit_selected_center_count": 0,
        "transit_count": 0,
        "transit_missing_bin_fraction_mean": np.nan,
        "transit_missing_bin_fraction_max": np.nan,
        "processed_ok": False,
        "failure_reason": "",
    }

    paths, missing = local_fits_paths_for_row(row)

    output_record["n_local_fits_files"] = len(paths)
    output_record["n_missing_fits_files"] = len(missing)

    if not paths:
        output_record["failure_reason"] = "no local FITS files found"
        return output_record, None

    try:
        time, flux, stats = stitch_lightcurves(paths)

        period = row_float(row, "koi_period")
        epoch = row_float(row, "koi_time0bk")
        duration_hours = row_float(row, "koi_duration")

        phase = phase_fold(time, period=period, epoch=epoch)

        global_phase, global_view, global_missing_fraction = median_bin(
            phase,
            flux,
            x_min=-0.5,
            x_max=0.5,
            n_bins=GLOBAL_BINS,
        )

        half_width = local_window_half_width(
            period_days=period,
            duration_hours=duration_hours,
        )

        local_phase, local_view, local_missing_fraction = median_bin(
            phase,
            flux,
            x_min=-half_width,
            x_max=half_width,
            n_bins=LOCAL_BINS,
        )

        aux_local_payload: dict[str, Any] = {}
        aux_local_record: dict[str, float] = {}

        for scale_name, scale in AUX_LOCAL_VIEW_SCALES.items():
            scaled_half_width = scaled_local_half_width(
                base_half_width=half_width,
                scale=scale,
            )

            scaled_phase, scaled_view, scaled_missing_fraction = median_bin_or_empty(
                phase=phase,
                flux=flux,
                half_width=scaled_half_width,
            )

            aux_local_payload[f"local_phase_{scale_name}"] = scaled_phase
            aux_local_payload[f"local_view_{scale_name}"] = scaled_view
            aux_local_payload[f"local_window_half_width_{scale_name}"] = scaled_half_width

            aux_local_record[f"local_window_half_width_{scale_name}"] = scaled_half_width
            aux_local_record[
                f"local_missing_bin_fraction_before_interp_{scale_name}"
            ] = scaled_missing_fraction

        transit_payload = build_transit_windows(
            time=time,
            flux=flux,
            period=period,
            epoch=epoch,
            half_width_phase=half_width,
        )

        transit_record = {
            "transit_candidate_count": transit_payload["transit_candidate_count"],
            "transit_selected_center_count": transit_payload[
                "transit_selected_center_count"
            ],
            "transit_count": transit_payload["transit_count"],
            "transit_missing_bin_fraction_mean": transit_payload[
                "transit_missing_bin_fraction_mean"
            ],
            "transit_missing_bin_fraction_max": transit_payload[
                "transit_missing_bin_fraction_max"
            ],
        }

        output_record.update(
            {
                "n_fits_files": stats["n_fits_files"],
                "n_failed_fits_files": stats["n_failed_fits_files"],
                "n_raw_points": stats["n_raw_points"],
                "n_clean_points": stats["n_clean_points"],
                "clean_fraction": stats["clean_fraction"],
                "time_min": stats["time_min"],
                "time_max": stats["time_max"],
                "time_span_days": stats["time_span_days"],
                "global_missing_bin_fraction_before_interp": global_missing_fraction,
                "local_missing_bin_fraction_before_interp": local_missing_fraction,
                "local_window_half_width": half_width,
                **aux_local_record,
                **transit_record,
                "processed_ok": True,
            }
        )

        feature_values = {
            feature: row_float(row, feature)
            for feature in TABULAR_FEATURES
        }

        processed = {
            "kepid": kepid,
            "kepoi_name": kepoi_name,
            "disposition": str(row["koi_disposition"]),
            "label": int(row["binary_label"]),
            "global_phase": global_phase,
            "global_view": global_view,
            "local_phase": local_phase,
            "local_view": local_view,
            "local_window_half_width": half_width,
            **aux_local_payload,
            "transit_view": transit_payload["transit_view"],
            "transit_mask": transit_payload["transit_mask"],
            "transit_count": transit_payload["transit_count"],
            "tabular_raw": feature_values,
        }

        return output_record, processed

    except Exception as exc:
        output_record["failure_reason"] = f"{type(exc).__name__}: {exc}"
        return output_record, None


def main() -> None:
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(f"missing manifest: {MANIFEST_PATH}")

    manifest = pd.read_csv(MANIFEST_PATH)
    validate_manifest_columns(manifest)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    processed_records = []
    processed_payloads = []

    for _, row in manifest.iterrows():
        record, payload = process_row(row)
        processed_records.append(record)

        status = "ok" if record["processed_ok"] else "failed"
        print(
            f"{status}: kepid={record['kepid']} "
            f"koi={record['kepoi_name']} "
            f"label={record['binary_label']} "
            f"clean_points={record['n_clean_points']} "
            f"reason={record['failure_reason']}"
        )

        if payload is not None:
            processed_payloads.append(payload)

    processed_manifest = pd.DataFrame(processed_records)
    successful_processed_manifest = (
        processed_manifest[processed_manifest["processed_ok"]]
        .reset_index(drop=True)
    )

    processed_manifest.to_csv(PROCESSED_MANIFEST_PATH, index=False)
    successful_processed_manifest.to_csv(PROCESSED_SUCCESSFUL_MANIFEST_PATH, index=False)

    if not processed_payloads:
        raise RuntimeError("no KOIs processed successfully")

    if len(successful_processed_manifest) != len(processed_payloads):
        raise RuntimeError(
            "successful processed manifest rows do not match processed payloads: "
            f"{len(successful_processed_manifest)} vs {len(processed_payloads)}"
        )

    tabular_rows = [payload["tabular_raw"] for payload in processed_payloads]
    tabular = transform_tabular_features(tabular_rows)

    global_phase = processed_payloads[0]["global_phase"]
    global_view = np.stack(
        [payload["global_view"] for payload in processed_payloads]
    ).astype(np.float32)

    local_phase = np.stack(
        [payload["local_phase"] for payload in processed_payloads]
    ).astype(np.float32)

    local_view = np.stack(
        [payload["local_view"] for payload in processed_payloads]
    ).astype(np.float32)

    local_phase_narrow = np.stack(
        [payload["local_phase_narrow"] for payload in processed_payloads]
    ).astype(np.float32)

    local_view_narrow = np.stack(
        [payload["local_view_narrow"] for payload in processed_payloads]
    ).astype(np.float32)

    local_phase_wide = np.stack(
        [payload["local_phase_wide"] for payload in processed_payloads]
    ).astype(np.float32)

    local_view_wide = np.stack(
        [payload["local_view_wide"] for payload in processed_payloads]
    ).astype(np.float32)

    labels = np.asarray(
        [payload["label"] for payload in processed_payloads],
        dtype=np.int64,
    )

    kepid = np.asarray(
        [payload["kepid"] for payload in processed_payloads],
        dtype=np.int64,
    )

    kepoi_name = np.asarray(
        [payload["kepoi_name"] for payload in processed_payloads],
        dtype=str,
    )

    disposition = np.asarray(
        [payload["disposition"] for payload in processed_payloads],
        dtype=str,
    )

    local_window_half_width = np.asarray(
        [payload["local_window_half_width"] for payload in processed_payloads],
        dtype=np.float32,
    )

    local_window_half_width_narrow = np.asarray(
        [payload["local_window_half_width_narrow"] for payload in processed_payloads],
        dtype=np.float32,
    )

    local_window_half_width_wide = np.asarray(
        [payload["local_window_half_width_wide"] for payload in processed_payloads],
        dtype=np.float32,
    )

    transit_view = np.stack(
        [payload["transit_view"] for payload in processed_payloads]
    ).astype(np.float32)

    transit_mask = np.stack(
        [payload["transit_mask"] for payload in processed_payloads]
    ).astype(bool)

    transit_count = np.asarray(
        [payload["transit_count"] for payload in processed_payloads],
        dtype=np.int64,
    )

    np.savez_compressed(
        PROCESSED_NPZ_PATH,
        global_phase=global_phase.astype(np.float32),
        global_view=global_view,
        local_phase=local_phase,
        local_view=local_view,
        local_phase_narrow=local_phase_narrow,
        local_view_narrow=local_view_narrow,
        local_phase_wide=local_phase_wide,
        local_view_wide=local_view_wide,
        transit_view=transit_view,
        transit_mask=transit_mask,
        transit_count=transit_count,
        tabular_features=tabular["matrix"],
        labels=labels,
        kepid=kepid,
        kepoi_name=kepoi_name,
        disposition=disposition,
        local_window_half_width=local_window_half_width,
        local_window_half_width_narrow=local_window_half_width_narrow,
        local_window_half_width_wide=local_window_half_width_wide,
        feature_names=tabular["feature_names"],
        feature_medians=tabular["feature_medians"],
        feature_means=tabular["feature_means"],
        feature_stds=tabular["feature_stds"],
    )

    print()
    print("wrote:", PROCESSED_NPZ_PATH)
    print("wrote:", PROCESSED_MANIFEST_PATH)
    print("wrote:", PROCESSED_SUCCESSFUL_MANIFEST_PATH)
    print("successful_rows:", len(processed_payloads))
    print("failed_rows:", int((~processed_manifest["processed_ok"]).sum()))
    print("global_view_shape:", global_view.shape)
    print("local_view_shape:", local_view.shape)
    print("local_view_narrow_shape:", local_view_narrow.shape)
    print("local_view_wide_shape:", local_view_wide.shape)
    print("transit_view_shape:", transit_view.shape)
    print("transit_mask_shape:", transit_mask.shape)
    print("transit_count_summary:")
    print(pd.Series(transit_count).describe())
    print("tabular_features_shape:", tabular["matrix"].shape)
    print("label_counts:")
    print(pd.Series(labels).value_counts().sort_index())


if __name__ == "__main__":
    main()