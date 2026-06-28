from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd

from kepler_vetting.processing.common import (
    MODEL_READY_MAX_GLOBAL_MISSING_FRACTION,
    MODEL_READY_MAX_LOCAL_MISSING_FRACTION,
    MODEL_READY_MIN_CLEAN_POINTS,
    MODEL_READY_MIN_FITS_FILES,
    PROCESSED_MANIFEST_PATH,
    PROCESSED_NPZ_PATH,
)


def processed_ok_mask(frame: pd.DataFrame) -> pd.Series:
    return frame["processed_ok"].astype(str).str.lower().isin(["true", "1", "yes"])


def readiness_reasons(row: pd.Series) -> list[str]:
    reasons = []

    if int(row["n_clean_points"]) < MODEL_READY_MIN_CLEAN_POINTS:
        reasons.append(
            f"n_clean_points<{MODEL_READY_MIN_CLEAN_POINTS}"
        )

    if int(row["n_fits_files"]) < MODEL_READY_MIN_FITS_FILES:
        reasons.append(
            f"n_fits_files<{MODEL_READY_MIN_FITS_FILES}"
        )

    if (
        float(row["global_missing_bin_fraction_before_interp"])
        > MODEL_READY_MAX_GLOBAL_MISSING_FRACTION
    ):
        reasons.append(
            "global_missing_bin_fraction_before_interp>"
            f"{MODEL_READY_MAX_GLOBAL_MISSING_FRACTION}"
        )

    if (
        float(row["local_missing_bin_fraction_before_interp"])
        > MODEL_READY_MAX_LOCAL_MISSING_FRACTION
    ):
        reasons.append(
            "local_missing_bin_fraction_before_interp>"
            f"{MODEL_READY_MAX_LOCAL_MISSING_FRACTION}"
        )

    return reasons


def build_model_readiness_report() -> pd.DataFrame:
    if not PROCESSED_NPZ_PATH.exists():
        raise FileNotFoundError(f"missing processed dataset: {PROCESSED_NPZ_PATH}")

    if not PROCESSED_MANIFEST_PATH.exists():
        raise FileNotFoundError(f"missing processed manifest: {PROCESSED_MANIFEST_PATH}")

    data = np.load(PROCESSED_NPZ_PATH)
    processed_manifest = pd.read_csv(PROCESSED_MANIFEST_PATH)

    successful = processed_manifest[processed_ok_mask(processed_manifest)].reset_index(drop=True)

    labels = data["labels"]
    if len(successful) != len(labels):
        raise ValueError(
            "processed manifest successful rows do not match .npz labels: "
            f"{len(successful)} vs {len(labels)}"
        )

    global_phase = data["global_phase"]
    global_view = data["global_view"]
    local_phase = data["local_phase"]
    local_view = data["local_view"]

    if global_view.shape[0] != len(successful):
        raise ValueError(
            "global_view rows do not match processed manifest: "
            f"{global_view.shape[0]} vs {len(successful)}"
        )

    if local_view.shape[0] != len(successful):
        raise ValueError(
            "local_view rows do not match processed manifest: "
            f"{local_view.shape[0]} vs {len(successful)}"
        )

    rows = []

    for idx, row in successful.iterrows():
        global_min_idx = int(np.argmin(global_view[idx]))
        local_min_idx = int(np.argmin(local_view[idx]))

        global_min_phase = float(global_phase[global_min_idx])
        local_min_phase = float(local_phase[idx, local_min_idx])

        record = {
            "processed_row_index": idx,
            "kepid": int(row["kepid"]),
            "kepoi_name": row["kepoi_name"],
            "koi_disposition": row["koi_disposition"],
            "binary_label": int(row["binary_label"]),
            "n_fits_files": int(row["n_fits_files"]),
            "n_raw_points": int(row["n_raw_points"]),
            "n_clean_points": int(row["n_clean_points"]),
            "clean_fraction": float(row["clean_fraction"]),
            "time_span_days": float(row["time_span_days"]),
            "global_missing_bin_fraction_before_interp": float(
                row["global_missing_bin_fraction_before_interp"]
            ),
            "local_missing_bin_fraction_before_interp": float(
                row["local_missing_bin_fraction_before_interp"]
            ),
            "local_window_half_width": float(row["local_window_half_width"]),
            "global_min_phase": global_min_phase,
            "global_min_abs_phase": abs(global_min_phase),
            "global_min_flux": float(global_view[idx, global_min_idx]),
            "local_min_phase": local_min_phase,
            "local_min_abs_phase": abs(local_min_phase),
            "local_min_flux": float(local_view[idx, local_min_idx]),
        }

        reasons = readiness_reasons(pd.Series(record))
        record["model_ready"] = len(reasons) == 0
        record["model_ready_failure_reasons"] = " | ".join(reasons)

        rows.append(record)

    return pd.DataFrame(rows)


def reason_counts(report: pd.DataFrame) -> Counter:
    counts: Counter = Counter()

    failed = report[~report["model_ready"]]
    for raw_reasons in failed["model_ready_failure_reasons"]:
        for reason in str(raw_reasons).split("|"):
            reason = reason.strip()
            if reason:
                counts[reason] += 1

    return counts