from __future__ import annotations

import os
from pathlib import Path
import re
from typing import Any

import numpy as np
import pandas as pd
from astropy.io import fits


def configured_dataset_tag() -> str:
    tag = os.environ.get("KEPLER_VETTING_DATASET_TAG", "").strip()

    if not tag:
        return ""

    if not re.fullmatch(r"[A-Za-z0-9_.-]+", tag):
        raise ValueError(
            "KEPLER_VETTING_DATASET_TAG may only contain letters, numbers, "
            f"underscore, dash, or dot; got {tag!r}"
        )

    return tag


def env_path(name: str, default: Path) -> Path:
    raw_value = os.environ.get(name, "").strip()

    if raw_value:
        return Path(raw_value)

    return default


def tagged_file(directory: Path, filename: str) -> Path:
    path = Path(filename)

    if not DATASET_TAG:
        return directory / path.name

    return directory / f"{path.stem}_{DATASET_TAG}{path.suffix}"


DATASET_TAG = configured_dataset_tag()

RAW_LIGHTCURVE_ROOT = env_path(
    "KEPLER_VETTING_RAW_LIGHTCURVE_ROOT",
    Path("data/raw/lightcurves"),
)
PROCESSED_DIR = env_path(
    "KEPLER_VETTING_PROCESSED_DIR",
    Path("data/processed"),
)

MANIFEST_PATH = env_path(
    "KEPLER_VETTING_MANIFEST_PATH",
    tagged_file(Path("data/metadata"), "lightcurve_manifest.csv"),
)

PROCESSED_NPZ_PATH = tagged_file(
    PROCESSED_DIR,
    "kepler_q1_q17_dr25_sample.npz",
)
PROCESSED_MANIFEST_PATH = tagged_file(
    PROCESSED_DIR,
    "processed_manifest.csv",
)
PROCESSED_SUCCESSFUL_MANIFEST_PATH = tagged_file(
    PROCESSED_DIR,
    "processed_successful_manifest.csv",
)
MODEL_READINESS_REPORT_PATH = tagged_file(
    PROCESSED_DIR,
    "model_readiness_report.csv",
)
MODEL_READY_MANIFEST_PATH = tagged_file(
    PROCESSED_DIR,
    "model_ready_manifest.csv",
)
MODEL_READY_NPZ_PATH = tagged_file(
    PROCESSED_DIR,
    "kepler_q1_q17_dr25_model_ready.npz",
)
MODEL_READY_EXCLUDED_ROWS_PATH = tagged_file(
    PROCESSED_DIR,
    "model_ready_excluded_rows.csv",
)

RUN_METRICS_DIR = env_path(
    "KEPLER_VETTING_METRICS_DIR",
    Path("outputs/metrics") / DATASET_TAG if DATASET_TAG else Path("outputs/metrics"),
)
RUN_MODEL_DIR = env_path(
    "KEPLER_VETTING_MODEL_DIR",
    Path("artifacts/models") / DATASET_TAG if DATASET_TAG else Path("artifacts/models"),
)

MODEL_READY_MIN_CLEAN_POINTS = 5000
MODEL_READY_MIN_FITS_FILES = 2
MODEL_READY_MAX_GLOBAL_MISSING_FRACTION = 0.25
MODEL_READY_MAX_LOCAL_MISSING_FRACTION = 0.20

GLOBAL_BINS = 2001
LOCAL_BINS = 201
TRANSIT_BINS = 101
MAX_TRANSIT_WINDOWS = 32
FLUX_CLIP_ABS = 0.05

TABULAR_FEATURES = [
    "koi_period",
    "koi_duration",
    "koi_depth",
    "koi_prad",
    "koi_teq",
    "koi_insol",
    "koi_model_snr",
    "koi_steff",
    "koi_slogg",
    "koi_srad",
    "koi_kepmag",
]

LOG1P_FEATURES = {
    "koi_period",
    "koi_duration",
    "koi_depth",
    "koi_prad",
    "koi_insol",
    "koi_model_snr",
}

REQUIRED_MANIFEST_COLUMNS = [
    "kepid",
    "kepoi_name",
    "koi_disposition",
    "binary_label",
    "koi_time0bk",
    "selected_fits_urls",
    *TABULAR_FEATURES,
]


def kic_from_kepid(kepid: int) -> str:
    return f"{int(kepid):09d}"


def selected_filenames(row: pd.Series) -> list[str]:
    raw_urls = str(row.get("selected_fits_urls", "")).strip()

    if not raw_urls or raw_urls.lower() == "nan":
        return []

    filenames = []
    for url in raw_urls.split("|"):
        url = url.strip()
        if url:
            filenames.append(url.rsplit("/", 1)[-1])

    return filenames


def local_fits_paths_for_row(
    row: pd.Series,
    raw_root: Path = RAW_LIGHTCURVE_ROOT,
) -> tuple[list[Path], list[str]]:
    kepid = int(row["kepid"])
    kic = kic_from_kepid(kepid)
    target_dir = raw_root / kic

    expected_filenames = selected_filenames(row)

    if not expected_filenames:
        paths = sorted(target_dir.glob("*_llc.fits"))
        return paths, []

    paths = []
    missing = []

    for filename in expected_filenames:
        path = target_dir / filename
        if path.exists() and path.stat().st_size > 0:
            paths.append(path)
        else:
            missing.append(str(path))

    return paths, missing


def validate_manifest_columns(manifest: pd.DataFrame) -> None:
    missing = [col for col in REQUIRED_MANIFEST_COLUMNS if col not in manifest.columns]

    if missing:
        raise ValueError(f"manifest missing required columns: {missing}")

    labels = set(pd.to_numeric(manifest["binary_label"], errors="coerce").dropna().astype(int))
    if not labels.issubset({0, 1}):
        raise ValueError(f"binary_label must only contain 0/1 values; found {sorted(labels)}")


def read_clean_lightcurve_segment(path: Path) -> dict[str, Any]:
    with fits.open(path, memmap=False) as hdul:
        table = hdul[1].data
        columns = set(table.columns.names)

        required = {"TIME", "PDCSAP_FLUX", "SAP_QUALITY"}
        missing = sorted(required - columns)
        if missing:
            raise ValueError(f"{path} missing FITS columns: {missing}")

        time = np.asarray(table["TIME"], dtype=np.float64)
        flux = np.asarray(table["PDCSAP_FLUX"], dtype=np.float64)
        quality = np.asarray(table["SAP_QUALITY"], dtype=np.int64)

    raw_points = len(time)

    clean_mask = (
        np.isfinite(time)
        & np.isfinite(flux)
        & (quality == 0)
    )

    clean_time = time[clean_mask]
    clean_flux = flux[clean_mask]

    if clean_flux.size == 0:
        return {
            "time": np.array([], dtype=np.float64),
            "flux": np.array([], dtype=np.float64),
            "raw_points": raw_points,
            "clean_points": 0,
        }

    median_flux = np.nanmedian(clean_flux)

    if not np.isfinite(median_flux) or median_flux == 0:
        return {
            "time": np.array([], dtype=np.float64),
            "flux": np.array([], dtype=np.float64),
            "raw_points": raw_points,
            "clean_points": 0,
        }

    normalized_flux = clean_flux / median_flux - 1.0
    normalized_flux = np.clip(normalized_flux, -FLUX_CLIP_ABS, FLUX_CLIP_ABS)

    return {
        "time": clean_time,
        "flux": normalized_flux,
        "raw_points": raw_points,
        "clean_points": int(clean_time.size),
    }


def stitch_lightcurves(paths: list[Path]) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    all_time = []
    all_flux = []

    raw_points = 0
    clean_points = 0
    failed_files = []

    for path in paths:
        try:
            segment = read_clean_lightcurve_segment(path)
        except Exception as exc:
            failed_files.append(f"{path}: {type(exc).__name__}: {exc}")
            continue

        raw_points += int(segment["raw_points"])
        clean_points += int(segment["clean_points"])

        if segment["clean_points"] > 0:
            all_time.append(segment["time"])
            all_flux.append(segment["flux"])

    if not all_time:
        raise ValueError("no clean light-curve points found")

    time = np.concatenate(all_time)
    flux = np.concatenate(all_flux)

    sort_idx = np.argsort(time)
    time = time[sort_idx]
    flux = flux[sort_idx]

    time_span_days = float(np.nanmax(time) - np.nanmin(time)) if time.size else 0.0

    stats = {
        "n_fits_files": len(paths),
        "n_failed_fits_files": len(failed_files),
        "failed_fits_files": " | ".join(failed_files),
        "n_raw_points": raw_points,
        "n_clean_points": clean_points,
        "clean_fraction": clean_points / raw_points if raw_points else 0.0,
        "time_min": float(np.nanmin(time)),
        "time_max": float(np.nanmax(time)),
        "time_span_days": time_span_days,
    }

    return time, flux, stats


def phase_fold(time: np.ndarray, period: float, epoch: float) -> np.ndarray:
    if not np.isfinite(period) or period <= 0:
        raise ValueError(f"invalid period: {period}")

    if not np.isfinite(epoch):
        raise ValueError(f"invalid epoch: {epoch}")

    return ((time - epoch + 0.5 * period) % period) / period - 0.5


def median_bin(
    x: np.ndarray,
    y: np.ndarray,
    x_min: float,
    x_max: float,
    n_bins: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    if x_max <= x_min:
        raise ValueError(f"invalid bin range: {x_min} to {x_max}")

    mask = (
        np.isfinite(x)
        & np.isfinite(y)
        & (x >= x_min)
        & (x <= x_max)
    )

    x = x[mask]
    y = y[mask]

    if x.size == 0:
        raise ValueError(f"no points in bin range: {x_min} to {x_max}")

    edges = np.linspace(x_min, x_max, n_bins + 1, dtype=np.float64)
    centers = 0.5 * (edges[:-1] + edges[1:])

    bin_idx = np.searchsorted(edges, x, side="right") - 1
    bin_idx[x == x_max] = n_bins - 1

    valid_idx = (bin_idx >= 0) & (bin_idx < n_bins)
    bin_idx = bin_idx[valid_idx]
    y = y[valid_idx]

    values = np.full(n_bins, np.nan, dtype=np.float64)

    for idx in np.unique(bin_idx):
        values[idx] = np.nanmedian(y[bin_idx == idx])

    valid_bins = np.isfinite(values)
    missing_fraction = 1.0 - float(valid_bins.mean())

    if not valid_bins.any():
        raise ValueError("all bins are empty")

    if valid_bins.sum() == 1:
        values[~valid_bins] = values[valid_bins][0]
    else:
        values[~valid_bins] = np.interp(
            centers[~valid_bins],
            centers[valid_bins],
            values[valid_bins],
        )

    values = values - np.nanmedian(values)
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)

    return centers.astype(np.float32), values.astype(np.float32), missing_fraction


def local_window_half_width(period_days: float, duration_hours: float) -> float:
    if not np.isfinite(period_days) or period_days <= 0:
        return 0.05

    if not np.isfinite(duration_hours) or duration_hours <= 0:
        return 0.05

    duration_days = duration_hours / 24.0
    duration_phase = duration_days / period_days

    return float(min(0.5, max(0.05, 3.0 * duration_phase)))


def transform_tabular_features(rows: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    frame = pd.DataFrame(rows)

    columns = []
    medians = []
    means = []
    stds = []

    for feature in TABULAR_FEATURES:
        values = pd.to_numeric(frame[feature], errors="coerce").to_numpy(dtype=np.float64)

        if feature in LOG1P_FEATURES:
            values = np.where(values > 0, np.log1p(values), np.nan)

        median = np.nanmedian(values)
        if not np.isfinite(median):
            median = 0.0

        values = np.where(np.isfinite(values), values, median)

        mean = float(np.mean(values))
        std = float(np.std(values))

        if not np.isfinite(std) or std == 0:
            std = 1.0

        standardized = (values - mean) / std

        columns.append(standardized.astype(np.float32))
        medians.append(median)
        means.append(mean)
        stds.append(std)

    matrix = np.column_stack(columns).astype(np.float32)

    return {
        "matrix": matrix,
        "feature_names": np.array(TABULAR_FEATURES, dtype=str),
        "feature_medians": np.asarray(medians, dtype=np.float32),
        "feature_means": np.asarray(means, dtype=np.float32),
        "feature_stds": np.asarray(stds, dtype=np.float32),
    }