from __future__ import annotations

import argparse
import math
import random
import re
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin
from urllib.request import Request, urlopen

import pandas as pd
from tqdm import tqdm


KOI_PATH = Path("data/raw/koi_q1_q17_dr25.csv")
OUT_DIR = Path("outputs/data")
DEFAULT_OUT_PATH = OUT_DIR / "lightcurve_scale_estimate.csv"
DEFAULT_SUMMARY_PATH = OUT_DIR / "lightcurve_scale_summary.csv"

MAX_DIRECTORY_ATTEMPTS = 4
MAX_HEAD_ATTEMPTS = 3
RETRY_SLEEP_SECONDS = 5
RANDOM_SEED = 42

LABEL_MAP = {
    "CONFIRMED": 1,
    "CANDIDATE": 1,
    "FALSE POSITIVE": 0,
}

REQUIRED_COLUMNS = [
    "kepid",
    "kepoi_name",
    "koi_disposition",
    "koi_period",
    "koi_time0bk",
]


def parse_limit_per_class(value: str) -> int | None:
    normalized = value.strip().lower()

    if normalized == "all":
        return None

    parsed = int(normalized)

    if parsed <= 0:
        raise argparse.ArgumentTypeError(
            f"--limit-per-class must be positive or 'all'; got {value}"
        )

    return parsed


def bytes_to_gib(value: int | float) -> float:
    return float(value) / (1024.0**3)


def kepler_lightcurve_dir(kepid: int) -> str:
    kic = f"{int(kepid):09d}"
    return f"https://archive.stsci.edu/pub/kepler/lightcurves/{kic[:4]}/{kic}/"


def read_url_text(url: str) -> str:
    req = Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0",
        },
    )

    with urlopen(req, timeout=90) as response:
        return response.read().decode("utf-8", errors="replace")


def read_url_text_with_retries(url: str) -> str:
    last_error = None

    for attempt in range(1, MAX_DIRECTORY_ATTEMPTS + 1):
        try:
            return read_url_text(url)
        except Exception as exc:
            last_error = exc

            if attempt < MAX_DIRECTORY_ATTEMPTS:
                time.sleep(RETRY_SLEEP_SECONDS)

    raise RuntimeError(
        f"failed after {MAX_DIRECTORY_ATTEMPTS} attempts for {url}: {last_error}"
    )


def list_long_cadence_fits(directory_url: str) -> list[str]:
    html = read_url_text_with_retries(directory_url)
    hrefs = re.findall(r'href=["\']([^"\']+)["\']', html)

    return sorted(
        urljoin(directory_url, href) for href in hrefs if href.endswith("_llc.fits")
    )


def head_content_length(url: str) -> tuple[int | None, str]:
    last_error = None

    for attempt in range(1, MAX_HEAD_ATTEMPTS + 1):
        try:
            req = Request(
                url,
                method="HEAD",
                headers={
                    "User-Agent": "Mozilla/5.0",
                },
            )

            with urlopen(req, timeout=90) as response:
                content_length = response.headers.get("Content-Length")

                if content_length is None:
                    return None, "missing_content_length"

                parsed = int(content_length)

                if parsed < 0:
                    return None, "negative_content_length"

                return parsed, "ok"

        except HTTPError as exc:
            last_error = exc

            if exc.code in {403, 405}:
                return None, f"head_not_allowed_http_{exc.code}"

        except (URLError, TimeoutError, OSError, ValueError) as exc:
            last_error = exc

        if attempt < MAX_HEAD_ATTEMPTS:
            time.sleep(RETRY_SLEEP_SECONDS)

    return None, f"head_failed_{type(last_error).__name__}"


def load_candidate_rows(limit_per_class: int | None) -> list[dict]:
    if not KOI_PATH.exists():
        raise FileNotFoundError(f"missing KOI CSV: {KOI_PATH}")

    df = pd.read_csv(KOI_PATH)

    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]

    if missing:
        raise ValueError(f"KOI table is missing required columns: {missing}")

    df = df.dropna(subset=REQUIRED_COLUMNS)
    df = df[df["koi_disposition"].isin(LABEL_MAP)].copy()

    df["kepid"] = df["kepid"].astype(int)
    df["binary_label"] = df["koi_disposition"].map(LABEL_MAP).astype(int)

    rng = random.Random(RANDOM_SEED)

    positive_rows = df[df["binary_label"] == 1].to_dict("records")
    negative_rows = df[df["binary_label"] == 0].to_dict("records")

    rng.shuffle(positive_rows)
    rng.shuffle(negative_rows)

    selected = []
    seen_kepids = set()

    def collect(rows: list[dict], desired_label: int) -> None:
        class_count = 0

        for row in rows:
            if limit_per_class is not None and class_count >= limit_per_class:
                break

            kepid = int(row["kepid"])

            # Keep the scale estimate aligned with the current manifest policy.
            if kepid in seen_kepids:
                continue

            selected.append(row)
            seen_kepids.add(kepid)
            class_count += 1

    collect(positive_rows, desired_label=1)
    collect(negative_rows, desired_label=0)

    return selected


def summarize_records(
    records: list[dict],
    max_files_per_target: int,
    limit_per_class: int | None,
) -> pd.DataFrame:
    frame = pd.DataFrame(records)

    if frame.empty:
        return pd.DataFrame(
            [
                {
                    "limit_per_class": "all"
                    if limit_per_class is None
                    else limit_per_class,
                    "max_files_per_target": max_files_per_target,
                    "n_targets": 0,
                    "n_positive_targets": 0,
                    "n_negative_targets": 0,
                    "n_targets_with_fits": 0,
                    "n_targets_without_fits": 0,
                    "n_selected_fits": 0,
                    "n_size_known_fits": 0,
                    "n_size_unknown_fits": 0,
                    "known_size_gib": 0.0,
                    "estimated_total_size_gib": float("nan"),
                    "mean_known_file_size_mib": float("nan"),
                }
            ]
        )

    selected = frame[frame["has_long_cadence_fits"]].copy()

    n_selected_fits = int(selected["n_selected_fits"].sum())
    n_size_known_fits = int(selected["n_size_known_fits"].sum())
    n_size_unknown_fits = int(n_selected_fits - n_size_known_fits)

    known_size_bytes = int(selected["known_selected_size_bytes"].sum())

    if n_size_known_fits > 0:
        mean_known_size_bytes = known_size_bytes / n_size_known_fits
        estimated_total_size_bytes = known_size_bytes + (
            n_size_unknown_fits * mean_known_size_bytes
        )
        mean_known_file_size_mib = mean_known_size_bytes / (1024.0**2)
        estimated_total_size_gib = bytes_to_gib(estimated_total_size_bytes)
    else:
        mean_known_file_size_mib = float("nan")
        estimated_total_size_gib = float("nan")

    return pd.DataFrame(
        [
            {
                "limit_per_class": "all"
                if limit_per_class is None
                else limit_per_class,
                "max_files_per_target": max_files_per_target,
                "n_targets": int(frame.shape[0]),
                "n_positive_targets": int((frame["binary_label"] == 1).sum()),
                "n_negative_targets": int((frame["binary_label"] == 0).sum()),
                "n_targets_with_fits": int(frame["has_long_cadence_fits"].sum()),
                "n_targets_without_fits": int((~frame["has_long_cadence_fits"]).sum()),
                "n_selected_fits": n_selected_fits,
                "n_size_known_fits": n_size_known_fits,
                "n_size_unknown_fits": n_size_unknown_fits,
                "known_size_gib": bytes_to_gib(known_size_bytes),
                "estimated_total_size_gib": estimated_total_size_gib,
                "mean_known_file_size_mib": mean_known_file_size_mib,
            }
        ]
    )


def build_scale_records(
    candidate_rows: list[dict],
    max_files_per_target: int,
    include_file_sizes: bool,
) -> list[dict]:
    records = []

    for row in tqdm(candidate_rows, desc="estimating lightcurve scale"):
        kepid = int(row["kepid"])
        directory_url = kepler_lightcurve_dir(kepid)

        record = {
            "kepid": kepid,
            "kepoi_name": row["kepoi_name"],
            "koi_disposition": row["koi_disposition"],
            "binary_label": int(row["binary_label"]),
            "directory_url": directory_url,
            "directory_scan_ok": False,
            "directory_scan_error": "",
            "has_long_cadence_fits": False,
            "n_available_fits": 0,
            "n_selected_fits": 0,
            "n_size_known_fits": 0,
            "n_size_unknown_fits": 0,
            "known_selected_size_bytes": 0,
            "known_selected_size_gib": 0.0,
            "selected_fits_urls": "",
            "selected_fits_size_statuses": "",
        }

        try:
            file_urls = list_long_cadence_fits(directory_url)
            record["directory_scan_ok"] = True
        except Exception as exc:
            record["directory_scan_error"] = f"{type(exc).__name__}: {exc}"
            records.append(record)
            continue

        selected_urls = file_urls[:max_files_per_target]

        record["has_long_cadence_fits"] = len(file_urls) > 0
        record["n_available_fits"] = len(file_urls)
        record["n_selected_fits"] = len(selected_urls)
        record["selected_fits_urls"] = "|".join(selected_urls)

        if include_file_sizes:
            size_statuses = []
            known_size_bytes = 0
            known_count = 0

            for file_url in selected_urls:
                size_bytes, status = head_content_length(file_url)
                size_statuses.append(status)

                if size_bytes is not None:
                    known_size_bytes += size_bytes
                    known_count += 1

            record["n_size_known_fits"] = known_count
            record["n_size_unknown_fits"] = len(selected_urls) - known_count
            record["known_selected_size_bytes"] = known_size_bytes
            record["known_selected_size_gib"] = bytes_to_gib(known_size_bytes)
            record["selected_fits_size_statuses"] = "|".join(size_statuses)

        records.append(record)

    return records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Estimate Kepler long-cadence FITS scale for KOI targets. "
            "This does not download FITS files."
        )
    )

    parser.add_argument(
        "--limit-per-class",
        type=parse_limit_per_class,
        default=500,
        help="Targets per binary class to estimate, or 'all'. Default: 500.",
    )

    parser.add_argument(
        "--max-files-per-target",
        type=int,
        default=4,
        help="Maximum long-cadence FITS URLs selected per target. Default: 4.",
    )

    parser.add_argument(
        "--skip-file-sizes",
        action="store_true",
        help="Skip HEAD requests for Content-Length. Faster, but no size estimate.",
    )

    parser.add_argument(
        "--out-path",
        type=Path,
        default=DEFAULT_OUT_PATH,
        help=f"Per-target scale report path. Default: {DEFAULT_OUT_PATH}",
    )

    parser.add_argument(
        "--summary-path",
        type=Path,
        default=DEFAULT_SUMMARY_PATH,
        help=f"Summary report path. Default: {DEFAULT_SUMMARY_PATH}",
    )

    args = parser.parse_args()

    if args.max_files_per_target <= 0:
        raise ValueError(
            f"--max-files-per-target must be positive; got {args.max_files_per_target}"
        )

    return args


def main() -> None:
    args = parse_args()

    include_file_sizes = not args.skip_file_sizes

    print("koi_table:", KOI_PATH)
    print(
        "limit_per_class:",
        "all" if args.limit_per_class is None else args.limit_per_class,
    )
    print("max_files_per_target:", args.max_files_per_target)
    print("include_file_sizes:", include_file_sizes)
    print("downloads_fits_files: False")
    print()

    candidate_rows = load_candidate_rows(
        limit_per_class=args.limit_per_class,
    )

    print("candidate_targets_before_directory_scan:", len(candidate_rows))
    print()

    records = build_scale_records(
        candidate_rows=candidate_rows,
        max_files_per_target=args.max_files_per_target,
        include_file_sizes=include_file_sizes,
    )

    report = pd.DataFrame(records)
    summary = summarize_records(
        records=records,
        max_files_per_target=args.max_files_per_target,
        limit_per_class=args.limit_per_class,
    )

    args.out_path.parent.mkdir(parents=True, exist_ok=True)
    args.summary_path.parent.mkdir(parents=True, exist_ok=True)

    report.to_csv(args.out_path, index=False)
    summary.to_csv(args.summary_path, index=False)

    print("scale summary:")
    print(
        summary.to_string(
            index=False,
            float_format=lambda value: (
                "nan" if not math.isfinite(value) else f"{value:.3f}"
            ),
        )
    )
    print()
    print("directory scan failures:", int((~report["directory_scan_ok"]).sum()))

    if include_file_sizes:
        print("size-known FITS:", int(summary["n_size_known_fits"].iloc[0]))
        print("size-unknown FITS:", int(summary["n_size_unknown_fits"].iloc[0]))

    print()
    print("wrote:", args.out_path)
    print("wrote:", args.summary_path)
    print("ESTIMATE_LIGHTCURVE_SCALE_OK")


if __name__ == "__main__":
    main()
