from __future__ import annotations

import argparse
import random
import re
import time
from pathlib import Path
from urllib.parse import urljoin
from urllib.request import Request, urlopen

import pandas as pd


KOI_PATH = Path("data/raw/koi_q1_q17_dr25.csv")
DEFAULT_OUT_PATH = Path("data/metadata/lightcurve_manifest.csv")

DEFAULT_TARGETS_PER_CLASS = "500"
DEFAULT_MAX_FILES_PER_TARGET = 4
RANDOM_SEED = 42
MAX_DIRECTORY_ATTEMPTS = 4
RETRY_SLEEP_SECONDS = 5

LABEL_MAP = {
    "CONFIRMED": 1,
    "CANDIDATE": 1,
    "FALSE POSITIVE": 0,
}

MANIFEST_COLUMNS = [
    "kepid",
    "kepoi_name",
    "koi_disposition",
    "binary_label",
    "koi_period",
    "koi_time0bk",
    "koi_duration",
    "koi_depth",
    "koi_model_snr",
    "koi_prad",
    "koi_teq",
    "koi_insol",
    "koi_steff",
    "koi_slogg",
    "koi_srad",
    "koi_kepmag",
    "directory_url",
    "n_available_fits",
    "selected_fits_urls",
]

REQUIRED_KOI_COLUMNS = [
    "kepid",
    "kepoi_name",
    "koi_disposition",
    "koi_period",
    "koi_time0bk",
]


def parse_targets_per_class(value: str) -> int | None:
    normalized = str(value).strip().lower()

    if normalized == "all":
        return None

    parsed = int(normalized)

    if parsed <= 0:
        raise argparse.ArgumentTypeError(
            f"--targets-per-class must be positive or 'all'; got {value}"
        )

    return parsed


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
                print(
                    f"retry {attempt}/{MAX_DIRECTORY_ATTEMPTS} failed for {url}: "
                    f"{type(exc).__name__}: {exc}"
                )
                time.sleep(RETRY_SLEEP_SECONDS)

    raise RuntimeError(
        f"failed after {MAX_DIRECTORY_ATTEMPTS} attempts for {url}: {last_error}"
    )


def list_long_cadence_fits(directory_url: str) -> list[str]:
    html = read_url_text_with_retries(directory_url)
    hrefs = re.findall(r'href=["\']([^"\']+)["\']', html)

    return sorted(
        urljoin(directory_url, href)
        for href in hrefs
        if href.endswith("_llc.fits")
    )


def load_koi_table() -> pd.DataFrame:
    if not KOI_PATH.exists():
        raise FileNotFoundError(f"Missing KOI CSV: {KOI_PATH}")

    df = pd.read_csv(KOI_PATH)

    missing = [
        column
        for column in REQUIRED_KOI_COLUMNS
        if column not in df.columns
    ]

    if missing:
        raise ValueError(f"KOI table is missing required columns: {missing}")

    df = df.dropna(subset=REQUIRED_KOI_COLUMNS)
    df = df[df["koi_disposition"].isin(LABEL_MAP)].copy()

    df["kepid"] = df["kepid"].astype(int)
    df["kepoi_name"] = df["kepoi_name"].astype(str)
    df["binary_label"] = df["koi_disposition"].map(LABEL_MAP).astype(int)

    return df


def select_candidate_rows(
    df: pd.DataFrame,
    targets_per_class: int | None,
) -> pd.DataFrame:
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
            if targets_per_class is not None and class_count >= targets_per_class:
                break

            kepid = int(row["kepid"])

            if kepid in seen_kepids:
                continue

            selected.append(row)
            seen_kepids.add(kepid)
            class_count += 1

    collect(positive_rows, desired_label=1)
    collect(negative_rows, desired_label=0)

    return pd.DataFrame(selected)


def build_manifest_by_scanning(
    df: pd.DataFrame,
    targets_per_class: int | None,
    max_files_per_target: int,
) -> pd.DataFrame:
    candidates = select_candidate_rows(
        df=df,
        targets_per_class=targets_per_class,
    )

    selected = []

    for _, row in candidates.iterrows():
        kepid = int(row["kepid"])
        directory_url = kepler_lightcurve_dir(kepid)

        try:
            file_urls = list_long_cadence_fits(directory_url)
        except Exception as exc:
            print(f"skip kepid={kepid}: {type(exc).__name__}: {exc}")
            continue

        if not file_urls:
            print(f"skip kepid={kepid}: no long-cadence FITS files")
            continue

        selected_urls = file_urls[:max_files_per_target]

        record = manifest_record_from_row(
            row=row,
            directory_url=directory_url,
            n_available_fits=len(file_urls),
            selected_fits_urls=selected_urls,
        )

        selected.append(record)

        print(
            f"selected label={record['binary_label']} "
            f"kepid={record['kepid']} "
            f"koi={record['kepoi_name']} "
            f"disposition={record['koi_disposition']} "
            f"files={len(file_urls)}"
        )

    return pd.DataFrame(selected)


def load_scale_estimate(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"missing scale estimate CSV: {path}")

    frame = pd.read_csv(path)

    required_columns = [
        "kepid",
        "kepoi_name",
        "binary_label",
        "directory_url",
        "directory_scan_ok",
        "has_long_cadence_fits",
        "n_available_fits",
        "selected_fits_urls",
    ]

    missing = [
        column
        for column in required_columns
        if column not in frame.columns
    ]

    if missing:
        raise ValueError(f"scale estimate is missing required columns: {missing}")

    frame["kepid"] = frame["kepid"].astype(int)
    frame["kepoi_name"] = frame["kepoi_name"].astype(str)
    frame["binary_label"] = frame["binary_label"].astype(int)

    frame = frame[
        frame["directory_scan_ok"].astype(bool)
        & frame["has_long_cadence_fits"].astype(bool)
        & frame["selected_fits_urls"].notna()
        & (frame["selected_fits_urls"].astype(str).str.len() > 0)
    ].copy()

    return frame


def build_manifest_from_scale_estimate(
    df: pd.DataFrame,
    scale_estimate_path: Path,
    targets_per_class: int | None,
    balance_to_min_class: bool,
    max_files_per_target: int,
) -> pd.DataFrame:
    scale = load_scale_estimate(scale_estimate_path)

    # The scale estimate already carries koi_disposition, binary_label,
    # kepid, and kepoi_name. Only merge in the remaining KOI metadata fields
    # so pandas does not suffix duplicate disposition columns.
    metadata_columns = [
        "kepid",
        "kepoi_name",
        "binary_label",
        "koi_period",
        "koi_time0bk",
        "koi_duration",
        "koi_depth",
        "koi_model_snr",
        "koi_prad",
        "koi_teq",
        "koi_insol",
        "koi_steff",
        "koi_slogg",
        "koi_srad",
        "koi_kepmag",
    ]

    metadata = df[metadata_columns].copy()

    merged = scale.merge(
        metadata,
        on=[
            "kepid",
            "kepoi_name",
            "binary_label",
        ],
        how="inner",
        validate="one_to_one",
    )

    if merged.empty:
        raise RuntimeError(
            f"scale estimate {scale_estimate_path} did not match KOI metadata"
        )

    if "koi_disposition" not in merged.columns:
        raise ValueError(
            "merged scale estimate is missing koi_disposition; "
            f"available columns={sorted(merged.columns)}"
        )

    if balance_to_min_class:
        per_class_limit = int(merged["binary_label"].value_counts().min())
    else:
        per_class_limit = targets_per_class

    selected_parts = []

    for label in [
        1,
        0,
    ]:
        class_rows = merged[merged["binary_label"] == label].copy()

        if per_class_limit is not None:
            class_rows = class_rows.head(per_class_limit)

        selected_parts.append(class_rows)

    selected = pd.concat(selected_parts, ignore_index=True)

    records = []

    for _, row in selected.iterrows():
        selected_urls = str(row["selected_fits_urls"]).split("|")
        selected_urls = selected_urls[:max_files_per_target]

        records.append(
            manifest_record_from_row(
                row=row,
                directory_url=row["directory_url"],
                n_available_fits=int(row["n_available_fits"]),
                selected_fits_urls=selected_urls,
            )
        )

    return pd.DataFrame(records)

def manifest_record_from_row(
    row: pd.Series,
    directory_url: str,
    n_available_fits: int,
    selected_fits_urls: list[str],
) -> dict:
    return {
        "kepid": int(row["kepid"]),
        "kepoi_name": row["kepoi_name"],
        "koi_disposition": row["koi_disposition"],
        "binary_label": int(row["binary_label"]),
        "koi_period": row["koi_period"],
        "koi_time0bk": row["koi_time0bk"],
        "koi_duration": row.get("koi_duration"),
        "koi_depth": row.get("koi_depth"),
        "koi_model_snr": row.get("koi_model_snr"),
        "koi_prad": row.get("koi_prad"),
        "koi_teq": row.get("koi_teq"),
        "koi_insol": row.get("koi_insol"),
        "koi_steff": row.get("koi_steff"),
        "koi_slogg": row.get("koi_slogg"),
        "koi_srad": row.get("koi_srad"),
        "koi_kepmag": row.get("koi_kepmag"),
        "directory_url": directory_url,
        "n_available_fits": n_available_fits,
        "selected_fits_urls": "|".join(selected_fits_urls),
    }


def validate_manifest(manifest: pd.DataFrame) -> None:
    missing = [
        column
        for column in MANIFEST_COLUMNS
        if column not in manifest.columns
    ]

    if missing:
        raise ValueError(f"manifest is missing required columns: {missing}")

    if manifest.empty:
        raise ValueError("manifest is empty")

    if manifest["kepid"].duplicated().any():
        duplicates = manifest[manifest["kepid"].duplicated()]["kepid"].head(10).tolist()
        raise ValueError(f"manifest contains duplicate kepid values: {duplicates}")

    if set(manifest["binary_label"].tolist()) - {0, 1}:
        raise ValueError("manifest binary_label must only contain 0/1 values")

    missing_urls = (
        manifest["selected_fits_urls"].isna()
        | (manifest["selected_fits_urls"].astype(str).str.len() == 0)
    )

    if missing_urls.any():
        raise ValueError(f"manifest contains {int(missing_urls.sum())} rows without FITS URLs")


def write_manifest(manifest: pd.DataFrame, out_path: Path) -> None:
    manifest = manifest[MANIFEST_COLUMNS].copy()
    validate_manifest(manifest)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(out_path, index=False)

    print()
    print("wrote:", out_path)
    print("shape:", manifest.shape)
    print("binary_label_counts:")
    print(manifest["binary_label"].value_counts().sort_index())
    print("disposition_counts:")
    print(manifest["koi_disposition"].value_counts())
    print("selected_fits_count:", int(manifest["selected_fits_urls"].str.split("|").map(len).sum()))
    print("unique_kepids:", int(manifest["kepid"].nunique()))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a Kepler long-cadence FITS manifest from KOI metadata."
    )

    parser.add_argument(
        "--targets-per-class",
        type=parse_targets_per_class,
        default=parse_targets_per_class(DEFAULT_TARGETS_PER_CLASS),
        help="Targets per binary class, or 'all'. Default: 500.",
    )

    parser.add_argument(
        "--balance-to-min-class",
        action="store_true",
        help=(
            "When using a cached scale estimate, limit both classes to the smaller "
            "available class count."
        ),
    )

    parser.add_argument(
        "--max-files-per-target",
        type=int,
        default=DEFAULT_MAX_FILES_PER_TARGET,
        help="Maximum long-cadence FITS URLs to select per target. Default: 4.",
    )

    parser.add_argument(
        "--from-scale-estimate",
        type=Path,
        default=None,
        help=(
            "Optional path to outputs/data/lightcurve_scale_estimate.csv. "
            "When provided, reuse scanned FITS URLs instead of scanning MAST again."
        ),
    )

    parser.add_argument(
        "--out-path",
        type=Path,
        default=DEFAULT_OUT_PATH,
        help=f"Output manifest CSV path. Default: {DEFAULT_OUT_PATH}",
    )

    args = parser.parse_args()

    if args.max_files_per_target <= 0:
        raise ValueError(
            f"--max-files-per-target must be positive; got {args.max_files_per_target}"
        )

    if args.balance_to_min_class and args.from_scale_estimate is None:
        raise ValueError("--balance-to-min-class requires --from-scale-estimate")

    return args


def main() -> None:
    args = parse_args()

    print("koi_table:", KOI_PATH)
    print(
        "targets_per_class:",
        "all" if args.targets_per_class is None else args.targets_per_class,
    )
    print("balance_to_min_class:", args.balance_to_min_class)
    print("max_files_per_target:", args.max_files_per_target)
    print("from_scale_estimate:", args.from_scale_estimate)
    print("out_path:", args.out_path)
    print()

    df = load_koi_table()

    if args.from_scale_estimate is not None:
        manifest = build_manifest_from_scale_estimate(
            df=df,
            scale_estimate_path=args.from_scale_estimate,
            targets_per_class=args.targets_per_class,
            balance_to_min_class=args.balance_to_min_class,
            max_files_per_target=args.max_files_per_target,
        )
    else:
        manifest = build_manifest_by_scanning(
            df=df,
            targets_per_class=args.targets_per_class,
            max_files_per_target=args.max_files_per_target,
        )

    write_manifest(
        manifest=manifest,
        out_path=args.out_path,
    )


if __name__ == "__main__":
    main()