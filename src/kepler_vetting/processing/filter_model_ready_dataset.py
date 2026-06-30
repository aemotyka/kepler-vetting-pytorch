from __future__ import annotations

import argparse
from collections import Counter

import numpy as np
import pandas as pd

from kepler_vetting.processing.common import (
    MODEL_READY_MANIFEST_PATH,
    MODEL_READY_NPZ_PATH,
    MODEL_READINESS_REPORT_PATH,
    PROCESSED_DIR,
    PROCESSED_NPZ_PATH,
)
from kepler_vetting.processing.model_readiness import build_model_readiness_report


DROPPED_ROWS_PATH = PROCESSED_DIR / "model_ready_dropped_rows.csv"

DROPPED_ROW_COLUMNS = [
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


def subset_array(
    name: str,
    array: np.ndarray,
    keep_mask: np.ndarray,
    n_rows: int,
) -> np.ndarray:
    if array.ndim >= 1 and array.shape[0] == n_rows:
        return array[keep_mask]

    return array


def reason_counts(frame: pd.DataFrame) -> Counter:
    counts: Counter = Counter()

    for raw_reasons in frame["model_ready_failure_reasons"]:
        for reason in str(raw_reasons).split("|"):
            reason = reason.strip()

            if reason:
                counts[reason] += 1

    return counts


def print_dropped_rows_summary(dropped: pd.DataFrame) -> None:
    print()
    print("dropped rows summary:")

    if dropped.empty:
        print("none")
        return

    print("dropped_rows:", len(dropped))

    print()
    print("dropped by label:")
    print(dropped["binary_label"].value_counts().sort_index())

    print()
    print("dropped by disposition:")
    print(dropped["koi_disposition"].value_counts())

    print()
    print("dropped by failure reason:")
    reason_frame = pd.DataFrame(
        [
            {
                "failure_reason": reason,
                "n": count,
            }
            for reason, count in reason_counts(dropped).most_common()
        ]
    )

    print(reason_frame.to_string(index=False))

    print()
    print("dropped quality summary:")
    print(
        dropped[
            [
                "n_fits_files",
                "n_clean_points",
                "global_missing_bin_fraction_before_interp",
                "local_missing_bin_fraction_before_interp",
            ]
        ].describe()
    )

    print()
    print("worst dropped rows by global missing fraction:")
    print(
        dropped.sort_values(
            [
                "global_missing_bin_fraction_before_interp",
                "local_missing_bin_fraction_before_interp",
                "n_clean_points",
            ],
            ascending=[
                False,
                False,
                True,
            ],
        )[DROPPED_ROW_COLUMNS]
        .head(10)
        .to_string(index=False)
    )

    print()
    print("worst dropped rows by local missing fraction:")
    print(
        dropped.sort_values(
            [
                "local_missing_bin_fraction_before_interp",
                "global_missing_bin_fraction_before_interp",
                "n_clean_points",
            ],
            ascending=[
                False,
                False,
                True,
            ],
        )[DROPPED_ROW_COLUMNS]
        .head(10)
        .to_string(index=False)
    )

    print()
    print("lowest clean-point dropped rows:")
    print(
        dropped.sort_values(
            [
                "n_clean_points",
                "n_fits_files",
                "global_missing_bin_fraction_before_interp",
                "local_missing_bin_fraction_before_interp",
            ],
            ascending=[
                True,
                True,
                False,
                False,
            ],
        )[DROPPED_ROW_COLUMNS]
        .head(10)
        .to_string(index=False)
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Filter the processed Kepler dataset to model-ready rows."
    )

    parser.add_argument(
        "--print-all-dropped-rows",
        action="store_true",
        help=(
            "Print every dropped row. By default, only a concise summary is printed "
            "and the full dropped-row detail is written to CSV."
        ),
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

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

    dropped = report[~report["model_ready"]].reset_index(drop=True)
    dropped.to_csv(DROPPED_ROWS_PATH, index=False)

    print("input_dataset:", PROCESSED_NPZ_PATH)
    print("output_dataset:", MODEL_READY_NPZ_PATH)
    print("readiness_report:", MODEL_READINESS_REPORT_PATH)
    print("model_ready_manifest:", MODEL_READY_MANIFEST_PATH)
    print("dropped_rows_detail:", DROPPED_ROWS_PATH)
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

    print_dropped_rows_summary(dropped)

    if args.print_all_dropped_rows and not dropped.empty:
        print()
        print("all dropped rows:")
        print(dropped[DROPPED_ROW_COLUMNS].to_string(index=False))

    print()
    print("FILTER_MODEL_READY_DATASET_OK")


if __name__ == "__main__":
    main()