from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from kepler_vetting.modeling.thresholds import (
    FIXED_THRESHOLD_VARIANT,
    VAL_TUNED_F1_THRESHOLD_VARIANT,
)


METRICS_DIR = Path("outputs/metrics")

ERROR_SUMMARY_PATH = METRICS_DIR / "tabular_vs_fused_error_summary.csv"
CHANGED_PREDICTIONS_PATH = METRICS_DIR / "tabular_vs_fused_changed_predictions.csv"
DISPOSITION_SUMMARY_PATH = METRICS_DIR / "tabular_vs_fused_disposition_summary.csv"
FEATURE_SUMMARY_PATH = METRICS_DIR / "tabular_vs_fused_feature_summary.csv"

FEATURE_DIFFERENCES_PATH = METRICS_DIR / "tabular_vs_fused_feature_differences.csv"
RECURRING_CHANGED_ROWS_PATH = METRICS_DIR / "tabular_vs_fused_recurring_changed_rows.csv"
STRONGEST_DISAGREEMENTS_PATH = METRICS_DIR / "tabular_vs_fused_strongest_disagreements.csv"


METRIC_VARIANT_ORDER = {
    FIXED_THRESHOLD_VARIANT: 0,
    VAL_TUNED_F1_THRESHOLD_VARIANT: 1,
}

SPLIT_ORDER = {
    "test": 0,
    "val": 1,
    "train": 2,
}

OUTCOME_ORDER = {
    "fused_only_correct": 0,
    "tabular_only_correct": 1,
    "both_wrong": 2,
    "both_correct": 3,
}


ERROR_SUMMARY_COLUMNS = [
    "metric_variant",
    "split",
    "n",
    "tabular_accuracy",
    "fused_accuracy",
    "accuracy_delta",
    "both_correct",
    "both_wrong",
    "fused_only_correct",
    "tabular_only_correct",
    "net_fused_correct_gain",
    "changed_prediction_count",
    "changed_prediction_rate",
]


DISPOSITION_COLUMNS = [
    "metric_variant",
    "split",
    "display_disposition",
    "outcome",
    "n",
]


RECURRING_ROW_COLUMNS = [
    "metric_variant",
    "row_index",
    "kepid",
    "kepoi_name",
    "display_disposition",
    "y_true",
    "outcome",
    "seed_count",
    "mean_tabular_score",
    "mean_fused_score",
    "mean_score_delta_fused_minus_tabular",
    "mean_abs_score_delta",
]


FEATURE_DIFFERENCE_COLUMNS = [
    "metric_variant",
    "split",
    "feature",
    "fused_only_correct_mean",
    "tabular_only_correct_mean",
    "both_wrong_mean",
    "mean_diff_fused_only_minus_tabular_only",
    "abs_mean_diff_fused_only_vs_tabular_only",
]


STRONGEST_DISAGREEMENT_COLUMNS = [
    "metric_variant",
    "seed",
    "split",
    "row_index",
    "kepid",
    "kepoi_name",
    "display_disposition",
    "y_true",
    "outcome",
    "tabular_score",
    "fused_score",
    "score_delta_fused_minus_tabular",
    "abs_score_delta",
    "tabular_pred",
    "fused_pred",
]


def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"missing required file: {path}")


def sort_columns(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()

    if "metric_variant" in frame.columns:
        frame["metric_variant_order"] = frame["metric_variant"].map(METRIC_VARIANT_ORDER)

    if "split" in frame.columns:
        frame["split_order"] = frame["split"].map(SPLIT_ORDER)

    if "outcome" in frame.columns:
        frame["outcome_order"] = frame["outcome"].map(OUTCOME_ORDER)

    sort_keys = [
        column
        for column in [
            "split_order",
            "metric_variant_order",
            "outcome_order",
        ]
        if column in frame.columns
    ]

    if sort_keys:
        frame = frame.sort_values(sort_keys).reset_index(drop=True)

    frame = frame.drop(
        columns=[
            column
            for column in [
                "metric_variant_order",
                "split_order",
                "outcome_order",
            ]
            if column in frame.columns
        ]
    )

    return frame


def load_error_summary() -> pd.DataFrame:
    require_file(ERROR_SUMMARY_PATH)

    frame = pd.read_csv(ERROR_SUMMARY_PATH)

    missing = set(ERROR_SUMMARY_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(
            f"{ERROR_SUMMARY_PATH} is missing required columns: {sorted(missing)}"
        )

    return sort_columns(frame)


def load_changed_predictions() -> pd.DataFrame:
    require_file(CHANGED_PREDICTIONS_PATH)

    frame = pd.read_csv(CHANGED_PREDICTIONS_PATH)

    required_columns = {
        "metric_variant",
        "seed",
        "split",
        "row_index",
        "kepid",
        "kepoi_name",
        "y_true",
        "outcome",
        "tabular_score",
        "fused_score",
        "score_delta_fused_minus_tabular",
        "tabular_pred",
        "fused_pred",
    }

    missing = required_columns - set(frame.columns)
    if missing:
        raise ValueError(
            f"{CHANGED_PREDICTIONS_PATH} is missing required columns: {sorted(missing)}"
        )

    if "koi_disposition" in frame.columns:
        frame["display_disposition"] = frame["koi_disposition"]
    elif "disposition" in frame.columns:
        frame["display_disposition"] = frame["disposition"]
    else:
        frame["display_disposition"] = "unknown"

    frame["abs_score_delta"] = frame["score_delta_fused_minus_tabular"].abs()

    return sort_columns(frame)


def load_disposition_summary() -> pd.DataFrame:
    require_file(DISPOSITION_SUMMARY_PATH)

    frame = pd.read_csv(DISPOSITION_SUMMARY_PATH)

    required_columns = {
        "metric_variant",
        "split",
        "disposition",
        "outcome",
        "n",
    }

    missing = required_columns - set(frame.columns)
    if missing:
        raise ValueError(
            f"{DISPOSITION_SUMMARY_PATH} is missing required columns: {sorted(missing)}"
        )

    frame = frame.rename(columns={"disposition": "display_disposition"})

    return sort_columns(frame)


def load_feature_summary() -> pd.DataFrame:
    require_file(FEATURE_SUMMARY_PATH)

    frame = pd.read_csv(FEATURE_SUMMARY_PATH)

    required_columns = {
        "metric_variant",
        "split",
        "outcome",
        "feature",
        "n",
        "mean",
    }

    missing = required_columns - set(frame.columns)
    if missing:
        raise ValueError(
            f"{FEATURE_SUMMARY_PATH} is missing required columns: {sorted(missing)}"
        )

    return sort_columns(frame)


def build_feature_differences(feature_summary: pd.DataFrame) -> pd.DataFrame:
    pivot = (
        feature_summary
        .pivot_table(
            index=[
                "metric_variant",
                "split",
                "feature",
            ],
            columns="outcome",
            values="mean",
            aggfunc="first",
        )
        .reset_index()
    )

    for outcome in [
        "fused_only_correct",
        "tabular_only_correct",
        "both_wrong",
    ]:
        if outcome not in pivot.columns:
            pivot[outcome] = np.nan

    pivot = pivot.rename(
        columns={
            "fused_only_correct": "fused_only_correct_mean",
            "tabular_only_correct": "tabular_only_correct_mean",
            "both_wrong": "both_wrong_mean",
        }
    )

    pivot["mean_diff_fused_only_minus_tabular_only"] = (
        pivot["fused_only_correct_mean"]
        - pivot["tabular_only_correct_mean"]
    )

    pivot["abs_mean_diff_fused_only_vs_tabular_only"] = (
        pivot["mean_diff_fused_only_minus_tabular_only"].abs()
    )

    pivot["metric_variant_order"] = pivot["metric_variant"].map(METRIC_VARIANT_ORDER)
    pivot["split_order"] = pivot["split"].map(SPLIT_ORDER)

    pivot = pivot.sort_values(
        [
            "split_order",
            "metric_variant_order",
            "abs_mean_diff_fused_only_vs_tabular_only",
        ],
        ascending=[
            True,
            True,
            False,
        ],
    ).reset_index(drop=True)

    pivot = pivot.drop(
        columns=[
            "metric_variant_order",
            "split_order",
        ]
    )

    return pivot


def build_recurring_changed_rows(changed_predictions: pd.DataFrame) -> pd.DataFrame:
    group_columns = [
        "metric_variant",
        "row_index",
        "kepid",
        "kepoi_name",
        "display_disposition",
        "y_true",
        "outcome",
    ]

    recurring = (
        changed_predictions
        .groupby(group_columns, as_index=False)
        .agg(
            seed_count=("seed", "nunique"),
            mean_tabular_score=("tabular_score", "mean"),
            mean_fused_score=("fused_score", "mean"),
            mean_score_delta_fused_minus_tabular=(
                "score_delta_fused_minus_tabular",
                "mean",
            ),
            mean_abs_score_delta=("abs_score_delta", "mean"),
            mean_tabular_pred=("tabular_pred", "mean"),
            mean_fused_pred=("fused_pred", "mean"),
        )
    )

    recurring["metric_variant_order"] = recurring["metric_variant"].map(
        METRIC_VARIANT_ORDER
    )
    recurring["outcome_order"] = recurring["outcome"].map(OUTCOME_ORDER)

    recurring = recurring.sort_values(
        [
            "metric_variant_order",
            "outcome_order",
            "seed_count",
            "mean_abs_score_delta",
        ],
        ascending=[
            True,
            True,
            False,
            False,
        ],
    ).reset_index(drop=True)

    recurring = recurring.drop(
        columns=[
            "metric_variant_order",
            "outcome_order",
        ]
    )

    return recurring


def build_strongest_disagreements(changed_predictions: pd.DataFrame) -> pd.DataFrame:
    strongest = changed_predictions.sort_values(
        [
            "split",
            "metric_variant",
            "abs_score_delta",
        ],
        ascending=[
            True,
            True,
            False,
        ],
    ).reset_index(drop=True)

    return strongest


def print_table(
    title: str,
    frame: pd.DataFrame,
    columns: list[str],
    top_n: int | None = None,
) -> None:
    print(title)

    if frame.empty:
        print("(no rows)")
        print()
        return

    display = frame.copy()

    if top_n is not None:
        display = display.head(top_n)

    display_columns = [
        column
        for column in columns
        if column in display.columns
    ]

    print(
        display[display_columns].to_string(
            index=False,
            float_format=lambda value: f"{value:.3f}",
        )
    )
    print()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize tabular vs fused model error analysis outputs."
    )

    parser.add_argument(
        "--top-n",
        type=int,
        default=15,
        help="Number of rows to print for detail sections.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    error_summary = load_error_summary()
    changed_predictions = load_changed_predictions()
    disposition_summary = load_disposition_summary()
    feature_summary = load_feature_summary()

    feature_differences = build_feature_differences(feature_summary)
    recurring_changed_rows = build_recurring_changed_rows(changed_predictions)
    strongest_disagreements = build_strongest_disagreements(changed_predictions)

    feature_differences.to_csv(FEATURE_DIFFERENCES_PATH, index=False)
    recurring_changed_rows.to_csv(RECURRING_CHANGED_ROWS_PATH, index=False)
    strongest_disagreements.to_csv(STRONGEST_DISAGREEMENTS_PATH, index=False)

    test_error_summary = error_summary[error_summary["split"] == "test"]

    print_table(
        title="test-set tabular vs fused summary:",
        frame=test_error_summary,
        columns=ERROR_SUMMARY_COLUMNS,
    )

    test_disposition_changed = (
        changed_predictions[changed_predictions["split"] == "test"]
        .groupby(
            [
                "metric_variant",
                "display_disposition",
                "outcome",
            ],
            as_index=False,
        )
        .size()
        .rename(columns={"size": "n"})
    )
    test_disposition_changed = sort_columns(test_disposition_changed)

    print_table(
        title="changed test predictions by disposition:",
        frame=test_disposition_changed,
        columns=DISPOSITION_COLUMNS,
        top_n=None,
    )

    test_feature_differences = feature_differences[
        feature_differences["split"] == "test"
    ].copy()

    print_table(
        title=f"top {args.top_n} test feature differences: fused-only correct vs tabular-only correct",
        frame=test_feature_differences,
        columns=FEATURE_DIFFERENCE_COLUMNS,
        top_n=args.top_n,
    )

    test_recurring_rows = recurring_changed_rows[
        recurring_changed_rows["metric_variant"].isin(METRIC_VARIANT_ORDER)
    ].copy()

    print_table(
        title=f"top {args.top_n} recurring changed rows:",
        frame=test_recurring_rows,
        columns=RECURRING_ROW_COLUMNS,
        top_n=args.top_n,
    )

    test_strongest = strongest_disagreements[
        strongest_disagreements["split"] == "test"
    ].copy()

    print_table(
        title=f"top {args.top_n} strongest individual test disagreements:",
        frame=test_strongest,
        columns=STRONGEST_DISAGREEMENT_COLUMNS,
        top_n=args.top_n,
    )

    print("wrote:", FEATURE_DIFFERENCES_PATH)
    print("wrote:", RECURRING_CHANGED_ROWS_PATH)
    print("wrote:", STRONGEST_DISAGREEMENTS_PATH)
    print("SUMMARIZE_ERROR_ANALYSIS_OK")


if __name__ == "__main__":
    main()