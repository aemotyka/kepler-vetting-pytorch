from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from kepler_vetting.modeling.local_derived_features import (
    LOCAL_DERIVED_FEATURE_NAMES,
    build_local_derived_feature_matrix,
)
from kepler_vetting.modeling.thresholds import (
    FIXED_THRESHOLD_VARIANT,
    VAL_TUNED_F1_THRESHOLD_VARIANT,
)
from kepler_vetting.processing.common import (
    MODEL_READY_MANIFEST_PATH,
    MODEL_READY_NPZ_PATH,
)


METRICS_DIR = Path("outputs/metrics")

MODEL_COMPARISON_BY_SEED_PATH = METRICS_DIR / "model_comparison_by_seed.csv"

TABULAR_PREDICTIONS_PATH = METRICS_DIR / "tabular_baseline_predictions.csv"
TABULAR_LOCAL_FEATURES_PREDICTIONS_PATH = (
    METRICS_DIR / "tabular_local_features_predictions.csv"
)
LOCAL_CNN_PREDICTIONS_PATH = METRICS_DIR / "lightcurve_cnn_predictions.csv"
GLOBAL_CNN_PREDICTIONS_PATH = METRICS_DIR / "global_lightcurve_cnn_predictions.csv"
FUSED_PREDICTIONS_PATH = METRICS_DIR / "fused_local_model_predictions.csv"
FUSED_LOCAL_FEATURES_PREDICTIONS_PATH = (
    METRICS_DIR / "fused_local_features_model_predictions.csv"
)
STACKED_SCORE_PREDICTIONS_PATH = METRICS_DIR / "stacked_score_model_predictions.csv"

PAIRWISE_SUMMARY_PATH = METRICS_DIR / "pairwise_model_error_summary.csv"
PAIRWISE_CHANGED_PATH = METRICS_DIR / "pairwise_model_changed_predictions.csv"
PAIRWISE_DISPOSITION_PATH = METRICS_DIR / "pairwise_model_disposition_summary.csv"
PAIRWISE_FEATURE_SUMMARY_PATH = METRICS_DIR / "pairwise_model_feature_summary.csv"
PAIRWISE_FEATURE_DIFFERENCES_PATH = METRICS_DIR / "pairwise_model_feature_differences.csv"
PAIRWISE_RECURRING_PATH = METRICS_DIR / "pairwise_model_recurring_changed_rows.csv"
PAIRWISE_STRONGEST_PATH = METRICS_DIR / "pairwise_model_strongest_disagreements.csv"


METRIC_VARIANTS = [
    FIXED_THRESHOLD_VARIANT,
    VAL_TUNED_F1_THRESHOLD_VARIANT,
]

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
    "right_only_correct": 0,
    "left_only_correct": 1,
    "both_wrong": 2,
    "both_correct": 3,
}

REQUIRED_PREDICTION_COLUMNS = {
    "model",
    "seed",
    "split",
    "row_index",
    "kepid",
    "kepoi_name",
    "disposition",
    "y_true",
    "planet_like_score",
}

SUMMARY_COLUMNS = [
    "pair_id",
    "left_display_model",
    "right_display_model",
    "metric_variant",
    "split",
    "n",
    "left_accuracy",
    "right_accuracy",
    "accuracy_delta_right_minus_left",
    "both_correct",
    "both_wrong",
    "right_only_correct",
    "left_only_correct",
    "net_right_correct_gain",
    "changed_prediction_count",
    "changed_prediction_rate",
]

CHANGED_COLUMNS = [
    "pair_id",
    "metric_variant",
    "seed",
    "split",
    "row_index",
    "kepid",
    "kepoi_name",
    "display_disposition",
    "y_true",
    "outcome",
    "left_display_model",
    "right_display_model",
    "left_score",
    "right_score",
    "score_delta_right_minus_left",
    "abs_score_delta",
    "left_pred",
    "right_pred",
]

FEATURE_DIFFERENCE_COLUMNS = [
    "pair_id",
    "metric_variant",
    "split",
    "feature",
    "right_only_correct_mean",
    "left_only_correct_mean",
    "both_wrong_mean",
    "mean_diff_right_only_minus_left_only",
    "abs_mean_diff_right_only_vs_left_only",
]

RECURRING_COLUMNS = [
    "pair_id",
    "metric_variant",
    "row_index",
    "kepid",
    "kepoi_name",
    "display_disposition",
    "y_true",
    "outcome",
    "seed_count",
    "mean_left_score",
    "mean_right_score",
    "mean_score_delta_right_minus_left",
    "mean_abs_score_delta",
]

STRONGEST_COLUMNS = [
    "pair_id",
    "metric_variant",
    "seed",
    "split",
    "row_index",
    "kepid",
    "kepoi_name",
    "display_disposition",
    "y_true",
    "outcome",
    "left_score",
    "right_score",
    "score_delta_right_minus_left",
    "abs_score_delta",
    "left_pred",
    "right_pred",
]


@dataclass(frozen=True)
class ModelSpec:
    display_model: str
    model_name: str
    predictions_path: Path


@dataclass(frozen=True)
class PairSpec:
    pair_id: str
    left: ModelSpec
    right: ModelSpec


MODEL_SPECS = {
    "tabular_logistic_regression": ModelSpec(
        display_model="tabular_logistic_regression",
        model_name="logistic_regression",
        predictions_path=TABULAR_PREDICTIONS_PATH,
    ),
    "tabular_local_features_logistic_regression": ModelSpec(
        display_model="tabular_local_features_logistic_regression",
        model_name="tabular_local_features_logistic_regression",
        predictions_path=TABULAR_LOCAL_FEATURES_PREDICTIONS_PATH,
    ),
    "local_view_cnn": ModelSpec(
        display_model="local_view_cnn",
        model_name="local_view_cnn",
        predictions_path=LOCAL_CNN_PREDICTIONS_PATH,
    ),
    "global_view_cnn": ModelSpec(
        display_model="global_view_cnn",
        model_name="global_view_cnn",
        predictions_path=GLOBAL_CNN_PREDICTIONS_PATH,
    ),
    "fused_tabular_local_cnn": ModelSpec(
        display_model="fused_tabular_local_cnn",
        model_name="fused_tabular_local_cnn",
        predictions_path=FUSED_PREDICTIONS_PATH,
    ),
    "fused_tabular_local_features_cnn": ModelSpec(
        display_model="fused_tabular_local_features_cnn",
        model_name="fused_tabular_local_features_cnn",
        predictions_path=FUSED_LOCAL_FEATURES_PREDICTIONS_PATH,
    ),
    "stacked_score_logistic_regression": ModelSpec(
        display_model="stacked_score_logistic_regression",
        model_name="stacked_score_logistic_regression",
        predictions_path=STACKED_SCORE_PREDICTIONS_PATH,
    ),
}

PAIR_SPECS = [
    PairSpec(
        pair_id="tabular_vs_fused",
        left=MODEL_SPECS["tabular_logistic_regression"],
        right=MODEL_SPECS["fused_tabular_local_cnn"],
    ),
    PairSpec(
        pair_id="tabular_local_features_vs_fused",
        left=MODEL_SPECS["tabular_local_features_logistic_regression"],
        right=MODEL_SPECS["fused_tabular_local_cnn"],
    ),
    PairSpec(
        pair_id="tabular_vs_fused_local_features",
        left=MODEL_SPECS["tabular_logistic_regression"],
        right=MODEL_SPECS["fused_tabular_local_features_cnn"],
    ),
    PairSpec(
        pair_id="tabular_local_features_vs_fused_local_features",
        left=MODEL_SPECS["tabular_local_features_logistic_regression"],
        right=MODEL_SPECS["fused_tabular_local_features_cnn"],
    ),
    PairSpec(
        pair_id="fused_vs_fused_local_features",
        left=MODEL_SPECS["fused_tabular_local_cnn"],
        right=MODEL_SPECS["fused_tabular_local_features_cnn"],
    ),
    PairSpec(
        pair_id="tabular_local_features_vs_stacked",
        left=MODEL_SPECS["tabular_local_features_logistic_regression"],
        right=MODEL_SPECS["stacked_score_logistic_regression"],
    ),
    PairSpec(
        pair_id="fused_vs_stacked",
        left=MODEL_SPECS["fused_tabular_local_cnn"],
        right=MODEL_SPECS["stacked_score_logistic_regression"],
    ),
    PairSpec(
        pair_id="fused_local_features_vs_stacked",
        left=MODEL_SPECS["fused_tabular_local_features_cnn"],
        right=MODEL_SPECS["stacked_score_logistic_regression"],
    ),
    PairSpec(
        pair_id="local_cnn_vs_stacked",
        left=MODEL_SPECS["local_view_cnn"],
        right=MODEL_SPECS["stacked_score_logistic_regression"],
    ),
    PairSpec(
        pair_id="global_cnn_vs_stacked",
        left=MODEL_SPECS["global_view_cnn"],
        right=MODEL_SPECS["stacked_score_logistic_regression"],
    ),
]


def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"missing required file: {path}")


def sort_frame(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()

    if "pair_id" in frame.columns:
        pair_order = {
            pair.pair_id: idx
            for idx, pair in enumerate(PAIR_SPECS)
        }
        frame["pair_order"] = frame["pair_id"].map(pair_order)

    if "split" in frame.columns:
        frame["split_order"] = frame["split"].map(SPLIT_ORDER)

    if "metric_variant" in frame.columns:
        frame["metric_variant_order"] = frame["metric_variant"].map(METRIC_VARIANT_ORDER)

    if "outcome" in frame.columns:
        frame["outcome_order"] = frame["outcome"].map(OUTCOME_ORDER)

    sort_keys = [
        column
        for column in [
            "pair_order",
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
                "pair_order",
                "split_order",
                "metric_variant_order",
                "outcome_order",
            ]
            if column in frame.columns
        ]
    )

    return frame


def load_predictions(spec: ModelSpec) -> pd.DataFrame:
    require_file(spec.predictions_path)

    frame = pd.read_csv(spec.predictions_path)

    missing = REQUIRED_PREDICTION_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(
            f"{spec.predictions_path} is missing required columns: {sorted(missing)}"
        )

    frame = frame[frame["model"] == spec.model_name].copy()

    if frame.empty:
        raise ValueError(
            f"{spec.predictions_path} has no rows for model={spec.model_name}"
        )

    frame["display_model"] = spec.display_model
    frame["seed"] = frame["seed"].astype(int)
    frame["row_index"] = frame["row_index"].astype(int)
    frame["y_true"] = frame["y_true"].astype(int)
    frame["planet_like_score"] = frame["planet_like_score"].astype(float)
    frame["kepid"] = frame["kepid"].astype(int)
    frame["kepoi_name"] = frame["kepoi_name"].astype(str)

    return frame


def load_thresholds() -> pd.DataFrame:
    require_file(MODEL_COMPARISON_BY_SEED_PATH)

    thresholds = pd.read_csv(MODEL_COMPARISON_BY_SEED_PATH)

    required_columns = {
        "model",
        "metric_variant",
        "seed",
        "threshold",
    }

    missing = required_columns - set(thresholds.columns)
    if missing:
        raise ValueError(
            f"{MODEL_COMPARISON_BY_SEED_PATH} is missing required columns: {sorted(missing)}"
        )

    thresholds = thresholds[
        [
            "model",
            "metric_variant",
            "seed",
            "threshold",
        ]
    ].drop_duplicates().copy()

    thresholds["seed"] = thresholds["seed"].astype(int)
    thresholds["threshold"] = thresholds["threshold"].astype(float)

    return thresholds


def lookup_threshold(
    thresholds: pd.DataFrame,
    model_name: str,
    metric_variant: str,
    seed: int,
) -> float:
    if metric_variant == FIXED_THRESHOLD_VARIANT:
        return 0.5

    matches = thresholds[
        (thresholds["model"] == model_name)
        & (thresholds["metric_variant"] == metric_variant)
        & (thresholds["seed"] == seed)
    ]

    if matches.empty:
        raise ValueError(
            "missing threshold for "
            f"model={model_name}, metric_variant={metric_variant}, seed={seed}"
        )

    unique_thresholds = matches["threshold"].drop_duplicates()

    if unique_thresholds.shape[0] != 1:
        raise ValueError(
            "ambiguous threshold for "
            f"model={model_name}, metric_variant={metric_variant}, seed={seed}: "
            f"{unique_thresholds.tolist()}"
        )

    return float(unique_thresholds.iloc[0])


def load_model_ready_context() -> pd.DataFrame:
    require_file(MODEL_READY_NPZ_PATH)
    require_file(MODEL_READY_MANIFEST_PATH)

    data = np.load(MODEL_READY_NPZ_PATH)
    manifest = pd.read_csv(MODEL_READY_MANIFEST_PATH).reset_index(
        names="row_index"
    )

    labels = data["labels"].astype(int)
    kepid = data["kepid"].astype(int)
    kepoi_name = data["kepoi_name"].astype(str)

    n_rows = labels.shape[0]

    if len(manifest) != n_rows:
        raise ValueError(
            "model-ready manifest row count does not match .npz labels: "
            f"{len(manifest)} vs {n_rows}"
        )

    if not np.array_equal(manifest["row_index"].to_numpy(dtype=int), np.arange(n_rows)):
        raise ValueError("model-ready manifest row_index is not contiguous")

    if not np.array_equal(manifest["kepid"].to_numpy(dtype=int), kepid):
        raise ValueError("model-ready manifest kepid values do not match .npz")

    if not np.array_equal(manifest["kepoi_name"].astype(str).to_numpy(), kepoi_name):
        raise ValueError("model-ready manifest kepoi_name values do not match .npz")

    if not np.array_equal(manifest["binary_label"].to_numpy(dtype=int), labels):
        raise ValueError("model-ready manifest labels do not match .npz")

    local_matrix, local_feature_names = build_local_derived_feature_matrix(data)

    local_features = pd.DataFrame(
        local_matrix,
        columns=local_feature_names.astype(str).tolist(),
    )
    local_features.insert(0, "row_index", np.arange(n_rows))

    context = manifest.merge(
        local_features,
        on="row_index",
        how="left",
        validate="one_to_one",
    )

    return context


def validate_pair_merge(merged: pd.DataFrame, pair: PairSpec) -> None:
    if merged.empty:
        raise ValueError(f"{pair.pair_id}: predictions did not overlap")

    checks = [
        ("y_true", "y_true"),
        ("kepid", "kepid"),
        ("kepoi_name", "kepoi_name"),
    ]

    for left_col, right_col in checks:
        left = merged[f"{left_col}_left"].to_numpy()
        right = merged[f"{right_col}_right"].to_numpy()

        if not np.array_equal(left, right):
            raise ValueError(
                f"{pair.pair_id}: {left_col} values differ after prediction merge"
            )


def build_pair_rows(
    pair: PairSpec,
    left_predictions: pd.DataFrame,
    right_predictions: pd.DataFrame,
    thresholds: pd.DataFrame,
) -> pd.DataFrame:
    merged = left_predictions.merge(
        right_predictions,
        on=[
            "seed",
            "split",
            "row_index",
        ],
        suffixes=("_left", "_right"),
        how="inner",
        validate="one_to_one",
    )

    validate_pair_merge(merged, pair)

    rows = []

    for metric_variant in METRIC_VARIANTS:
        for seed, seed_frame in merged.groupby("seed"):
            left_threshold = lookup_threshold(
                thresholds=thresholds,
                model_name=pair.left.model_name,
                metric_variant=metric_variant,
                seed=int(seed),
            )
            right_threshold = lookup_threshold(
                thresholds=thresholds,
                model_name=pair.right.model_name,
                metric_variant=metric_variant,
                seed=int(seed),
            )

            current = seed_frame.copy()

            current["pair_id"] = pair.pair_id
            current["left_model"] = pair.left.model_name
            current["right_model"] = pair.right.model_name
            current["left_display_model"] = pair.left.display_model
            current["right_display_model"] = pair.right.display_model
            current["metric_variant"] = metric_variant
            current["left_threshold"] = left_threshold
            current["right_threshold"] = right_threshold

            current["y_true"] = current["y_true_left"].astype(int)
            current["kepid"] = current["kepid_left"].astype(int)
            current["kepoi_name"] = current["kepoi_name_left"].astype(str)

            current["left_score"] = current["planet_like_score_left"].astype(float)
            current["right_score"] = current["planet_like_score_right"].astype(float)

            current["left_pred"] = (current["left_score"] >= left_threshold).astype(int)
            current["right_pred"] = (current["right_score"] >= right_threshold).astype(int)

            current["left_correct"] = current["left_pred"] == current["y_true"]
            current["right_correct"] = current["right_pred"] == current["y_true"]

            current["prediction_changed"] = current["left_pred"] != current["right_pred"]
            current["score_delta_right_minus_left"] = (
                current["right_score"] - current["left_score"]
            )
            current["abs_score_delta"] = current[
                "score_delta_right_minus_left"
            ].abs()

            current["outcome"] = np.select(
                [
                    current["left_correct"] & current["right_correct"],
                    (~current["left_correct"]) & (~current["right_correct"]),
                    (~current["left_correct"]) & current["right_correct"],
                    current["left_correct"] & (~current["right_correct"]),
                ],
                [
                    "both_correct",
                    "both_wrong",
                    "right_only_correct",
                    "left_only_correct",
                ],
                default="unknown",
            )

            rows.append(current)

    combined = pd.concat(rows, ignore_index=True)

    keep_columns = [
        "pair_id",
        "left_model",
        "right_model",
        "left_display_model",
        "right_display_model",
        "metric_variant",
        "seed",
        "split",
        "row_index",
        "kepid",
        "kepoi_name",
        "disposition_left",
        "y_true",
        "left_threshold",
        "right_threshold",
        "left_score",
        "right_score",
        "score_delta_right_minus_left",
        "abs_score_delta",
        "left_pred",
        "right_pred",
        "prediction_changed",
        "left_correct",
        "right_correct",
        "outcome",
    ]

    combined = combined[keep_columns].rename(
        columns={
            "disposition_left": "prediction_disposition",
        }
    )

    return combined


def attach_context(rows: pd.DataFrame, context: pd.DataFrame) -> pd.DataFrame:
    extra_columns = [
        column
        for column in context.columns
        if column not in rows.columns
    ]

    context_subset = context[
        [
            "row_index",
            *extra_columns,
        ]
    ].copy()

    return rows.merge(
        context_subset,
        on="row_index",
        how="left",
        validate="many_to_one",
    )


def summarize_pair_errors(rows: pd.DataFrame) -> pd.DataFrame:
    summary_rows = []

    for group_key, group in rows.groupby(
        [
            "pair_id",
            "left_display_model",
            "right_display_model",
            "metric_variant",
            "split",
        ]
    ):
        pair_id, left_display_model, right_display_model, metric_variant, split_name = group_key

        both_correct = int((group["outcome"] == "both_correct").sum())
        both_wrong = int((group["outcome"] == "both_wrong").sum())
        right_only_correct = int((group["outcome"] == "right_only_correct").sum())
        left_only_correct = int((group["outcome"] == "left_only_correct").sum())
        changed_prediction_count = int(group["prediction_changed"].sum())

        left_accuracy = float(group["left_correct"].mean())
        right_accuracy = float(group["right_correct"].mean())

        summary_rows.append(
            {
                "pair_id": pair_id,
                "left_display_model": left_display_model,
                "right_display_model": right_display_model,
                "metric_variant": metric_variant,
                "split": split_name,
                "n": int(group.shape[0]),
                "left_accuracy": left_accuracy,
                "right_accuracy": right_accuracy,
                "accuracy_delta_right_minus_left": right_accuracy - left_accuracy,
                "both_correct": both_correct,
                "both_wrong": both_wrong,
                "right_only_correct": right_only_correct,
                "left_only_correct": left_only_correct,
                "net_right_correct_gain": right_only_correct - left_only_correct,
                "changed_prediction_count": changed_prediction_count,
                "changed_prediction_rate": changed_prediction_count / max(group.shape[0], 1),
            }
        )

    return sort_frame(pd.DataFrame(summary_rows))


def disposition_column(rows: pd.DataFrame) -> str:
    if "koi_disposition" in rows.columns:
        return "koi_disposition"

    if "prediction_disposition" in rows.columns:
        return "prediction_disposition"

    return "disposition"


def summarize_by_disposition(rows: pd.DataFrame) -> pd.DataFrame:
    disp_col = disposition_column(rows)

    summary = (
        rows
        .groupby(
            [
                "pair_id",
                "metric_variant",
                "split",
                disp_col,
                "outcome",
            ],
            as_index=False,
        )
        .agg(
            n=("row_index", "count"),
            mean_left_score=("left_score", "mean"),
            mean_right_score=("right_score", "mean"),
            mean_score_delta_right_minus_left=(
                "score_delta_right_minus_left",
                "mean",
            ),
        )
        .rename(columns={disp_col: "display_disposition"})
    )

    return sort_frame(summary)


def feature_columns(rows: pd.DataFrame) -> list[str]:
    candidates = [
        "n_fits_files",
        "n_clean_points",
        "clean_fraction",
        "time_span_days",
        "global_missing_bin_fraction_before_interp",
        "local_missing_bin_fraction_before_interp",
        "local_window_half_width",
        "global_min_phase",
        "global_min_abs_phase",
        "global_min_flux",
        "local_min_phase",
        "local_min_abs_phase",
        "local_min_flux",
        *LOCAL_DERIVED_FEATURE_NAMES,
    ]

    return [
        column
        for column in candidates
        if column in rows.columns
    ]


def summarize_features(rows: pd.DataFrame) -> pd.DataFrame:
    features = feature_columns(rows)

    summary_rows = []

    for group_key, group in rows.groupby(
        [
            "pair_id",
            "metric_variant",
            "split",
            "outcome",
        ]
    ):
        pair_id, metric_variant, split_name, outcome = group_key

        for feature in features:
            values = pd.to_numeric(group[feature], errors="coerce").dropna()

            if values.empty:
                continue

            summary_rows.append(
                {
                    "pair_id": pair_id,
                    "metric_variant": metric_variant,
                    "split": split_name,
                    "outcome": outcome,
                    "feature": feature,
                    "n": int(values.shape[0]),
                    "mean": float(values.mean()),
                    "std": float(values.std()),
                    "min": float(values.min()),
                    "median": float(values.median()),
                    "max": float(values.max()),
                }
            )

    return sort_frame(pd.DataFrame(summary_rows))


def build_feature_differences(feature_summary: pd.DataFrame) -> pd.DataFrame:
    pivot = (
        feature_summary
        .pivot_table(
            index=[
                "pair_id",
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
        "right_only_correct",
        "left_only_correct",
        "both_wrong",
    ]:
        if outcome not in pivot.columns:
            pivot[outcome] = np.nan

    pivot = pivot.rename(
        columns={
            "right_only_correct": "right_only_correct_mean",
            "left_only_correct": "left_only_correct_mean",
            "both_wrong": "both_wrong_mean",
        }
    )

    pivot["mean_diff_right_only_minus_left_only"] = (
        pivot["right_only_correct_mean"]
        - pivot["left_only_correct_mean"]
    )
    pivot["abs_mean_diff_right_only_vs_left_only"] = (
        pivot["mean_diff_right_only_minus_left_only"].abs()
    )

    pivot = pivot.sort_values(
        [
            "pair_id",
            "split",
            "metric_variant",
            "abs_mean_diff_right_only_vs_left_only",
        ],
        ascending=[
            True,
            True,
            True,
            False,
        ],
    ).reset_index(drop=True)

    return sort_frame(pivot)


def changed_rows(rows: pd.DataFrame) -> pd.DataFrame:
    changed = rows[
        rows["prediction_changed"]
        | (rows["left_correct"] != rows["right_correct"])
    ].copy()

    if "koi_disposition" in changed.columns:
        changed["display_disposition"] = changed["koi_disposition"]
    elif "prediction_disposition" in changed.columns:
        changed["display_disposition"] = changed["prediction_disposition"]
    else:
        changed["display_disposition"] = "unknown"

    return changed


def build_recurring_changed_rows(changed: pd.DataFrame) -> pd.DataFrame:
    recurring = (
        changed
        .groupby(
            [
                "pair_id",
                "metric_variant",
                "row_index",
                "kepid",
                "kepoi_name",
                "display_disposition",
                "y_true",
                "outcome",
            ],
            as_index=False,
        )
        .agg(
            seed_count=("seed", "nunique"),
            mean_left_score=("left_score", "mean"),
            mean_right_score=("right_score", "mean"),
            mean_score_delta_right_minus_left=(
                "score_delta_right_minus_left",
                "mean",
            ),
            mean_abs_score_delta=("abs_score_delta", "mean"),
            mean_left_pred=("left_pred", "mean"),
            mean_right_pred=("right_pred", "mean"),
        )
    )

    recurring = recurring.sort_values(
        [
            "pair_id",
            "metric_variant",
            "outcome",
            "seed_count",
            "mean_abs_score_delta",
        ],
        ascending=[
            True,
            True,
            True,
            False,
            False,
        ],
    ).reset_index(drop=True)

    return sort_frame(recurring)


def build_strongest_disagreements(changed: pd.DataFrame) -> pd.DataFrame:
    strongest = changed.sort_values(
        [
            "pair_id",
            "split",
            "metric_variant",
            "abs_score_delta",
        ],
        ascending=[
            True,
            True,
            True,
            False,
        ],
    ).reset_index(drop=True)

    return sort_frame(strongest)


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
        description="Compare pairwise model errors across current prediction files."
    )

    parser.add_argument(
        "--top-n",
        type=int,
        default=12,
        help="Number of detail rows to print per section.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    thresholds = load_thresholds()
    context = load_model_ready_context()

    prediction_cache: dict[str, pd.DataFrame] = {}

    for spec in MODEL_SPECS.values():
        cache_key = spec.display_model

        if cache_key not in prediction_cache:
            prediction_cache[cache_key] = load_predictions(spec)

    pair_frames = []

    for pair in PAIR_SPECS:
        pair_rows = build_pair_rows(
            pair=pair,
            left_predictions=prediction_cache[pair.left.display_model],
            right_predictions=prediction_cache[pair.right.display_model],
            thresholds=thresholds,
        )
        pair_frames.append(pair_rows)

    rows = pd.concat(pair_frames, ignore_index=True)
    rows = attach_context(rows=rows, context=context)

    summary = summarize_pair_errors(rows)
    changed = changed_rows(rows)
    disposition_summary = summarize_by_disposition(rows)
    feature_summary = summarize_features(rows)
    feature_differences = build_feature_differences(feature_summary)
    recurring = build_recurring_changed_rows(changed)
    strongest = build_strongest_disagreements(changed)

    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    summary.to_csv(PAIRWISE_SUMMARY_PATH, index=False)
    changed.to_csv(PAIRWISE_CHANGED_PATH, index=False)
    disposition_summary.to_csv(PAIRWISE_DISPOSITION_PATH, index=False)
    feature_summary.to_csv(PAIRWISE_FEATURE_SUMMARY_PATH, index=False)
    feature_differences.to_csv(PAIRWISE_FEATURE_DIFFERENCES_PATH, index=False)
    recurring.to_csv(PAIRWISE_RECURRING_PATH, index=False)
    strongest.to_csv(PAIRWISE_STRONGEST_PATH, index=False)

    test_summary = summary[summary["split"] == "test"].copy()

    print_table(
        title="pairwise test-set error summary:",
        frame=test_summary,
        columns=SUMMARY_COLUMNS,
    )

    strict_pair = "fused_vs_stacked"
    
    strict_changed = changed[
        (changed["pair_id"] == strict_pair)
        & (changed["split"] == "test")
    ].copy()

    strict_disposition = (
        strict_changed
        .groupby(
            [
                "pair_id",
                "metric_variant",
                "display_disposition",
                "outcome",
            ],
            as_index=False,
        )
        .size()
        .rename(columns={"size": "n"})
    )

    print_table(
        title="strict pair changed test predictions by disposition:",
        frame=sort_frame(strict_disposition),
        columns=[
            "pair_id",
            "metric_variant",
            "display_disposition",
            "outcome",
            "n",
        ],
    )

    strict_features = feature_differences[
        (feature_differences["pair_id"] == strict_pair)
        & (feature_differences["split"] == "test")
    ].copy()

    print_table(
        title=(
            f"top {args.top_n} strict-pair feature differences: "
            "right-only correct vs left-only correct"
        ),
        frame=strict_features,
        columns=FEATURE_DIFFERENCE_COLUMNS,
        top_n=args.top_n,
    )

    strict_recurring = recurring[
        recurring["pair_id"] == strict_pair
    ].copy()

    print_table(
        title=f"top {args.top_n} strict-pair recurring changed rows:",
        frame=strict_recurring,
        columns=RECURRING_COLUMNS,
        top_n=args.top_n,
    )

    strict_strongest = strongest[
        (strongest["pair_id"] == strict_pair)
        & (strongest["split"] == "test")
    ].copy()

    print_table(
        title=f"top {args.top_n} strict-pair strongest individual test disagreements:",
        frame=strict_strongest,
        columns=STRONGEST_COLUMNS,
        top_n=args.top_n,
    )

    print("wrote:", PAIRWISE_SUMMARY_PATH)
    print("wrote:", PAIRWISE_CHANGED_PATH)
    print("wrote:", PAIRWISE_DISPOSITION_PATH)
    print("wrote:", PAIRWISE_FEATURE_SUMMARY_PATH)
    print("wrote:", PAIRWISE_FEATURE_DIFFERENCES_PATH)
    print("wrote:", PAIRWISE_RECURRING_PATH)
    print("wrote:", PAIRWISE_STRONGEST_PATH)
    print("COMPARE_PAIRWISE_MODEL_ERRORS_OK")


if __name__ == "__main__":
    main()