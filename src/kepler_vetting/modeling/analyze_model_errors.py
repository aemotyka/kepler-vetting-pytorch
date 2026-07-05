from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from kepler_vetting.modeling.thresholds import (
    FIXED_THRESHOLD_VARIANT,
    VAL_TUNED_F1_THRESHOLD_VARIANT,
)


METRICS_DIR = Path("outputs/metrics")
PROCESSED_DIR = Path("data/processed")

TABULAR_PREDICTIONS_PATH = METRICS_DIR / "tabular_baseline_predictions.csv"
FUSED_PREDICTIONS_PATH = METRICS_DIR / "fused_local_model_predictions.csv"
MODEL_COMPARISON_BY_SEED_PATH = METRICS_DIR / "model_comparison_by_seed.csv"
MODEL_READY_MANIFEST_PATH = PROCESSED_DIR / "model_ready_manifest.csv"

ERROR_SUMMARY_PATH = METRICS_DIR / "tabular_vs_fused_error_summary.csv"
CHANGED_PREDICTIONS_PATH = METRICS_DIR / "tabular_vs_fused_changed_predictions.csv"
DISPOSITION_SUMMARY_PATH = METRICS_DIR / "tabular_vs_fused_disposition_summary.csv"
FEATURE_SUMMARY_PATH = METRICS_DIR / "tabular_vs_fused_feature_summary.csv"


TABULAR_MODEL = "logistic_regression"
FUSED_MODEL = "fused_tabular_local_cnn"

METRIC_VARIANTS = [
    FIXED_THRESHOLD_VARIANT,
    VAL_TUNED_F1_THRESHOLD_VARIANT,
]

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

QUALITY_FEATURES = [
    "n_fits_files",
    "n_clean_points",
    "clean_fraction",
    "time_span_days",
    "global_missing_bin_fraction_before_interp",
    "local_missing_bin_fraction_before_interp",
    "local_window_half_width",
]

COMPACT_SUMMARY_COLUMNS = [
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


def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"missing required file: {path}")


def load_predictions(path: Path, model_name: str) -> pd.DataFrame:
    require_file(path)

    frame = pd.read_csv(path)

    required_columns = {
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

    missing = required_columns - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")

    frame = frame[frame["model"] == model_name].copy()

    if frame.empty:
        raise ValueError(f"{path} has no rows for model={model_name}")

    frame["seed"] = frame["seed"].astype(int)
    frame["row_index"] = frame["row_index"].astype(int)
    frame["y_true"] = frame["y_true"].astype(int)
    frame["planet_like_score"] = frame["planet_like_score"].astype(float)

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

    thresholds = (
        thresholds[
            thresholds["model"].isin(
                [
                    TABULAR_MODEL,
                    FUSED_MODEL,
                ]
            )
        ][
            [
                "model",
                "metric_variant",
                "seed",
                "threshold",
            ]
        ]
        .drop_duplicates()
        .copy()
    )

    if thresholds.empty:
        raise ValueError("threshold table has no tabular/fused rows")

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

    return float(matches["threshold"].iloc[0])


def load_manifest() -> pd.DataFrame:
    require_file(MODEL_READY_MANIFEST_PATH)

    manifest = pd.read_csv(MODEL_READY_MANIFEST_PATH)
    manifest = manifest.reset_index().rename(columns={"index": "row_index"})
    manifest["row_index"] = manifest["row_index"].astype(int)

    return manifest


def build_variant_rows(
    tabular: pd.DataFrame,
    fused: pd.DataFrame,
    thresholds: pd.DataFrame,
) -> pd.DataFrame:
    merged = tabular.merge(
        fused,
        on=[
            "seed",
            "split",
            "row_index",
        ],
        suffixes=("_tabular", "_fused"),
        how="inner",
        validate="one_to_one",
    )

    if merged.empty:
        raise ValueError("tabular/fused predictions did not overlap")

    if not (
        merged["y_true_tabular"].to_numpy() == merged["y_true_fused"].to_numpy()
    ).all():
        raise ValueError("tabular/fused y_true values differ after merge")

    rows = []

    for metric_variant in METRIC_VARIANTS:
        for seed, seed_frame in merged.groupby("seed"):
            tabular_threshold = lookup_threshold(
                thresholds=thresholds,
                model_name=TABULAR_MODEL,
                metric_variant=metric_variant,
                seed=int(seed),
            )
            fused_threshold = lookup_threshold(
                thresholds=thresholds,
                model_name=FUSED_MODEL,
                metric_variant=metric_variant,
                seed=int(seed),
            )

            current = seed_frame.copy()

            current["metric_variant"] = metric_variant
            current["tabular_threshold"] = tabular_threshold
            current["fused_threshold"] = fused_threshold

            current["y_true"] = current["y_true_tabular"].astype(int)

            current["tabular_score"] = current["planet_like_score_tabular"].astype(
                float
            )
            current["fused_score"] = current["planet_like_score_fused"].astype(float)

            current["tabular_pred"] = (
                current["tabular_score"] >= tabular_threshold
            ).astype(int)
            current["fused_pred"] = (current["fused_score"] >= fused_threshold).astype(
                int
            )

            current["tabular_correct"] = current["tabular_pred"] == current["y_true"]
            current["fused_correct"] = current["fused_pred"] == current["y_true"]

            current["prediction_changed"] = (
                current["tabular_pred"] != current["fused_pred"]
            )

            current["score_delta_fused_minus_tabular"] = (
                current["fused_score"] - current["tabular_score"]
            )

            current["outcome"] = np.select(
                [
                    current["tabular_correct"] & current["fused_correct"],
                    (~current["tabular_correct"]) & (~current["fused_correct"]),
                    (~current["tabular_correct"]) & current["fused_correct"],
                    current["tabular_correct"] & (~current["fused_correct"]),
                ],
                [
                    "both_correct",
                    "both_wrong",
                    "fused_only_correct",
                    "tabular_only_correct",
                ],
                default="unknown",
            )

            rows.append(current)

    combined = pd.concat(rows, ignore_index=True)

    keep_columns = [
        "metric_variant",
        "seed",
        "split",
        "row_index",
        "kepid_tabular",
        "kepoi_name_tabular",
        "disposition_tabular",
        "y_true",
        "tabular_threshold",
        "fused_threshold",
        "tabular_score",
        "fused_score",
        "score_delta_fused_minus_tabular",
        "tabular_pred",
        "fused_pred",
        "prediction_changed",
        "tabular_correct",
        "fused_correct",
        "outcome",
    ]

    combined = combined[keep_columns].rename(
        columns={
            "kepid_tabular": "kepid",
            "kepoi_name_tabular": "kepoi_name",
            "disposition_tabular": "disposition",
        }
    )

    return combined


def attach_manifest(
    comparison_rows: pd.DataFrame, manifest: pd.DataFrame
) -> pd.DataFrame:
    extra_columns = [
        column for column in manifest.columns if column not in comparison_rows.columns
    ]

    manifest_subset = manifest[
        [
            "row_index",
            *extra_columns,
        ]
    ].copy()

    return comparison_rows.merge(
        manifest_subset,
        on="row_index",
        how="left",
        validate="many_to_one",
    )


def summarize_errors(comparison_rows: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for (metric_variant, split_name), group in comparison_rows.groupby(
        [
            "metric_variant",
            "split",
        ]
    ):
        both_correct = int((group["outcome"] == "both_correct").sum())
        both_wrong = int((group["outcome"] == "both_wrong").sum())
        fused_only_correct = int((group["outcome"] == "fused_only_correct").sum())
        tabular_only_correct = int((group["outcome"] == "tabular_only_correct").sum())

        tabular_accuracy = float(group["tabular_correct"].mean())
        fused_accuracy = float(group["fused_correct"].mean())

        changed_prediction_count = int(group["prediction_changed"].sum())

        rows.append(
            {
                "metric_variant": metric_variant,
                "split": split_name,
                "n": int(group.shape[0]),
                "tabular_accuracy": tabular_accuracy,
                "fused_accuracy": fused_accuracy,
                "accuracy_delta": fused_accuracy - tabular_accuracy,
                "both_correct": both_correct,
                "both_wrong": both_wrong,
                "fused_only_correct": fused_only_correct,
                "tabular_only_correct": tabular_only_correct,
                "net_fused_correct_gain": fused_only_correct - tabular_only_correct,
                "changed_prediction_count": changed_prediction_count,
                "changed_prediction_rate": changed_prediction_count
                / max(group.shape[0], 1),
            }
        )

    summary = pd.DataFrame(rows)

    split_order = {
        "test": 0,
        "val": 1,
        "train": 2,
    }

    variant_order = {
        FIXED_THRESHOLD_VARIANT: 0,
        VAL_TUNED_F1_THRESHOLD_VARIANT: 1,
    }

    summary["split_order"] = summary["split"].map(split_order)
    summary["metric_variant_order"] = summary["metric_variant"].map(variant_order)

    summary = summary.sort_values(
        [
            "split_order",
            "metric_variant_order",
        ]
    ).drop(
        columns=[
            "split_order",
            "metric_variant_order",
        ]
    )

    return summary


def summarize_by_disposition(comparison_rows: pd.DataFrame) -> pd.DataFrame:
    disposition_column = "koi_disposition"

    if disposition_column not in comparison_rows.columns:
        disposition_column = "disposition"

    rows = []

    group_columns = [
        "metric_variant",
        "split",
        disposition_column,
        "outcome",
    ]

    for group_key, group in comparison_rows.groupby(group_columns):
        metric_variant, split_name, disposition, outcome = group_key

        rows.append(
            {
                "metric_variant": metric_variant,
                "split": split_name,
                "disposition": disposition,
                "outcome": outcome,
                "n": int(group.shape[0]),
                "mean_tabular_score": float(group["tabular_score"].mean()),
                "mean_fused_score": float(group["fused_score"].mean()),
                "mean_score_delta_fused_minus_tabular": float(
                    group["score_delta_fused_minus_tabular"].mean()
                ),
            }
        )

    return pd.DataFrame(rows).sort_values(
        [
            "metric_variant",
            "split",
            "disposition",
            "outcome",
        ]
    )


def summarize_features(comparison_rows: pd.DataFrame) -> pd.DataFrame:
    available_features = [
        feature
        for feature in [
            *TABULAR_FEATURES,
            *QUALITY_FEATURES,
        ]
        if feature in comparison_rows.columns
    ]

    rows = []

    for (metric_variant, split_name, outcome), group in comparison_rows.groupby(
        [
            "metric_variant",
            "split",
            "outcome",
        ]
    ):
        for feature in available_features:
            values = pd.to_numeric(group[feature], errors="coerce").dropna()

            if values.empty:
                continue

            rows.append(
                {
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

    return pd.DataFrame(rows).sort_values(
        [
            "metric_variant",
            "split",
            "outcome",
            "feature",
        ]
    )


def main() -> None:
    tabular = load_predictions(
        path=TABULAR_PREDICTIONS_PATH,
        model_name=TABULAR_MODEL,
    )
    fused = load_predictions(
        path=FUSED_PREDICTIONS_PATH,
        model_name=FUSED_MODEL,
    )
    thresholds = load_thresholds()
    manifest = load_manifest()

    comparison_rows = build_variant_rows(
        tabular=tabular,
        fused=fused,
        thresholds=thresholds,
    )

    comparison_rows = attach_manifest(
        comparison_rows=comparison_rows,
        manifest=manifest,
    )

    error_summary = summarize_errors(comparison_rows)
    disposition_summary = summarize_by_disposition(comparison_rows)
    feature_summary = summarize_features(comparison_rows)

    changed_predictions = comparison_rows[
        comparison_rows["prediction_changed"]
        | (comparison_rows["tabular_correct"] != comparison_rows["fused_correct"])
    ].copy()

    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    error_summary.to_csv(ERROR_SUMMARY_PATH, index=False)
    changed_predictions.to_csv(CHANGED_PREDICTIONS_PATH, index=False)
    disposition_summary.to_csv(DISPOSITION_SUMMARY_PATH, index=False)
    feature_summary.to_csv(FEATURE_SUMMARY_PATH, index=False)

    test_summary = error_summary[error_summary["split"] == "test"]

    print("test-set tabular vs fused error summary:")
    print(
        test_summary[COMPACT_SUMMARY_COLUMNS].to_string(
            index=False,
            float_format=lambda value: f"{value:.3f}",
        )
    )

    print()
    print("changed test predictions:")
    print(
        changed_predictions[changed_predictions["split"] == "test"]
        .groupby(["metric_variant", "outcome"])
        .size()
        .reset_index(name="n")
        .to_string(index=False)
    )

    print()
    print("wrote:", ERROR_SUMMARY_PATH)
    print("wrote:", CHANGED_PREDICTIONS_PATH)
    print("wrote:", DISPOSITION_SUMMARY_PATH)
    print("wrote:", FEATURE_SUMMARY_PATH)
    print("ANALYZE_MODEL_ERRORS_OK")


if __name__ == "__main__":
    main()
