from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from kepler_vetting.modeling.thresholds import (
    FIXED_THRESHOLD_VARIANT,
    VAL_TUNED_F1_THRESHOLD_VARIANT,
    evaluate_binary_scores,
    select_best_f1_threshold,
)
from kepler_vetting.processing.common import MODEL_READY_NPZ_PATH


METRICS_DIR = Path("outputs/metrics")

TABULAR_PREDICTIONS_PATH = METRICS_DIR / "tabular_baseline_predictions.csv"
TABULAR_LOCAL_FEATURES_PREDICTIONS_PATH = (
    METRICS_DIR / "tabular_local_features_predictions.csv"
)
LOCAL_LIGHTCURVE_PREDICTIONS_PATH = METRICS_DIR / "lightcurve_cnn_predictions.csv"
GLOBAL_LIGHTCURVE_PREDICTIONS_PATH = METRICS_DIR / "global_lightcurve_cnn_predictions.csv"
FUSED_LOCAL_PREDICTIONS_PATH = METRICS_DIR / "fused_local_model_predictions.csv"
FUSED_LOCAL_FEATURES_PREDICTIONS_PATH = (
    METRICS_DIR / "fused_local_features_model_predictions.csv"
)
STACKED_SCORE_PREDICTIONS_PATH = METRICS_DIR / "stacked_score_model_predictions.csv"

COMPARISON_BY_SEED_PATH = METRICS_DIR / "model_comparison_by_seed.csv"
COMPARISON_PATH = METRICS_DIR / "model_comparison.csv"


PREDICTION_SOURCES = [
    {
        "family": "tabular",
        "path": TABULAR_PREDICTIONS_PATH,
        "models": [
            "dummy_most_frequent",
            "logistic_regression",
        ],
    },
    {
        "family": "tabular_local_features",
        "path": TABULAR_LOCAL_FEATURES_PREDICTIONS_PATH,
        "models": [
            "tabular_local_features_logistic_regression",
        ],
    },
    {
        "family": "local_lightcurve",
        "path": LOCAL_LIGHTCURVE_PREDICTIONS_PATH,
        "models": [
            "local_view_cnn",
        ],
    },
    {
        "family": "global_lightcurve",
        "path": GLOBAL_LIGHTCURVE_PREDICTIONS_PATH,
        "models": [
            "global_view_cnn",
        ],
    },
    {
        "family": "fused_local",
        "path": FUSED_LOCAL_PREDICTIONS_PATH,
        "models": [
            "fused_tabular_local_cnn",
        ],
    },
    {
        "family": "fused_local_features",
        "path": FUSED_LOCAL_FEATURES_PREDICTIONS_PATH,
        "models": [
            "fused_tabular_local_features_cnn",
        ],
    },
    {
        "family": "stacked_score",
        "path": STACKED_SCORE_PREDICTIONS_PATH,
        "models": [
            "stacked_score_logistic_regression",
        ],
    },
]


DISPLAY_NAMES = {
    "dummy_most_frequent": "dummy_most_frequent",
    "logistic_regression": "tabular_logistic_regression",
    "tabular_local_features_logistic_regression": (
        "tabular_local_features_logistic_regression"
    ),
    "local_view_cnn": "local_view_cnn",
    "global_view_cnn": "global_view_cnn",
    "fused_tabular_local_cnn": "fused_tabular_local_cnn",
    "fused_tabular_local_features_cnn": "fused_tabular_local_features_cnn",
    "stacked_score_logistic_regression": "stacked_score_logistic_regression",
}

MODEL_ORDER = {
    "dummy_most_frequent": 0,
    "tabular_logistic_regression": 1,
    "tabular_local_features_logistic_regression": 2,
    "local_view_cnn": 3,
    "global_view_cnn": 4,
    "fused_tabular_local_cnn": 5,
    "fused_tabular_local_features_cnn": 6,
    "stacked_score_logistic_regression": 7,
}

METRIC_VARIANT_ORDER = {
    FIXED_THRESHOLD_VARIANT: 0,
    VAL_TUNED_F1_THRESHOLD_VARIANT: 1,
}

SPLIT_ORDER = {
    "test": 0,
    "val": 1,
    "train": 2,
}

REQUIRED_PREDICTION_COLUMNS = {
    "model",
    "seed",
    "split",
    "row_index",
    "kepid",
    "kepoi_name",
    "y_true",
    "planet_like_score",
}

SUMMARY_COLUMNS = [
    "threshold",
    "accuracy",
    "precision",
    "recall",
    "f1",
    "roc_auc",
    "tn",
    "fp",
    "fn",
    "tp",
]

COMPACT_COLUMNS = [
    "display_model",
    "metric_variant",
    "split",
    "threshold_mean",
    "accuracy_mean",
    "accuracy_std",
    "f1_mean",
    "f1_std",
    "roc_auc_mean",
    "roc_auc_std",
    "precision_mean",
    "recall_mean",
    "tn_mean",
    "fp_mean",
    "fn_mean",
    "tp_mean",
]


def load_current_model_ready_dataset() -> dict[str, np.ndarray]:
    if not MODEL_READY_NPZ_PATH.exists():
        raise FileNotFoundError(
            f"missing model-ready dataset: {MODEL_READY_NPZ_PATH}. "
            "Run kepler_vetting.processing.filter_model_ready_dataset first."
        )

    data = np.load(MODEL_READY_NPZ_PATH)

    required_arrays = [
        "labels",
        "kepid",
        "kepoi_name",
    ]

    missing = [
        name
        for name in required_arrays
        if name not in data.files
    ]

    if missing:
        raise ValueError(
            f"{MODEL_READY_NPZ_PATH} is missing required arrays: {missing}"
        )

    labels = data["labels"].astype(int)
    kepid = data["kepid"].astype(int)
    kepoi_name = data["kepoi_name"].astype(str)

    n_rows = labels.shape[0]

    if kepid.shape[0] != n_rows:
        raise ValueError(
            "model-ready kepid row count does not match labels: "
            f"{kepid.shape[0]} vs {n_rows}"
        )

    if kepoi_name.shape[0] != n_rows:
        raise ValueError(
            "model-ready kepoi_name row count does not match labels: "
            f"{kepoi_name.shape[0]} vs {n_rows}"
        )

    return {
        "labels": labels,
        "kepid": kepid,
        "kepoi_name": kepoi_name,
    }


def load_predictions(
    path: Path,
    family: str,
    models: list[str],
    current_dataset: dict[str, np.ndarray],
) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"missing predictions file: {path}. "
            "Run the corresponding training script first."
        )

    frame = pd.read_csv(path)
    frame["family"] = family

    missing = REQUIRED_PREDICTION_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(
            f"{path} is missing required columns: {sorted(missing)}"
        )

    frame = frame[frame["model"].isin(models)].copy()

    if frame.empty:
        raise ValueError(
            f"{path} did not contain any requested models: {models}"
        )

    missing_models = sorted(set(models) - set(frame["model"].unique()))
    if missing_models:
        raise ValueError(
            f"{path} is missing requested models: {missing_models}"
        )

    validate_predictions_against_current_dataset(
        frame=frame,
        path=path,
        models=models,
        current_dataset=current_dataset,
    )

    return frame


def validate_predictions_against_current_dataset(
    frame: pd.DataFrame,
    path: Path,
    models: list[str],
    current_dataset: dict[str, np.ndarray],
) -> None:
    labels = current_dataset["labels"]
    kepid = current_dataset["kepid"]
    kepoi_name = current_dataset["kepoi_name"]

    n_rows = labels.shape[0]
    expected_row_index = np.arange(n_rows)

    bad_splits = sorted(set(frame["split"].astype(str)) - set(SPLIT_ORDER))
    if bad_splits:
        raise ValueError(
            f"{path} contains unknown split names: {bad_splits}"
        )

    for model in models:
        model_frame = frame[frame["model"] == model]

        if model_frame.empty:
            raise ValueError(f"{path} has no rows for model={model}")

        for seed, group in model_frame.groupby("seed"):
            if len(group) != n_rows:
                raise ValueError(
                    f"{path} appears stale or incomplete for "
                    f"model={model}, seed={seed}: "
                    f"found {len(group)} prediction rows, expected {n_rows}. "
                    "Rerun the corresponding training script against the current "
                    "model-ready dataset."
                )

            row_index_values = pd.to_numeric(
                group["row_index"],
                errors="coerce",
            )

            if row_index_values.isna().any():
                raise ValueError(
                    f"{path} contains non-numeric row_index values for "
                    f"model={model}, seed={seed}"
                )

            row_index = row_index_values.astype(int).to_numpy()

            if np.any(row_index < 0) or np.any(row_index >= n_rows):
                raise ValueError(
                    f"{path} contains row_index values outside the current "
                    f"dataset range for model={model}, seed={seed}"
                )

            if len(np.unique(row_index)) != n_rows:
                raise ValueError(
                    f"{path} contains duplicate or missing row_index values for "
                    f"model={model}, seed={seed}"
                )

            if not np.array_equal(np.sort(row_index), expected_row_index):
                raise ValueError(
                    f"{path} does not cover exactly the current model-ready "
                    f"dataset rows for model={model}, seed={seed}"
                )

            y_true = pd.to_numeric(
                group["y_true"],
                errors="coerce",
            )

            if y_true.isna().any():
                raise ValueError(
                    f"{path} contains non-numeric y_true values for "
                    f"model={model}, seed={seed}"
                )

            y_true_array = y_true.astype(int).to_numpy()
            if not np.array_equal(y_true_array, labels[row_index]):
                raise ValueError(
                    f"{path} y_true values do not match the current "
                    f"model-ready labels for model={model}, seed={seed}. "
                    "Rerun the corresponding training script."
                )

            kepid_values = pd.to_numeric(
                group["kepid"],
                errors="coerce",
            )

            if kepid_values.isna().any():
                raise ValueError(
                    f"{path} contains non-numeric kepid values for "
                    f"model={model}, seed={seed}"
                )

            kepid_array = kepid_values.astype(int).to_numpy()
            if not np.array_equal(kepid_array, kepid[row_index]):
                raise ValueError(
                    f"{path} kepid values do not match the current "
                    f"model-ready dataset for model={model}, seed={seed}. "
                    "Rerun the corresponding training script."
                )

            kepoi_array = group["kepoi_name"].astype(str).to_numpy()
            if not np.array_equal(kepoi_array, kepoi_name[row_index]):
                raise ValueError(
                    f"{path} kepoi_name values do not match the current "
                    f"model-ready dataset for model={model}, seed={seed}. "
                    "Rerun the corresponding training script."
                )


def load_all_predictions() -> pd.DataFrame:
    current_dataset = load_current_model_ready_dataset()

    frames = [
        load_predictions(
            path=source["path"],
            family=source["family"],
            models=source["models"],
            current_dataset=current_dataset,
        )
        for source in PREDICTION_SOURCES
    ]

    predictions = pd.concat(
        frames,
        ignore_index=True,
    )

    predictions["display_model"] = predictions["model"].map(DISPLAY_NAMES)

    if predictions["display_model"].isna().any():
        missing = sorted(
            predictions.loc[
                predictions["display_model"].isna(),
                "model",
            ].unique()
        )
        raise ValueError(f"missing display names for models: {missing}")

    print(
        "validated prediction files against current model-ready dataset:",
        MODEL_READY_NPZ_PATH,
    )
    print("current_model_ready_rows:", current_dataset["labels"].shape[0])
    print()

    return predictions


def threshold_variants_for_group(group: pd.DataFrame) -> list[tuple[str, float]]:
    model_name = str(group["model"].iloc[0])

    variants = [
        (
            FIXED_THRESHOLD_VARIANT,
            0.5,
        )
    ]

    if model_name == "dummy_most_frequent":
        return variants

    val = group[group["split"] == "val"]

    if val.empty:
        raise ValueError(f"missing validation rows for model={model_name}")

    tuned_threshold = select_best_f1_threshold(
        y_true=val["y_true"].to_numpy(),
        y_score=val["planet_like_score"].to_numpy(),
    )

    variants.append(
        (
            VAL_TUNED_F1_THRESHOLD_VARIANT,
            tuned_threshold,
        )
    )

    return variants


def build_by_seed_comparison(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []

    group_columns = [
        "family",
        "model",
        "display_model",
        "seed",
    ]

    for group_key, group in predictions.groupby(group_columns):
        family, model_name, display_model, seed = group_key

        for metric_variant, threshold in threshold_variants_for_group(group):
            for split_name, split in group.groupby("split"):
                record = evaluate_binary_scores(
                    model_name=model_name,
                    metric_variant=metric_variant,
                    seed=int(seed),
                    split_name=split_name,
                    y_true=split["y_true"].to_numpy(),
                    y_score=split["planet_like_score"].to_numpy(),
                    threshold=threshold,
                )

                record["family"] = family
                record["display_model"] = display_model

                rows.append(record)

    by_seed = pd.DataFrame(rows)

    by_seed["model_order"] = by_seed["display_model"].map(MODEL_ORDER)
    by_seed["metric_variant_order"] = by_seed["metric_variant"].map(METRIC_VARIANT_ORDER)
    by_seed["split_order"] = by_seed["split"].map(SPLIT_ORDER)

    if by_seed["model_order"].isna().any():
        missing = sorted(
            by_seed.loc[
                by_seed["model_order"].isna(),
                "display_model",
            ].unique()
        )
        raise ValueError(f"missing model order for models: {missing}")

    if by_seed["metric_variant_order"].isna().any():
        missing = sorted(
            by_seed.loc[
                by_seed["metric_variant_order"].isna(),
                "metric_variant",
            ].unique()
        )
        raise ValueError(f"missing metric variant order for variants: {missing}")

    if by_seed["split_order"].isna().any():
        missing = sorted(
            by_seed.loc[
                by_seed["split_order"].isna(),
                "split",
            ].unique()
        )
        raise ValueError(f"missing split order for splits: {missing}")

    by_seed = by_seed.sort_values(
        [
            "split_order",
            "model_order",
            "metric_variant_order",
            "seed",
        ]
    ).reset_index(drop=True)

    return by_seed


def summarize_by_seed(by_seed: pd.DataFrame) -> pd.DataFrame:
    summary = (
        by_seed
        .groupby(
            [
                "family",
                "display_model",
                "metric_variant",
                "split",
                "model_order",
                "metric_variant_order",
                "split_order",
            ]
        )[SUMMARY_COLUMNS]
        .agg(["mean", "std", "min", "max"])
        .reset_index()
    )

    summary.columns = [
        "_".join(col).rstrip("_")
        for col in summary.columns.to_flat_index()
    ]

    summary = summary.sort_values(
        [
            "split_order",
            "model_order",
            "metric_variant_order",
        ]
    ).reset_index(drop=True)

    return summary


def main() -> None:
    predictions = load_all_predictions()

    by_seed = build_by_seed_comparison(predictions)
    summary = summarize_by_seed(by_seed)

    COMPARISON_PATH.parent.mkdir(parents=True, exist_ok=True)

    by_seed.to_csv(COMPARISON_BY_SEED_PATH, index=False)
    summary.to_csv(COMPARISON_PATH, index=False)

    test_comparison = summary[summary["split"] == "test"]

    print("test-set model comparison:")
    print(
        test_comparison[COMPACT_COLUMNS].to_string(
            index=False,
            float_format=lambda value: f"{value:.3f}",
        )
    )

    print()
    print("wrote:", COMPARISON_BY_SEED_PATH)
    print("wrote:", COMPARISON_PATH)
    print("COMPARE_MODEL_METRICS_OK")


if __name__ == "__main__":
    main()