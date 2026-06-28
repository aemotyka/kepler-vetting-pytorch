from __future__ import annotations

from pathlib import Path

import pandas as pd


METRICS_DIR = Path("outputs/metrics")

TABULAR_SUMMARY_PATH = METRICS_DIR / "tabular_baseline_metrics_summary.csv"
LIGHTCURVE_SUMMARY_PATH = METRICS_DIR / "lightcurve_cnn_metrics_summary.csv"

COMPARISON_PATH = METRICS_DIR / "model_comparison.csv"


DISPLAY_NAMES = {
    "dummy_most_frequent": "dummy_most_frequent",
    "logistic_regression": "tabular_logistic_regression",
    "local_view_cnn": "local_view_cnn",
}

MODEL_ORDER = {
    "dummy_most_frequent": 0,
    "tabular_logistic_regression": 1,
    "local_view_cnn": 2,
}

SPLIT_ORDER = {
    "test": 0,
    "val": 1,
    "train": 2,
}


COMPACT_COLUMNS = [
    "display_model",
    "split",
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


def load_summary(path: Path, family: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"missing metrics summary: {path}. "
            "Run the corresponding training script first."
        )

    frame = pd.read_csv(path)
    frame["family"] = family

    required_columns = {
        "model",
        "split",
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
    }

    missing = required_columns - set(frame.columns)
    if missing:
        raise ValueError(
            f"{path} is missing required columns: {sorted(missing)}"
        )

    return frame


def build_comparison() -> pd.DataFrame:
    tabular = load_summary(
        path=TABULAR_SUMMARY_PATH,
        family="tabular",
    )

    lightcurve = load_summary(
        path=LIGHTCURVE_SUMMARY_PATH,
        family="lightcurve",
    )

    tabular_keep = tabular[
        tabular["model"].isin(
            [
                "dummy_most_frequent",
                "logistic_regression",
            ]
        )
    ].copy()

    lightcurve_keep = lightcurve[
        lightcurve["model"].isin(
            [
                "local_view_cnn",
            ]
        )
    ].copy()

    comparison = pd.concat(
        [
            tabular_keep,
            lightcurve_keep,
        ],
        ignore_index=True,
    )

    comparison["display_model"] = comparison["model"].map(DISPLAY_NAMES)

    if comparison["display_model"].isna().any():
        missing = sorted(
            comparison.loc[
                comparison["display_model"].isna(),
                "model",
            ].unique()
        )
        raise ValueError(f"missing display names for models: {missing}")

    comparison["model_order"] = comparison["display_model"].map(MODEL_ORDER)
    comparison["split_order"] = comparison["split"].map(SPLIT_ORDER)

    if comparison["model_order"].isna().any():
        missing = sorted(
            comparison.loc[
                comparison["model_order"].isna(),
                "display_model",
            ].unique()
        )
        raise ValueError(f"missing model order for models: {missing}")

    if comparison["split_order"].isna().any():
        missing = sorted(
            comparison.loc[
                comparison["split_order"].isna(),
                "split",
            ].unique()
        )
        raise ValueError(f"missing split order for splits: {missing}")

    comparison = comparison.sort_values(
        [
            "split_order",
            "model_order",
        ]
    ).reset_index(drop=True)

    return comparison


def main() -> None:
    comparison = build_comparison()

    COMPARISON_PATH.parent.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(COMPARISON_PATH, index=False)

    test_comparison = comparison[comparison["split"] == "test"]

    print("test-set model comparison:")
    print(
        test_comparison[COMPACT_COLUMNS].to_string(
            index=False,
            float_format=lambda value: f"{value:.3f}",
        )
    )

    print()
    print("wrote:", COMPARISON_PATH)
    print("COMPARE_MODEL_METRICS_OK")


if __name__ == "__main__":
    main()