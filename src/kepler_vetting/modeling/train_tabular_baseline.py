from __future__ import annotations

from pathlib import Path
import pickle

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from kepler_vetting.modeling.splits import SPLIT_MODE, describe_split, split_indices

from kepler_vetting.processing.common import MODEL_READY_NPZ_PATH


EVAL_SEEDS = tuple(range(10))
FINAL_MODEL_SEED = 0

METRICS_DIR = Path("outputs/metrics")
MODEL_DIR = Path("artifacts/models")

PER_SEED_METRICS_PATH = METRICS_DIR / "tabular_baseline_metrics_by_seed.csv"
SUMMARY_METRICS_PATH = METRICS_DIR / "tabular_baseline_metrics_summary.csv"
PREDICTIONS_PATH = METRICS_DIR / "tabular_baseline_predictions.csv"
COEFFICIENTS_PATH = METRICS_DIR / "tabular_logistic_coefficients_by_seed.csv"
COEFFICIENT_SUMMARY_PATH = METRICS_DIR / "tabular_logistic_coefficients_summary.csv"
MODEL_PATH = MODEL_DIR / "tabular_logistic_regression.pkl"


def evaluate_classifier(
    model_name: str,
    seed: int,
    split_name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: np.ndarray | None,
) -> dict[str, float | int | str]:
    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
    ).ravel()

    record: dict[str, float | int | str] = {
        "model": model_name,
        "seed": seed,
        "split": split_name,
        "n": int(y_true.shape[0]),
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }

    if y_score is not None and len(np.unique(y_true)) == 2:
        record["roc_auc"] = roc_auc_score(y_true, y_score)
    else:
        record["roc_auc"] = np.nan

    return record


def predict_scores(model, x: np.ndarray) -> np.ndarray | None:
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(x)
        if proba.shape[1] == 2:
            return proba[:, 1]

    if hasattr(model, "decision_function"):
        scores = model.decision_function(x)
        return np.asarray(scores, dtype=np.float64)

    return None


def make_predictions_frame(
    model_name: str,
    seed: int,
    split_name: str,
    indices: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: np.ndarray | None,
    kepid: np.ndarray,
    kepoi_name: np.ndarray,
    disposition: np.ndarray,
) -> pd.DataFrame:
    if y_score is None:
        score_values = np.full(y_true.shape[0], np.nan, dtype=np.float64)
    else:
        score_values = y_score

    return pd.DataFrame(
        {
            "model": model_name,
            "seed": seed,
            "split": split_name,
            "row_index": indices,
            "kepid": kepid[indices],
            "kepoi_name": kepoi_name[indices],
            "disposition": disposition[indices],
            "y_true": y_true,
            "y_pred": y_pred,
            "planet_like_score": score_values,
            "correct": y_true == y_pred,
        }
    )


def load_unstandardized_tabular_features(data: np.lib.npyio.NpzFile) -> np.ndarray:
    standardized = data["tabular_features"].astype(np.float64)
    feature_means = data["feature_means"].astype(np.float64)
    feature_stds = data["feature_stds"].astype(np.float64)

    if standardized.ndim != 2:
        raise ValueError(f"tabular_features must be 2D; got shape {standardized.shape}")

    if feature_means.shape[0] != standardized.shape[1]:
        raise ValueError(
            "feature_means length does not match tabular feature count: "
            f"{feature_means.shape[0]} vs {standardized.shape[1]}"
        )

    if feature_stds.shape[0] != standardized.shape[1]:
        raise ValueError(
            "feature_stds length does not match tabular feature count: "
            f"{feature_stds.shape[0]} vs {standardized.shape[1]}"
        )

    if not np.isfinite(standardized).all():
        raise ValueError("tabular_features contains non-finite values")

    if not np.isfinite(feature_means).all():
        raise ValueError("feature_means contains non-finite values")

    if not np.isfinite(feature_stds).all():
        raise ValueError("feature_stds contains non-finite values")

    if np.any(feature_stds == 0):
        raise ValueError("feature_stds contains zero values")

    return standardized * feature_stds + feature_means


def fit_models(x_train: np.ndarray, y_train: np.ndarray, seed: int) -> dict[str, object]:
    return {
        "dummy_most_frequent": DummyClassifier(strategy="most_frequent"),
        "logistic_regression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=seed,
        ),
    }


def summarize_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    metric_columns = [
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

    summary = (
        metrics
        .groupby(["model", "split"])[metric_columns]
        .agg(["mean", "std", "min", "max"])
        .reset_index()
    )

    summary.columns = [
        "_".join(col).rstrip("_")
        for col in summary.columns.to_flat_index()
    ]

    return summary


def summarize_coefficients(coefficients: pd.DataFrame) -> pd.DataFrame:
    summary = (
        coefficients
        .groupby("feature", as_index=False)
        .agg(
            coefficient_mean=("coefficient", "mean"),
            coefficient_std=("coefficient", "std"),
            coefficient_min=("coefficient", "min"),
            coefficient_max=("coefficient", "max"),
            abs_coefficient_mean=("abs_coefficient", "mean"),
        )
        .sort_values("abs_coefficient_mean", ascending=False)
    )

    return summary


def main() -> None:
    if not MODEL_READY_NPZ_PATH.exists():
        raise FileNotFoundError(
            f"missing model-ready dataset: {MODEL_READY_NPZ_PATH}. "
            "Run kepler_vetting.processing.filter_model_ready_dataset first."
        )

    data = np.load(MODEL_READY_NPZ_PATH)

    x_unscaled = load_unstandardized_tabular_features(data)
    y = data["labels"].astype(np.int64)
    kepid = data["kepid"]
    kepoi_name = data["kepoi_name"]
    disposition = data["disposition"]
    feature_names = data["feature_names"].astype(str)

    if x_unscaled.shape[0] != y.shape[0]:
        raise ValueError(
            "tabular_features and labels row counts differ: "
            f"{x_unscaled.shape[0]} vs {y.shape[0]}"
        )

    if set(y.tolist()) - {0, 1}:
        raise ValueError(
            f"labels must only contain 0/1 values; got {sorted(set(y.tolist()))}"
        )

    print("dataset:", MODEL_READY_NPZ_PATH)
    print("n_rows:", y.shape[0])
    print("n_features:", x_unscaled.shape[1])
    print("feature_names:", feature_names.tolist())
    print("eval_seeds:", list(EVAL_SEEDS))
    print("split_mode:", SPLIT_MODE)
    print()

    metrics_rows = []
    prediction_frames = []
    coefficient_frames = []

    final_model_payload = None

    for seed in tqdm(EVAL_SEEDS, desc="tabular baseline seeds"):
        splits = split_indices(
            labels=y,
            groups=kepid,
            seed=seed,
        )

        if seed == FINAL_MODEL_SEED:
            print("split summary:")
            print(
                pd.DataFrame(
                    describe_split(
                        labels=y,
                        groups=kepid,
                        splits=splits,
                    )
                ).to_string(index=False)
            )
            print()

        x_train_unscaled = x_unscaled[splits["train"]]
        y_train = y[splits["train"]]

        scaler = StandardScaler()
        x_train = scaler.fit_transform(x_train_unscaled)

        split_features = {
            "train": x_train,
            "val": scaler.transform(x_unscaled[splits["val"]]),
            "test": scaler.transform(x_unscaled[splits["test"]]),
        }

        models = fit_models(x_train=x_train, y_train=y_train, seed=seed)

        for model_name, model in models.items():
            model.fit(x_train, y_train)

            for split_name, indices in splits.items():
                split_x = split_features[split_name]
                split_y = y[indices]

                y_pred = model.predict(split_x)
                y_score = predict_scores(model, split_x)

                metrics_rows.append(
                    evaluate_classifier(
                        model_name=model_name,
                        seed=seed,
                        split_name=split_name,
                        y_true=split_y,
                        y_pred=y_pred,
                        y_score=y_score,
                    )
                )

                prediction_frames.append(
                    make_predictions_frame(
                        model_name=model_name,
                        seed=seed,
                        split_name=split_name,
                        indices=indices,
                        y_true=split_y,
                        y_pred=y_pred,
                        y_score=y_score,
                        kepid=kepid,
                        kepoi_name=kepoi_name,
                        disposition=disposition,
                    )
                )

            if model_name == "logistic_regression":
                coefficient_frames.append(
                    pd.DataFrame(
                        {
                            "seed": seed,
                            "feature": feature_names,
                            "coefficient": model.coef_[0],
                            "abs_coefficient": np.abs(model.coef_[0]),
                        }
                    )
                )

                if seed == FINAL_MODEL_SEED:
                    final_model_payload = {
                        "model": model,
                        "scaler": scaler,
                        "feature_names": feature_names,
                        "seed": seed,
                        "source_dataset": str(MODEL_READY_NPZ_PATH),
                        "split_mode": SPLIT_MODE,
                        "note": (
                            "Evaluation model for the configured final seed. "
                            "Scaler was fit on that seed's train split only."
                        ),
                    }

    if final_model_payload is None:
        raise RuntimeError(f"did not capture final model for seed={FINAL_MODEL_SEED}")

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    metrics = pd.DataFrame(metrics_rows)
    metrics_summary = summarize_metrics(metrics)

    predictions = pd.concat(prediction_frames, ignore_index=True)

    coefficients = pd.concat(coefficient_frames, ignore_index=True)
    coefficient_summary = summarize_coefficients(coefficients)

    metrics.to_csv(PER_SEED_METRICS_PATH, index=False)
    metrics_summary.to_csv(SUMMARY_METRICS_PATH, index=False)
    predictions.to_csv(PREDICTIONS_PATH, index=False)
    coefficients.to_csv(COEFFICIENTS_PATH, index=False)
    coefficient_summary.to_csv(COEFFICIENT_SUMMARY_PATH, index=False)

    with MODEL_PATH.open("wb") as f:
        pickle.dump(final_model_payload, f)

    print("metrics summary:")
    print(metrics_summary.to_string(index=False))
    print()
    print("top logistic coefficient summary:")
    print(coefficient_summary.head(15).to_string(index=False))
    print()
    print("wrote:", PER_SEED_METRICS_PATH)
    print("wrote:", SUMMARY_METRICS_PATH)
    print("wrote:", PREDICTIONS_PATH)
    print("wrote:", COEFFICIENTS_PATH)
    print("wrote:", COEFFICIENT_SUMMARY_PATH)
    print("wrote:", MODEL_PATH)
    print("TRAIN_TABULAR_BASELINE_OK")


if __name__ == "__main__":
    main()