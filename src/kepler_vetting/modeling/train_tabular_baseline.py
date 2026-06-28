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
from sklearn.model_selection import train_test_split

from kepler_vetting.processing.common import MODEL_READY_NPZ_PATH


RANDOM_SEED = 42

METRICS_DIR = Path("outputs/metrics")
MODEL_DIR = Path("artifacts/models")

METRICS_PATH = METRICS_DIR / "tabular_baseline_metrics.csv"
PREDICTIONS_PATH = METRICS_DIR / "tabular_baseline_predictions.csv"
COEFFICIENTS_PATH = METRICS_DIR / "tabular_logistic_coefficients.csv"
MODEL_PATH = MODEL_DIR / "tabular_logistic_regression.pkl"


def split_indices(labels: np.ndarray) -> dict[str, np.ndarray]:
    all_indices = np.arange(labels.shape[0])

    train_val_idx, test_idx = train_test_split(
        all_indices,
        test_size=0.20,
        random_state=RANDOM_SEED,
        stratify=labels,
    )

    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=0.25,
        random_state=RANDOM_SEED,
        stratify=labels[train_val_idx],
    )

    return {
        "train": train_idx,
        "val": val_idx,
        "test": test_idx,
    }


def evaluate_classifier(
    model_name: str,
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


def main() -> None:
    if not MODEL_READY_NPZ_PATH.exists():
        raise FileNotFoundError(
            f"missing model-ready dataset: {MODEL_READY_NPZ_PATH}. "
            "Run kepler_vetting.processing.filter_model_ready_dataset first."
        )

    data = np.load(MODEL_READY_NPZ_PATH)

    x = data["tabular_features"].astype(np.float32)
    y = data["labels"].astype(np.int64)
    kepid = data["kepid"]
    kepoi_name = data["kepoi_name"]
    disposition = data["disposition"]
    feature_names = data["feature_names"].astype(str)

    if x.ndim != 2:
        raise ValueError(f"tabular_features must be 2D; got shape {x.shape}")

    if x.shape[0] != y.shape[0]:
        raise ValueError(
            "tabular_features and labels row counts differ: "
            f"{x.shape[0]} vs {y.shape[0]}"
        )

    if set(y.tolist()) - {0, 1}:
        raise ValueError(f"labels must only contain 0/1 values; got {sorted(set(y.tolist()))}")

    splits = split_indices(y)

    print("dataset:", MODEL_READY_NPZ_PATH)
    print("n_rows:", y.shape[0])
    print("n_features:", x.shape[1])
    print("feature_names:", feature_names.tolist())
    print()
    print("split sizes:")
    for split_name, indices in splits.items():
        counts = pd.Series(y[indices]).value_counts().sort_index()
        print(f"{split_name}: n={len(indices)}")
        print(counts.to_string())
        print()

    x_train = x[splits["train"]]
    y_train = y[splits["train"]]

    models = {
        "dummy_most_frequent": DummyClassifier(strategy="most_frequent"),
        "logistic_regression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=RANDOM_SEED,
        ),
    }

    metrics_rows = []
    prediction_frames = []

    for model_name, model in models.items():
        model.fit(x_train, y_train)

        for split_name, indices in splits.items():
            split_x = x[indices]
            split_y = y[indices]

            y_pred = model.predict(split_x)
            y_score = predict_scores(model, split_x)

            metrics_rows.append(
                evaluate_classifier(
                    model_name=model_name,
                    split_name=split_name,
                    y_true=split_y,
                    y_pred=y_pred,
                    y_score=y_score,
                )
            )

            prediction_frames.append(
                make_predictions_frame(
                    model_name=model_name,
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

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    metrics = pd.DataFrame(metrics_rows)
    metrics.to_csv(METRICS_PATH, index=False)

    predictions = pd.concat(prediction_frames, ignore_index=True)
    predictions.to_csv(PREDICTIONS_PATH, index=False)

    logistic_model = models["logistic_regression"]
    coefficients = pd.DataFrame(
        {
            "feature": feature_names,
            "coefficient": logistic_model.coef_[0],
            "abs_coefficient": np.abs(logistic_model.coef_[0]),
        }
    ).sort_values("abs_coefficient", ascending=False)

    coefficients.to_csv(COEFFICIENTS_PATH, index=False)

    with MODEL_PATH.open("wb") as f:
        pickle.dump(
            {
                "model": logistic_model,
                "feature_names": feature_names,
                "random_seed": RANDOM_SEED,
                "source_dataset": str(MODEL_READY_NPZ_PATH),
            },
            f,
        )

    print("metrics:")
    print(metrics.to_string(index=False))
    print()
    print("top logistic coefficients:")
    print(coefficients.head(15).to_string(index=False))
    print()
    print("wrote:", METRICS_PATH)
    print("wrote:", PREDICTIONS_PATH)
    print("wrote:", COEFFICIENTS_PATH)
    print("wrote:", MODEL_PATH)
    print("TRAIN_TABULAR_BASELINE_OK")


if __name__ == "__main__":
    main()