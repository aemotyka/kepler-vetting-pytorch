from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


FIXED_THRESHOLD_VARIANT = "fixed_threshold_0.5"
VAL_TUNED_F1_THRESHOLD_VARIANT = "val_tuned_f1_threshold"


def predict_from_scores(
    y_score: np.ndarray,
    threshold: float,
) -> np.ndarray:
    return (y_score >= threshold).astype(np.int64)


def candidate_thresholds(y_score: np.ndarray) -> np.ndarray:
    finite_scores = np.asarray(y_score, dtype=np.float64)
    finite_scores = finite_scores[np.isfinite(finite_scores)]

    if finite_scores.size == 0:
        raise ValueError("cannot choose threshold from empty/non-finite scores")

    grid = np.linspace(0.0, 1.0, 201)

    thresholds = np.unique(
        np.concatenate(
            [
                grid,
                finite_scores,
                np.array([0.5], dtype=np.float64),
            ]
        )
    )

    return thresholds


def select_best_f1_threshold(
    y_true: np.ndarray,
    y_score: np.ndarray,
) -> float:
    y_true = np.asarray(y_true, dtype=np.int64)
    y_score = np.asarray(y_score, dtype=np.float64)

    if y_true.shape[0] != y_score.shape[0]:
        raise ValueError(
            f"y_true and y_score length mismatch: {y_true.shape[0]} vs {y_score.shape[0]}"
        )

    best_threshold = 0.5
    best_f1 = -1.0
    best_distance_from_default = float("inf")

    for threshold in candidate_thresholds(y_score):
        y_pred = predict_from_scores(
            y_score=y_score,
            threshold=float(threshold),
        )

        score = f1_score(
            y_true,
            y_pred,
            zero_division=0,
        )

        distance_from_default = abs(float(threshold) - 0.5)

        if score > best_f1:
            best_threshold = float(threshold)
            best_f1 = float(score)
            best_distance_from_default = distance_from_default
            continue

        if (
            np.isclose(score, best_f1)
            and distance_from_default < best_distance_from_default
        ):
            best_threshold = float(threshold)
            best_distance_from_default = distance_from_default

    return best_threshold


def evaluate_binary_scores(
    model_name: str,
    metric_variant: str,
    seed: int,
    split_name: str,
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float,
) -> dict[str, float | int | str]:
    y_true = np.asarray(y_true, dtype=np.int64)
    y_score = np.asarray(y_score, dtype=np.float64)

    if y_true.shape[0] != y_score.shape[0]:
        raise ValueError(
            f"y_true and y_score length mismatch: {y_true.shape[0]} vs {y_score.shape[0]}"
        )

    if not np.isfinite(y_score).all():
        raise ValueError("y_score contains non-finite values")

    y_pred = predict_from_scores(
        y_score=y_score,
        threshold=threshold,
    )

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
    ).ravel()

    record: dict[str, float | int | str] = {
        "model": model_name,
        "metric_variant": metric_variant,
        "seed": seed,
        "split": split_name,
        "n": int(y_true.shape[0]),
        "threshold": float(threshold),
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }

    if len(np.unique(y_true)) == 2:
        record["roc_auc"] = roc_auc_score(y_true, y_score)
    else:
        record["roc_auc"] = np.nan

    return record
