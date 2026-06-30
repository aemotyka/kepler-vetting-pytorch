from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from kepler_vetting.modeling.lightcurve_common import (
    EVAL_SEEDS,
    FINAL_MODEL_SEED,
    SPLIT_MODE,
    evaluate_predictions,
    make_predictions_frame,
    summarize_metrics,
)
from kepler_vetting.processing.common import MODEL_READY_NPZ_PATH


MODEL_NAME = "stacked_score_logistic_regression"

METRICS_DIR = Path("outputs/metrics")
MODEL_DIR = Path("artifacts/models")

TABULAR_PREDICTIONS_PATH = METRICS_DIR / "tabular_baseline_predictions.csv"
TABULAR_LOCAL_FEATURES_PREDICTIONS_PATH = (
    METRICS_DIR / "tabular_local_features_predictions.csv"
)
LOCAL_CNN_PREDICTIONS_PATH = METRICS_DIR / "lightcurve_cnn_predictions.csv"
GLOBAL_CNN_PREDICTIONS_PATH = METRICS_DIR / "global_lightcurve_cnn_predictions.csv"
FUSED_LOCAL_PREDICTIONS_PATH = METRICS_DIR / "fused_local_model_predictions.csv"

PER_SEED_METRICS_PATH = METRICS_DIR / "stacked_score_model_metrics_by_seed.csv"
SUMMARY_METRICS_PATH = METRICS_DIR / "stacked_score_model_metrics_summary.csv"
PREDICTIONS_PATH = METRICS_DIR / "stacked_score_model_predictions.csv"
COEFFICIENTS_PATH = METRICS_DIR / "stacked_score_model_coefficients.csv"
MODEL_PATH = MODEL_DIR / "stacked_score_logistic_regression.npz"


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


@dataclass(frozen=True)
class BaseModelSpec:
    display_model: str
    model_name: str
    predictions_path: Path


BASE_MODELS = [
    BaseModelSpec(
        display_model="tabular_logistic_regression",
        model_name="logistic_regression",
        predictions_path=TABULAR_PREDICTIONS_PATH,
    ),
    BaseModelSpec(
        display_model="tabular_local_features_logistic_regression",
        model_name="tabular_local_features_logistic_regression",
        predictions_path=TABULAR_LOCAL_FEATURES_PREDICTIONS_PATH,
    ),
    BaseModelSpec(
        display_model="local_view_cnn",
        model_name="local_view_cnn",
        predictions_path=LOCAL_CNN_PREDICTIONS_PATH,
    ),
    BaseModelSpec(
        display_model="global_view_cnn",
        model_name="global_view_cnn",
        predictions_path=GLOBAL_CNN_PREDICTIONS_PATH,
    ),
    BaseModelSpec(
        display_model="fused_tabular_local_cnn",
        model_name="fused_tabular_local_cnn",
        predictions_path=FUSED_LOCAL_PREDICTIONS_PATH,
    ),
]


def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"missing required file: {path}")


def score_column_name(display_model: str) -> str:
    return f"score__{display_model}"


def load_base_predictions(spec: BaseModelSpec) -> pd.DataFrame:
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

    frame["seed"] = frame["seed"].astype(int)
    frame["row_index"] = frame["row_index"].astype(int)
    frame["kepid"] = frame["kepid"].astype(int)
    frame["kepoi_name"] = frame["kepoi_name"].astype(str)
    frame["y_true"] = frame["y_true"].astype(int)
    frame["planet_like_score"] = frame["planet_like_score"].astype(float)

    frame = frame.rename(
        columns={
            "planet_like_score": score_column_name(spec.display_model),
        }
    )

    keep_columns = [
        "seed",
        "split",
        "row_index",
        "kepid",
        "kepoi_name",
        "disposition",
        "y_true",
        score_column_name(spec.display_model),
    ]

    return frame[keep_columns]


def merge_base_predictions() -> pd.DataFrame:
    merged = None

    for spec in BASE_MODELS:
        current = load_base_predictions(spec)

        if merged is None:
            merged = current
            continue

        merged = merged.merge(
            current,
            on=[
                "seed",
                "split",
                "row_index",
            ],
            suffixes=("", "_right"),
            how="inner",
            validate="one_to_one",
        )

        for column in [
            "kepid",
            "kepoi_name",
            "disposition",
            "y_true",
        ]:
            right_column = f"{column}_right"

            if right_column not in merged.columns:
                continue

            if not np.array_equal(
                merged[column].to_numpy(),
                merged[right_column].to_numpy(),
            ):
                raise ValueError(
                    f"base prediction metadata mismatch after merging {spec.display_model}: "
                    f"{column}"
                )

            merged = merged.drop(columns=[right_column])

    if merged is None or merged.empty:
        raise ValueError("no base model predictions were loaded")

    return merged


def validate_against_model_ready_dataset(frame: pd.DataFrame) -> None:
    require_file(MODEL_READY_NPZ_PATH)

    data = np.load(MODEL_READY_NPZ_PATH)

    labels = data["labels"].astype(int)
    kepid = data["kepid"].astype(int)
    kepoi_name = data["kepoi_name"].astype(str)

    n_rows = labels.shape[0]
    expected_row_index = np.arange(n_rows)

    for seed, seed_frame in frame.groupby("seed"):
        if len(seed_frame) != n_rows:
            raise ValueError(
                f"seed={seed} has {len(seed_frame)} rows, expected {n_rows}"
            )

        row_index = seed_frame["row_index"].to_numpy(dtype=int)

        if len(np.unique(row_index)) != n_rows:
            raise ValueError(f"seed={seed} has duplicate or missing row_index values")

        if not np.array_equal(np.sort(row_index), expected_row_index):
            raise ValueError(f"seed={seed} does not cover all model-ready rows")

        if not np.array_equal(
            seed_frame["y_true"].to_numpy(dtype=int),
            labels[row_index],
        ):
            raise ValueError(f"seed={seed} y_true does not match model-ready labels")

        if not np.array_equal(
            seed_frame["kepid"].to_numpy(dtype=int),
            kepid[row_index],
        ):
            raise ValueError(f"seed={seed} kepid does not match model-ready dataset")

        if not np.array_equal(
            seed_frame["kepoi_name"].astype(str).to_numpy(),
            kepoi_name[row_index],
        ):
            raise ValueError(
                f"seed={seed} kepoi_name does not match model-ready dataset"
            )


def feature_columns() -> list[str]:
    return [
        score_column_name(spec.display_model)
        for spec in BASE_MODELS
    ]


def fit_stacker(
    train_frame: pd.DataFrame,
    features: list[str],
) -> tuple[StandardScaler, LogisticRegression]:
    x_train = train_frame[features].to_numpy(dtype=np.float64)
    y_train = train_frame["y_true"].to_numpy(dtype=int)

    if set(y_train.tolist()) - {0, 1}:
        raise ValueError(
            f"stacker train labels must be 0/1; got {sorted(set(y_train.tolist()))}"
        )

    if len(set(y_train.tolist())) != 2:
        raise ValueError("stacker validation-training split has a single class")

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)

    model = LogisticRegression(
        C=1.0,
        class_weight=None,
        max_iter=1000,
        solver="lbfgs",
    )
    model.fit(x_train_scaled, y_train)

    return scaler, model


def predict_stacker_scores(
    scaler: StandardScaler,
    model: LogisticRegression,
    frame: pd.DataFrame,
    features: list[str],
) -> np.ndarray:
    x = frame[features].to_numpy(dtype=np.float64)
    x_scaled = scaler.transform(x)

    scores = model.predict_proba(x_scaled)[:, 1]

    if not np.isfinite(scores).all():
        raise ValueError("stacked model produced non-finite scores")

    return scores


def main() -> None:
    merged = merge_base_predictions()
    validate_against_model_ready_dataset(merged)

    features = feature_columns()

    print("model:", MODEL_NAME)
    print("base_models:", [spec.display_model for spec in BASE_MODELS])
    print("features:", features)
    print("train_split_for_stacker: val")
    print("eval_seeds:", list(EVAL_SEEDS))
    print("split_mode:", SPLIT_MODE)
    print("rows_per_seed:", merged.groupby("seed").size().iloc[0])
    print()

    metrics_rows = []
    prediction_frames = []
    coefficient_rows = []

    final_payload = None

    for seed in EVAL_SEEDS:
        seed_frame = (
            merged[merged["seed"] == seed]
            .copy()
            .sort_values("row_index")
            .reset_index(drop=True)
        )

        stacker_train = seed_frame[seed_frame["split"] == "val"].copy()

        if stacker_train.empty:
            raise ValueError(f"seed={seed} has no validation rows for stacker training")

        scaler, model = fit_stacker(
            train_frame=stacker_train,
            features=features,
        )

        for feature_name, coefficient in zip(features, model.coef_[0]):
            coefficient_rows.append(
                {
                    "seed": seed,
                    "feature": feature_name,
                    "coefficient": float(coefficient),
                }
            )

        coefficient_rows.append(
            {
                "seed": seed,
                "feature": "intercept",
                "coefficient": float(model.intercept_[0]),
            }
        )

        seed_scores = predict_stacker_scores(
            scaler=scaler,
            model=model,
            frame=seed_frame,
            features=features,
        )

        seed_frame["stacked_score"] = seed_scores

        full_seed_frame = seed_frame.sort_values("row_index").copy()
        expected_row_index = np.arange(full_seed_frame.shape[0])

        if not np.array_equal(
            full_seed_frame["row_index"].to_numpy(dtype=int),
            expected_row_index,
        ):
            raise ValueError(
                f"seed={seed} row_index is not contiguous after sorting"
            )

        full_kepid = full_seed_frame["kepid"].to_numpy(dtype=int)
        full_kepoi_name = full_seed_frame["kepoi_name"].astype(str).to_numpy()
        full_disposition = full_seed_frame["disposition"].astype(str).to_numpy()

        for split_name, split_frame in seed_frame.groupby("split"):
            split_frame = split_frame.sort_values("row_index").copy()
            indices = split_frame["row_index"].to_numpy(dtype=int)
            scores = split_frame["stacked_score"].to_numpy(dtype=float)
            targets = split_frame["y_true"].to_numpy(dtype=int)

            record = evaluate_predictions(
                model_name=MODEL_NAME,
                seed=seed,
                split_name=split_name,
                y_true=targets,
                y_score=scores,
            )
            record["stacker_train_split"] = "val"

            metrics_rows.append(record)

            prediction_frames.append(
                make_predictions_frame(
                    model_name=MODEL_NAME,
                    seed=seed,
                    split_name=split_name,
                    indices=indices,
                    y_true=targets,
                    y_score=scores,
                    kepid=full_kepid,
                    kepoi_name=full_kepoi_name,
                    disposition=full_disposition,
                )
            )

        if seed == FINAL_MODEL_SEED:
            final_payload = {
                "seed": seed,
                "features": np.asarray(features, dtype=str),
                "base_models": np.asarray(
                    [spec.display_model for spec in BASE_MODELS],
                    dtype=str,
                ),
                "scaler_mean": scaler.mean_,
                "scaler_scale": scaler.scale_,
                "coef": model.coef_,
                "intercept": model.intercept_,
                "classes": model.classes_,
                "stacker_train_split": "val",
                "source_dataset": str(MODEL_READY_NPZ_PATH),
                "split_mode": SPLIT_MODE,
            }

    if final_payload is None:
        raise RuntimeError(f"did not capture final stacker for seed={FINAL_MODEL_SEED}")

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    metrics = pd.DataFrame(metrics_rows)
    metrics_summary = summarize_metrics(metrics)
    predictions = pd.concat(prediction_frames, ignore_index=True)
    coefficients = pd.DataFrame(coefficient_rows)

    metrics.to_csv(PER_SEED_METRICS_PATH, index=False)
    metrics_summary.to_csv(SUMMARY_METRICS_PATH, index=False)
    predictions.to_csv(PREDICTIONS_PATH, index=False)
    coefficients.to_csv(COEFFICIENTS_PATH, index=False)
    np.savez_compressed(MODEL_PATH, **final_payload)

    print("metrics summary:")
    print(metrics_summary.to_string(index=False))
    print()
    print("coefficient summary:")
    print(
        coefficients
        .groupby("feature")["coefficient"]
        .agg(["mean", "std", "min", "max"])
        .reset_index()
        .sort_values("mean", ascending=False)
        .to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )
    print()
    print("wrote:", PER_SEED_METRICS_PATH)
    print("wrote:", SUMMARY_METRICS_PATH)
    print("wrote:", PREDICTIONS_PATH)
    print("wrote:", COEFFICIENTS_PATH)
    print("wrote:", MODEL_PATH)
    print("TRAIN_STACKED_SCORE_MODEL_OK")


if __name__ == "__main__":
    main()