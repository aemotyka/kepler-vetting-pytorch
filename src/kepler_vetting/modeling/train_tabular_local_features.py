from __future__ import annotations

from pathlib import Path
import pickle

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from kepler_vetting.modeling.local_derived_features import (
    build_local_derived_feature_matrix,
    build_quality_feature_matrix,
    combine_feature_blocks,
)
from kepler_vetting.modeling.splits import SPLIT_MODE, describe_split, split_indices
from kepler_vetting.modeling.train_tabular_baseline import (
    EVAL_SEEDS,
    FINAL_MODEL_SEED,
    evaluate_classifier,
    load_unstandardized_tabular_features,
    make_predictions_frame,
    predict_scores,
    summarize_coefficients,
    summarize_metrics,
)
from kepler_vetting.processing.common import (
    MODEL_READY_MANIFEST_PATH,
    MODEL_READY_NPZ_PATH,
)


MODEL_NAME = "tabular_local_features_logistic_regression"

METRICS_DIR = Path("outputs/metrics")
MODEL_DIR = Path("artifacts/models")

PER_SEED_METRICS_PATH = METRICS_DIR / "tabular_local_features_metrics_by_seed.csv"
SUMMARY_METRICS_PATH = METRICS_DIR / "tabular_local_features_metrics_summary.csv"
PREDICTIONS_PATH = METRICS_DIR / "tabular_local_features_predictions.csv"
COEFFICIENTS_PATH = METRICS_DIR / "tabular_local_features_coefficients_by_seed.csv"
COEFFICIENT_SUMMARY_PATH = METRICS_DIR / "tabular_local_features_coefficients_summary.csv"
MODEL_PATH = MODEL_DIR / "tabular_local_features_logistic_regression.pkl"


def fit_train_medians(x_train: np.ndarray) -> np.ndarray:
    x_train = np.asarray(x_train, dtype=np.float64).copy()
    x_train[~np.isfinite(x_train)] = np.nan

    medians = np.nanmedian(x_train, axis=0)
    medians[~np.isfinite(medians)] = 0.0

    return medians.astype(np.float64)


def apply_median_imputation(
    x: np.ndarray,
    medians: np.ndarray,
) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64).copy()

    if x.ndim != 2:
        raise ValueError(f"feature matrix must be 2D; got shape {x.shape}")

    if x.shape[1] != medians.shape[0]:
        raise ValueError(
            "feature matrix column count does not match medians: "
            f"{x.shape[1]} vs {medians.shape[0]}"
        )

    bad_mask = ~np.isfinite(x)

    if bad_mask.any():
        row_idx, col_idx = np.where(bad_mask)
        x[row_idx, col_idx] = medians[col_idx]

    return x


def build_feature_matrix(
    data: np.lib.npyio.NpzFile,
    manifest: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    tabular_matrix = load_unstandardized_tabular_features(data)
    tabular_names = data["feature_names"].astype(str)

    local_matrix, local_names = build_local_derived_feature_matrix(data)
    quality_matrix, quality_names = build_quality_feature_matrix(manifest)

    return combine_feature_blocks(
        [
            (
                tabular_matrix,
                tabular_names,
            ),
            (
                local_matrix,
                local_names,
            ),
            (
                quality_matrix,
                quality_names,
            ),
        ]
    )


def main() -> None:
    if not MODEL_READY_NPZ_PATH.exists():
        raise FileNotFoundError(
            f"missing model-ready dataset: {MODEL_READY_NPZ_PATH}. "
            "Run kepler_vetting.processing.filter_model_ready_dataset first."
        )

    if not MODEL_READY_MANIFEST_PATH.exists():
        raise FileNotFoundError(
            f"missing model-ready manifest: {MODEL_READY_MANIFEST_PATH}. "
            "Run kepler_vetting.processing.filter_model_ready_dataset first."
        )

    data = np.load(MODEL_READY_NPZ_PATH)
    manifest = pd.read_csv(MODEL_READY_MANIFEST_PATH)

    x_unscaled, feature_names = build_feature_matrix(
        data=data,
        manifest=manifest,
    )

    y = data["labels"].astype(np.int64)
    kepid = data["kepid"]
    kepoi_name = data["kepoi_name"]
    disposition = data["disposition"]

    if x_unscaled.shape[0] != y.shape[0]:
        raise ValueError(
            "feature matrix and labels row counts differ: "
            f"{x_unscaled.shape[0]} vs {y.shape[0]}"
        )

    if manifest.shape[0] != y.shape[0]:
        raise ValueError(
            "model-ready manifest and labels row counts differ: "
            f"{manifest.shape[0]} vs {y.shape[0]}"
        )

    if set(y.tolist()) - {0, 1}:
        raise ValueError(
            f"labels must only contain 0/1 values; got {sorted(set(y.tolist()))}"
        )

    print("dataset:", MODEL_READY_NPZ_PATH)
    print("manifest:", MODEL_READY_MANIFEST_PATH)
    print("model:", MODEL_NAME)
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

    for seed in tqdm(EVAL_SEEDS, desc="tabular local-feature seeds"):
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

        train_medians = fit_train_medians(x_train_unscaled)

        x_train_imputed = apply_median_imputation(
            x=x_train_unscaled,
            medians=train_medians,
        )

        scaler = StandardScaler()
        x_train = scaler.fit_transform(x_train_imputed)

        split_features = {
            "train": x_train,
            "val": scaler.transform(
                apply_median_imputation(
                    x=x_unscaled[splits["val"]],
                    medians=train_medians,
                )
            ),
            "test": scaler.transform(
                apply_median_imputation(
                    x=x_unscaled[splits["test"]],
                    medians=train_medians,
                )
            ),
        }

        model = LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=seed,
        )
        model.fit(x_train, y_train)

        for split_name, indices in splits.items():
            split_x = split_features[split_name]
            split_y = y[indices]

            y_pred = model.predict(split_x)
            y_score = predict_scores(model, split_x)

            metrics_rows.append(
                evaluate_classifier(
                    model_name=MODEL_NAME,
                    seed=seed,
                    split_name=split_name,
                    y_true=split_y,
                    y_pred=y_pred,
                    y_score=y_score,
                )
            )

            prediction_frames.append(
                make_predictions_frame(
                    model_name=MODEL_NAME,
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
                "train_medians": train_medians,
                "feature_names": feature_names,
                "seed": seed,
                "source_dataset": str(MODEL_READY_NPZ_PATH),
                "split_mode": SPLIT_MODE,
                "source_manifest": str(MODEL_READY_MANIFEST_PATH),
                "note": (
                    "Evaluation model for the configured final seed. "
                    "Imputer medians and scaler were fit on that seed's train split only."
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
    print("top coefficient summary:")
    print(coefficient_summary.head(25).to_string(index=False))
    print()
    print("wrote:", PER_SEED_METRICS_PATH)
    print("wrote:", SUMMARY_METRICS_PATH)
    print("wrote:", PREDICTIONS_PATH)
    print("wrote:", COEFFICIENTS_PATH)
    print("wrote:", COEFFICIENT_SUMMARY_PATH)
    print("wrote:", MODEL_PATH)
    print("TRAIN_TABULAR_LOCAL_FEATURES_OK")


if __name__ == "__main__":
    main()