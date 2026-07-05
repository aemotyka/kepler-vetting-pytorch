from __future__ import annotations

from dataclasses import dataclass
import os

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
from kepler_vetting.modeling.train_tabular_baseline import (
    load_unstandardized_tabular_features,
)
from kepler_vetting.processing.common import (
    MODEL_READY_MANIFEST_PATH,
    MODEL_READY_NPZ_PATH,
    RUN_METRICS_DIR,
    RUN_MODEL_DIR,
)


MODEL_NAME = "rescue_stacked_logistic_regression"

METRICS_DIR = RUN_METRICS_DIR
MODEL_DIR = RUN_MODEL_DIR

TABULAR_PREDICTIONS_PATH = METRICS_DIR / "tabular_baseline_predictions.csv"
TABULAR_LOCAL_FEATURES_PREDICTIONS_PATH = (
    METRICS_DIR / "tabular_local_features_predictions.csv"
)
LOCAL_CNN_PREDICTIONS_PATH = METRICS_DIR / "lightcurve_cnn_predictions.csv"
GLOBAL_CNN_PREDICTIONS_PATH = METRICS_DIR / "global_lightcurve_cnn_predictions.csv"
FUSED_LOCAL_PREDICTIONS_PATH = METRICS_DIR / "fused_local_model_predictions.csv"
FUSED_LOCAL_FEATURES_PREDICTIONS_PATH = (
    METRICS_DIR / "fused_local_features_model_predictions.csv"
)
FUSED_RESIDUAL_LOCAL_PREDICTIONS_PATH = (
    METRICS_DIR / "fused_residual_local_model_predictions.csv"
)
FUSED_MULTISCALE_LOCAL_PREDICTIONS_PATH = (
    METRICS_DIR / "fused_multiscale_local_model_predictions.csv"
)
FUSED_TRANSIT_SET_PREDICTIONS_PATH = (
    METRICS_DIR / "fused_transit_set_model_predictions.csv"
)
FUSED_LOCAL_TRANSIT_SET_PREDICTIONS_PATH = (
    METRICS_DIR / "fused_local_transit_set_model_predictions.csv"
)

PER_SEED_METRICS_PATH = METRICS_DIR / "rescue_stacked_model_metrics_by_seed.csv"
SUMMARY_METRICS_PATH = METRICS_DIR / "rescue_stacked_model_metrics_summary.csv"
PREDICTIONS_PATH = METRICS_DIR / "rescue_stacked_model_predictions.csv"
COEFFICIENTS_PATH = METRICS_DIR / "rescue_stacked_model_coefficients.csv"
FEATURE_SUMMARY_PATH = METRICS_DIR / "rescue_stacked_model_feature_summary.csv"
MODEL_PATH = MODEL_DIR / "rescue_stacked_logistic_regression.npz"


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


CONTEXT_NUMERIC_COLUMNS = [
    "n_fits_files",
    "n_raw_points",
    "n_clean_points",
    "clean_fraction",
    "time_span_days",
    "global_missing_bin_fraction_before_interp",
    "local_missing_bin_fraction_before_interp",
    "local_window_half_width",
    "global_min_abs_phase",
    "global_min_flux",
    "local_min_abs_phase",
    "local_min_flux",
]


@dataclass(frozen=True)
class BaseModelSpec:
    display_model: str
    model_name: str
    predictions_path: object


ALL_BASE_MODELS = [
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
    BaseModelSpec(
        display_model="fused_tabular_local_features_cnn",
        model_name="fused_tabular_local_features_cnn",
        predictions_path=FUSED_LOCAL_FEATURES_PREDICTIONS_PATH,
    ),
    BaseModelSpec(
        display_model="fused_tabular_residual_local_cnn",
        model_name="fused_tabular_residual_local_cnn",
        predictions_path=FUSED_RESIDUAL_LOCAL_PREDICTIONS_PATH,
    ),
    BaseModelSpec(
        display_model="fused_tabular_multiscale_local_cnn",
        model_name="fused_tabular_multiscale_local_cnn",
        predictions_path=FUSED_MULTISCALE_LOCAL_PREDICTIONS_PATH,
    ),
    BaseModelSpec(
        display_model="fused_tabular_transit_set_cnn",
        model_name="fused_tabular_transit_set_cnn",
        predictions_path=FUSED_TRANSIT_SET_PREDICTIONS_PATH,
    ),
    BaseModelSpec(
        display_model="fused_tabular_local_transit_set_cnn",
        model_name="fused_tabular_local_transit_set_cnn",
        predictions_path=FUSED_LOCAL_TRANSIT_SET_PREDICTIONS_PATH,
    ),
]


def configured_base_model_set() -> str:
    value = os.environ.get("KEPLER_VETTING_BASE_MODEL_SET", "all").strip().lower()

    if value not in {"all", "lean"}:
        raise ValueError(
            "KEPLER_VETTING_BASE_MODEL_SET must be one of: all, lean; "
            f"got {value!r}"
        )

    return value


def selected_base_models() -> list[BaseModelSpec]:
    base_model_set = configured_base_model_set()

    if base_model_set == "all":
        return ALL_BASE_MODELS

    lean_display_models = {
        "tabular_logistic_regression",
        "tabular_local_features_logistic_regression",
        "local_view_cnn",
        "global_view_cnn",
        "fused_tabular_local_cnn",
        "fused_tabular_transit_set_cnn",
    }

    selected = [
        spec
        for spec in ALL_BASE_MODELS
        if spec.display_model in lean_display_models
    ]

    missing = lean_display_models - {
        spec.display_model
        for spec in selected
    }

    if missing:
        raise ValueError(f"lean base model set is missing models: {sorted(missing)}")

    return selected


BASE_MODEL_SET = configured_base_model_set()
BASE_MODELS = selected_base_models()


def require_file(path: object) -> None:
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
                    f"base prediction metadata mismatch after merging "
                    f"{spec.display_model}: {column}"
                )

            merged = merged.drop(columns=[right_column])

    if merged is None or merged.empty:
        raise ValueError("no base model predictions were loaded")

    return merged


def load_context_features() -> pd.DataFrame:
    require_file(MODEL_READY_NPZ_PATH)
    require_file(MODEL_READY_MANIFEST_PATH)

    data = np.load(MODEL_READY_NPZ_PATH)
    manifest = pd.read_csv(MODEL_READY_MANIFEST_PATH).reset_index(drop=True)

    labels = data["labels"].astype(int)
    kepid = data["kepid"].astype(int)
    kepoi_name = data["kepoi_name"].astype(str)
    disposition = data["disposition"].astype(str)

    n_rows = labels.shape[0]

    if len(manifest) != n_rows:
        raise ValueError(
            "model-ready manifest row count does not match dataset: "
            f"{len(manifest)} vs {n_rows}"
        )

    tabular = load_unstandardized_tabular_features(data)
    feature_names = data["feature_names"].astype(str).tolist()

    context = pd.DataFrame(
        {
            "row_index": np.arange(n_rows, dtype=int),
            "kepid": kepid,
            "kepoi_name": kepoi_name,
            "disposition": disposition,
            "y_true": labels,
        }
    )

    for idx, feature_name in enumerate(feature_names):
        context[f"tabular__{feature_name}"] = tabular[:, idx].astype(float)

    for column in CONTEXT_NUMERIC_COLUMNS:
        if column not in manifest.columns:
            continue

        context[f"context__{column}"] = pd.to_numeric(
            manifest[column],
            errors="coerce",
        ).astype(float)

    if "transit_count" in data.files:
        transit_count = data["transit_count"].astype(float)
        context["context__transit_count"] = transit_count
        context["context__log1p_transit_count"] = np.log1p(transit_count)

    return context


def validate_against_model_ready_dataset(frame: pd.DataFrame) -> None:
    context = load_context_features()

    n_rows = len(context)
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


def merge_context_features(frame: pd.DataFrame) -> pd.DataFrame:
    context = load_context_features()

    merged = frame.merge(
        context,
        on="row_index",
        suffixes=("", "_context"),
        how="inner",
        validate="many_to_one",
    )

    for column in [
        "kepid",
        "kepoi_name",
        "disposition",
        "y_true",
    ]:
        right_column = f"{column}_context"

        if right_column not in merged.columns:
            continue

        if not np.array_equal(
            merged[column].to_numpy(),
            merged[right_column].to_numpy(),
        ):
            raise ValueError(f"context metadata mismatch: {column}")

        merged = merged.drop(columns=[right_column])

    return merged


def add_rescue_features(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()

    score_columns = [
        score_column_name(spec.display_model)
        for spec in BASE_MODELS
    ]

    fused = score_column_name("fused_tabular_local_cnn")
    local = score_column_name("local_view_cnn")
    global_ = score_column_name("global_view_cnn")
    transit = score_column_name("fused_tabular_transit_set_cnn")
    local_transit = score_column_name("fused_tabular_local_transit_set_cnn")
    multiscale = score_column_name("fused_tabular_multiscale_local_cnn")
    residual = score_column_name("fused_tabular_residual_local_cnn")
    local_features = score_column_name("fused_tabular_local_features_cnn")

    rescue_columns = [
        local,
        global_,
        transit,
        local_transit,
        multiscale,
        residual,
        local_features,
    ]

    frame["rescue__max_score"] = frame[rescue_columns].max(axis=1)
    frame["rescue__mean_score"] = frame[rescue_columns].mean(axis=1)
    frame["rescue__min_score"] = frame[rescue_columns].min(axis=1)
    frame["rescue__score_range"] = (
        frame[rescue_columns].max(axis=1)
        - frame[rescue_columns].min(axis=1)
    )

    frame["rescue__local_minus_fused"] = frame[local] - frame[fused]
    frame["rescue__global_minus_fused"] = frame[global_] - frame[fused]
    frame["rescue__transit_minus_fused"] = frame[transit] - frame[fused]
    frame["rescue__local_transit_minus_fused"] = frame[local_transit] - frame[fused]
    frame["rescue__multiscale_minus_fused"] = frame[multiscale] - frame[fused]
    frame["rescue__residual_minus_fused"] = frame[residual] - frame[fused]
    frame["rescue__local_features_minus_fused"] = frame[local_features] - frame[fused]
    frame["rescue__max_minus_fused"] = frame["rescue__max_score"] - frame[fused]
    frame["rescue__mean_minus_fused"] = frame["rescue__mean_score"] - frame[fused]

    frame["rescue__fused_margin_abs"] = np.abs(frame[fused] - 0.5)
    frame["rescue__fused_negative"] = (frame[fused] < 0.5).astype(float)
    frame["rescue__fused_positive"] = (frame[fused] >= 0.5).astype(float)
    frame["rescue__max_rescue_positive"] = (
        frame["rescue__max_score"] >= 0.5
    ).astype(float)
    frame["rescue__fused_negative_max_rescue_positive"] = (
        (frame[fused] < 0.5)
        & (frame["rescue__max_score"] >= 0.5)
    ).astype(float)

    frame["rescue__positive_vote_count"] = (
        frame[score_columns] >= 0.5
    ).sum(axis=1).astype(float)
    frame["rescue__rescue_positive_vote_count"] = (
        frame[rescue_columns] >= 0.5
    ).sum(axis=1).astype(float)

    frame["rescue__local_global_vote_count"] = (
        frame[[local, global_]] >= 0.5
    ).sum(axis=1).astype(float)
    frame["rescue__global_or_local_positive"] = (
        (frame[local] >= 0.5)
        | (frame[global_] >= 0.5)
    ).astype(float)
    frame["rescue__global_and_local_positive"] = (
        (frame[local] >= 0.5)
        & (frame[global_] >= 0.5)
    ).astype(float)

    return frame


def feature_columns(frame: pd.DataFrame) -> list[str]:
    score_features = [
        score_column_name(spec.display_model)
        for spec in BASE_MODELS
    ]

    engineered_features = [
        column
        for column in frame.columns
        if column.startswith("rescue__")
    ]

    context_features = [
        column
        for column in frame.columns
        if column.startswith("context__") or column.startswith("tabular__")
    ]

    return score_features + engineered_features + context_features


def fit_feature_preprocessor(
    train_frame: pd.DataFrame,
    features: list[str],
) -> tuple[pd.Series, pd.Series, StandardScaler]:
    x_train = train_frame[features].apply(pd.to_numeric, errors="coerce")

    medians = x_train.median(axis=0)
    medians = medians.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    x_train = x_train.replace([np.inf, -np.inf], np.nan).fillna(medians)

    scaler = StandardScaler()
    scaler.fit(x_train.to_numpy(dtype=np.float64))

    return medians, pd.Series(features, dtype=str), scaler


def transform_features(
    frame: pd.DataFrame,
    features: list[str],
    medians: pd.Series,
    scaler: StandardScaler,
) -> np.ndarray:
    x = frame[features].apply(pd.to_numeric, errors="coerce")
    x = x.replace([np.inf, -np.inf], np.nan).fillna(medians)

    values = x.to_numpy(dtype=np.float64)

    if not np.isfinite(values).all():
        raise ValueError("rescue stacker features contain non-finite values after fill")

    return scaler.transform(values)


def fit_rescue_model(
    train_frame: pd.DataFrame,
    features: list[str],
) -> tuple[pd.Series, StandardScaler, LogisticRegression]:
    y_train = train_frame["y_true"].to_numpy(dtype=int)

    if set(y_train.tolist()) - {0, 1}:
        raise ValueError(
            f"rescue stacker train labels must be 0/1; got {sorted(set(y_train.tolist()))}"
        )

    if len(set(y_train.tolist())) != 2:
        raise ValueError("rescue stacker validation-training split has a single class")

    medians, _, scaler = fit_feature_preprocessor(
        train_frame=train_frame,
        features=features,
    )
    x_train_scaled = transform_features(
        frame=train_frame,
        features=features,
        medians=medians,
        scaler=scaler,
    )

    model = LogisticRegression(
        C=0.25,
        class_weight="balanced",
        max_iter=5000,
        solver="lbfgs",
    )
    model.fit(x_train_scaled, y_train)

    return medians, scaler, model


def predict_rescue_scores(
    frame: pd.DataFrame,
    features: list[str],
    medians: pd.Series,
    scaler: StandardScaler,
    model: LogisticRegression,
) -> np.ndarray:
    x = transform_features(
        frame=frame,
        features=features,
        medians=medians,
        scaler=scaler,
    )

    scores = model.predict_proba(x)[:, 1]

    if not np.isfinite(scores).all():
        raise ValueError("rescue stacker produced non-finite scores")

    return scores


def main() -> None:
    merged = merge_base_predictions()
    validate_against_model_ready_dataset(merged)

    merged = merge_context_features(merged)
    merged = add_rescue_features(merged)

    features = feature_columns(merged)

    print("model:", MODEL_NAME)
    print("base_model_set:", BASE_MODEL_SET)
    print("base_models:", [spec.display_model for spec in BASE_MODELS])
    print("n_features:", len(features))
    print("train_split_for_rescue_stacker: val")
    print("eval_seeds:", list(EVAL_SEEDS))
    print("split_mode:", SPLIT_MODE)
    print("rows_per_seed:", merged.groupby("seed").size().iloc[0])
    print()

    metrics_rows = []
    prediction_frames = []
    coefficient_rows = []
    feature_summary_rows = []

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

        medians, scaler, model = fit_rescue_model(
            train_frame=stacker_train,
            features=features,
        )

        for feature_name, coefficient in zip(features, model.coef_[0]):
            coefficient_rows.append(
                {
                    "seed": seed,
                    "feature": feature_name,
                    "coefficient": float(coefficient),
                    "abs_coefficient": abs(float(coefficient)),
                }
            )

        coefficient_rows.append(
            {
                "seed": seed,
                "feature": "intercept",
                "coefficient": float(model.intercept_[0]),
                "abs_coefficient": abs(float(model.intercept_[0])),
            }
        )

        for feature_name in features:
            train_values = stacker_train[feature_name].to_numpy(dtype=float)
            feature_summary_rows.append(
                {
                    "seed": seed,
                    "feature": feature_name,
                    "train_mean": float(np.nanmean(train_values)),
                    "train_std": float(np.nanstd(train_values)),
                    "train_median_fill": float(medians[feature_name]),
                }
            )

        seed_scores = predict_rescue_scores(
            frame=seed_frame,
            features=features,
            medians=medians,
            scaler=scaler,
            model=model,
        )

        seed_frame["rescue_stacked_score"] = seed_scores

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
            scores = split_frame["rescue_stacked_score"].to_numpy(dtype=float)
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
                "feature_medians": medians.reindex(features).to_numpy(dtype=float),
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
        raise RuntimeError(f"did not capture final rescue stacker for seed={FINAL_MODEL_SEED}")

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    metrics = pd.DataFrame(metrics_rows)
    metrics_summary = summarize_metrics(metrics)
    predictions = pd.concat(prediction_frames, ignore_index=True)
    coefficients = pd.DataFrame(coefficient_rows)
    feature_summary = pd.DataFrame(feature_summary_rows)

    metrics.to_csv(PER_SEED_METRICS_PATH, index=False)
    metrics_summary.to_csv(SUMMARY_METRICS_PATH, index=False)
    predictions.to_csv(PREDICTIONS_PATH, index=False)
    coefficients.to_csv(COEFFICIENTS_PATH, index=False)
    feature_summary.to_csv(FEATURE_SUMMARY_PATH, index=False)
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
        .assign(abs_mean=lambda frame: frame["mean"].abs())
        .sort_values("abs_mean", ascending=False)
        .head(30)
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
    print("wrote:", FEATURE_SUMMARY_PATH)
    print("wrote:", MODEL_PATH)
    print("TRAIN_RESCUE_STACKED_MODEL_OK")


if __name__ == "__main__":
    main()