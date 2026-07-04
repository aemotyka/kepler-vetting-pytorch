from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from kepler_vetting.modeling.lightcurve_common import (
    EVAL_SEEDS,
    FINAL_MODEL_SEED,
    SPLIT_MODE,
    evaluate_predictions,
    make_predictions_frame,
    summarize_metrics,
)
from kepler_vetting.processing.common import (
    MODEL_READY_NPZ_PATH,
    RUN_METRICS_DIR,
    RUN_MODEL_DIR,
)


MODEL_NAME = "selective_rescue_rule_model"

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
RESCUE_STACKED_PREDICTIONS_PATH = (
    METRICS_DIR / "rescue_stacked_model_predictions.csv"
)

PER_SEED_METRICS_PATH = METRICS_DIR / "selective_rescue_rule_model_metrics_by_seed.csv"
SUMMARY_METRICS_PATH = METRICS_DIR / "selective_rescue_rule_model_metrics_summary.csv"
PREDICTIONS_PATH = METRICS_DIR / "selective_rescue_rule_model_predictions.csv"
RULES_PATH = METRICS_DIR / "selective_rescue_rule_model_rules_by_seed.csv"
MODEL_PATH = MODEL_DIR / "selective_rescue_rule_model.npz"


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


RESCUE_MIN_SCORE_GRID = [0.65, 0.70, 0.75, 0.80, 0.85, 0.90]
RESCUE_MIN_VOTES_GRID = [2, 3, 4, 5]
RESCUE_FUSED_FLOOR_GRID = [0.00, 0.10, 0.20, 0.30, 0.40]

VETO_MAX_RESCUE_STACKED_GRID = [0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45]
VETO_MAX_VOTES_GRID = [0, 1, 2, 3, 4]
VETO_FUSED_CEILING_GRID = [0.55, 0.60, 0.65, 0.70, 0.80, 1.01]


@dataclass(frozen=True)
class BaseModelSpec:
    display_model: str
    model_name: str
    predictions_path: object


@dataclass(frozen=True)
class RuleParams:
    rescue_min_score: float
    rescue_min_votes: int
    rescue_fused_floor: float
    veto_max_rescue_stacked_score: float
    veto_max_votes: int
    veto_fused_ceiling: float


NO_OP_PARAMS = RuleParams(
    rescue_min_score=2.0,
    rescue_min_votes=999,
    rescue_fused_floor=1.0,
    veto_max_rescue_stacked_score=-1.0,
    veto_max_votes=-1,
    veto_fused_ceiling=-1.0,
)


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
    BaseModelSpec(
        display_model="rescue_stacked_logistic_regression",
        model_name="rescue_stacked_logistic_regression",
        predictions_path=RESCUE_STACKED_PREDICTIONS_PATH,
    ),
]


def require_file(path: object) -> None:
    if not path.exists():
        raise FileNotFoundError(f"missing required file: {path}")


def score_column_name(display_model: str) -> str:
    return f"score__{display_model}"


FUSED_SCORE_COLUMN = score_column_name("fused_tabular_local_cnn")
RESCUE_STACKED_SCORE_COLUMN = score_column_name("rescue_stacked_logistic_regression")

RESCUE_SCORE_COLUMNS = [
    score_column_name("tabular_local_features_logistic_regression"),
    score_column_name("local_view_cnn"),
    score_column_name("global_view_cnn"),
    score_column_name("fused_tabular_local_features_cnn"),
    score_column_name("fused_tabular_residual_local_cnn"),
    score_column_name("fused_tabular_multiscale_local_cnn"),
    score_column_name("fused_tabular_transit_set_cnn"),
    score_column_name("fused_tabular_local_transit_set_cnn"),
    score_column_name("rescue_stacked_logistic_regression"),
]


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

        ordered = seed_frame.sort_values("row_index")

        if not np.array_equal(
            ordered["y_true"].to_numpy(dtype=int),
            labels[ordered["row_index"].to_numpy(dtype=int)],
        ):
            raise ValueError(f"seed={seed} y_true does not match model-ready labels")

        if not np.array_equal(
            ordered["kepid"].to_numpy(dtype=int),
            kepid[ordered["row_index"].to_numpy(dtype=int)],
        ):
            raise ValueError(f"seed={seed} kepid does not match model-ready dataset")

        if not np.array_equal(
            ordered["kepoi_name"].astype(str).to_numpy(),
            kepoi_name[ordered["row_index"].to_numpy(dtype=int)],
        ):
            raise ValueError(f"seed={seed} kepoi_name does not match model-ready dataset")


def add_rule_features(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()

    missing = [
        column
        for column in [FUSED_SCORE_COLUMN, RESCUE_STACKED_SCORE_COLUMN, *RESCUE_SCORE_COLUMNS]
        if column not in frame.columns
    ]

    if missing:
        raise ValueError(f"missing score columns for selective rescue rule: {missing}")

    frame["rule__rescue_max_score"] = frame[RESCUE_SCORE_COLUMNS].max(axis=1)
    frame["rule__rescue_mean_score"] = frame[RESCUE_SCORE_COLUMNS].mean(axis=1)
    frame["rule__rescue_vote_count"] = (
        frame[RESCUE_SCORE_COLUMNS] >= 0.5
    ).sum(axis=1).astype(float)

    return frame


def apply_rule(
    frame: pd.DataFrame,
    params: RuleParams,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    fused = frame[FUSED_SCORE_COLUMN].to_numpy(dtype=float)
    rescue_max = frame["rule__rescue_max_score"].to_numpy(dtype=float)
    rescue_votes = frame["rule__rescue_vote_count"].to_numpy(dtype=float)
    rescue_stacked = frame[RESCUE_STACKED_SCORE_COLUMN].to_numpy(dtype=float)

    scores = fused.copy()

    rescue_mask = (
        (fused < 0.5)
        & (fused >= params.rescue_fused_floor)
        & (rescue_max >= params.rescue_min_score)
        & (rescue_votes >= params.rescue_min_votes)
    )

    veto_mask = (
        (fused >= 0.5)
        & (fused <= params.veto_fused_ceiling)
        & (rescue_stacked <= params.veto_max_rescue_stacked_score)
        & (rescue_votes <= params.veto_max_votes)
    )

    scores[rescue_mask] = np.maximum(
        0.500001,
        np.maximum(
            fused[rescue_mask],
            rescue_max[rescue_mask],
        ),
    )

    scores[veto_mask] = np.minimum(
        0.499999,
        np.minimum(
            fused[veto_mask],
            rescue_stacked[veto_mask],
        ),
    )

    if not np.isfinite(scores).all():
        raise ValueError("selective rescue rule produced non-finite scores")

    predictions = (scores >= 0.5).astype(int)

    return scores, rescue_mask, veto_mask


def rule_grid() -> list[RuleParams]:
    params = [NO_OP_PARAMS]

    for rescue_min_score in RESCUE_MIN_SCORE_GRID:
        for rescue_min_votes in RESCUE_MIN_VOTES_GRID:
            for rescue_fused_floor in RESCUE_FUSED_FLOOR_GRID:
                for veto_max_rescue_stacked_score in VETO_MAX_RESCUE_STACKED_GRID:
                    for veto_max_votes in VETO_MAX_VOTES_GRID:
                        for veto_fused_ceiling in VETO_FUSED_CEILING_GRID:
                            params.append(
                                RuleParams(
                                    rescue_min_score=rescue_min_score,
                                    rescue_min_votes=rescue_min_votes,
                                    rescue_fused_floor=rescue_fused_floor,
                                    veto_max_rescue_stacked_score=(
                                        veto_max_rescue_stacked_score
                                    ),
                                    veto_max_votes=veto_max_votes,
                                    veto_fused_ceiling=veto_fused_ceiling,
                                )
                            )

    return params


def score_rule_on_frame(
    frame: pd.DataFrame,
    params: RuleParams,
) -> dict[str, float | int]:
    scores, rescue_mask, veto_mask = apply_rule(frame=frame, params=params)

    y_true = frame["y_true"].to_numpy(dtype=int)
    y_pred = (scores >= 0.5).astype(int)

    fused_pred = (frame[FUSED_SCORE_COLUMN].to_numpy(dtype=float) >= 0.5).astype(int)

    changed = y_pred != fused_pred
    right_only = (y_pred == y_true) & (fused_pred != y_true)
    left_only = (y_pred != y_true) & (fused_pred == y_true)

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "changed_count": int(changed.sum()),
        "rescue_count": int(rescue_mask.sum()),
        "veto_count": int(veto_mask.sum()),
        "right_only_correct": int(right_only.sum()),
        "left_only_correct": int(left_only.sum()),
        "net_gain_vs_fused": int(right_only.sum() - left_only.sum()),
    }


def tune_rule(train_frame: pd.DataFrame) -> tuple[RuleParams, dict[str, float | int]]:
    best_params = NO_OP_PARAMS
    best_record = score_rule_on_frame(train_frame, NO_OP_PARAMS)

    best_key = (
        best_record["accuracy"],
        best_record["f1"],
        best_record["net_gain_vs_fused"],
        -best_record["changed_count"],
    )

    for params in rule_grid()[1:]:
        record = score_rule_on_frame(train_frame, params)

        key = (
            record["accuracy"],
            record["f1"],
            record["net_gain_vs_fused"],
            -record["changed_count"],
        )

        if key > best_key:
            best_key = key
            best_params = params
            best_record = record

    return best_params, best_record


def params_to_record(
    seed: int,
    params: RuleParams,
    train_record: dict[str, float | int],
) -> dict[str, float | int | str]:
    record = {
        "seed": seed,
        "rescue_min_score": params.rescue_min_score,
        "rescue_min_votes": params.rescue_min_votes,
        "rescue_fused_floor": params.rescue_fused_floor,
        "veto_max_rescue_stacked_score": params.veto_max_rescue_stacked_score,
        "veto_max_votes": params.veto_max_votes,
        "veto_fused_ceiling": params.veto_fused_ceiling,
    }

    for key, value in train_record.items():
        record[f"val_{key}"] = value

    return record


def main() -> None:
    merged = merge_base_predictions()
    validate_against_model_ready_dataset(merged)
    merged = add_rule_features(merged)

    print("model:", MODEL_NAME)
    print("base_models:", [spec.display_model for spec in BASE_MODELS])
    print("rescue_score_columns:", RESCUE_SCORE_COLUMNS)
    print("train_split_for_rule_tuning: val")
    print("eval_seeds:", list(EVAL_SEEDS))
    print("split_mode:", SPLIT_MODE)
    print("rows_per_seed:", merged.groupby("seed").size().iloc[0])
    print("grid_size_including_no_op:", len(rule_grid()))
    print()

    metrics_rows = []
    prediction_frames = []
    rule_rows = []

    final_payload = None

    for seed in EVAL_SEEDS:
        seed_frame = (
            merged[merged["seed"] == seed]
            .copy()
            .sort_values("row_index")
            .reset_index(drop=True)
        )

        rule_train = seed_frame[seed_frame["split"] == "val"].copy()

        if rule_train.empty:
            raise ValueError(f"seed={seed} has no validation rows for rule tuning")

        best_params, best_train_record = tune_rule(rule_train)

        rule_rows.append(
            params_to_record(
                seed=seed,
                params=best_params,
                train_record=best_train_record,
            )
        )

        full_seed_frame = seed_frame.sort_values("row_index").copy()
        expected_row_index = np.arange(full_seed_frame.shape[0])

        if not np.array_equal(
            full_seed_frame["row_index"].to_numpy(dtype=int),
            expected_row_index,
        ):
            raise ValueError(f"seed={seed} row_index is not contiguous after sorting")

        full_kepid = full_seed_frame["kepid"].to_numpy(dtype=int)
        full_kepoi_name = full_seed_frame["kepoi_name"].astype(str).to_numpy()
        full_disposition = full_seed_frame["disposition"].astype(str).to_numpy()

        for split_name, split_frame in seed_frame.groupby("split"):
            split_frame = split_frame.sort_values("row_index").copy()

            indices = split_frame["row_index"].to_numpy(dtype=int)
            targets = split_frame["y_true"].to_numpy(dtype=int)

            scores, rescue_mask, veto_mask = apply_rule(
                frame=split_frame,
                params=best_params,
            )

            record = evaluate_predictions(
                model_name=MODEL_NAME,
                seed=seed,
                split_name=split_name,
                y_true=targets,
                y_score=scores,
            )
            record["rule_train_split"] = "val"
            record["rescue_override_count"] = int(rescue_mask.sum())
            record["veto_override_count"] = int(veto_mask.sum())
            record["total_override_count"] = int(rescue_mask.sum() + veto_mask.sum())

            metrics_rows.append(record)

            predictions = make_predictions_frame(
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

            predictions["rule_rescue_override"] = rescue_mask
            predictions["rule_veto_override"] = veto_mask
            predictions["rule_rescue_max_score"] = split_frame[
                "rule__rescue_max_score"
            ].to_numpy(dtype=float)
            predictions["rule_rescue_vote_count"] = split_frame[
                "rule__rescue_vote_count"
            ].to_numpy(dtype=float)
            predictions["fused_score_before_rule"] = split_frame[
                FUSED_SCORE_COLUMN
            ].to_numpy(dtype=float)

            prediction_frames.append(predictions)

        if seed == FINAL_MODEL_SEED:
            final_payload = {
                "seed": seed,
                "base_models": np.asarray(
                    [spec.display_model for spec in BASE_MODELS],
                    dtype=str,
                ),
                "rescue_score_columns": np.asarray(RESCUE_SCORE_COLUMNS, dtype=str),
                "rescue_min_score": best_params.rescue_min_score,
                "rescue_min_votes": best_params.rescue_min_votes,
                "rescue_fused_floor": best_params.rescue_fused_floor,
                "veto_max_rescue_stacked_score": (
                    best_params.veto_max_rescue_stacked_score
                ),
                "veto_max_votes": best_params.veto_max_votes,
                "veto_fused_ceiling": best_params.veto_fused_ceiling,
                "rule_train_split": "val",
                "source_dataset": str(MODEL_READY_NPZ_PATH),
                "split_mode": SPLIT_MODE,
            }

    if final_payload is None:
        raise RuntimeError(
            f"did not capture final selective rescue rule for seed={FINAL_MODEL_SEED}"
        )

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    metrics = pd.DataFrame(metrics_rows)
    metrics_summary = summarize_metrics(metrics)
    predictions = pd.concat(prediction_frames, ignore_index=True)
    rules = pd.DataFrame(rule_rows)

    metrics.to_csv(PER_SEED_METRICS_PATH, index=False)
    metrics_summary.to_csv(SUMMARY_METRICS_PATH, index=False)
    predictions.to_csv(PREDICTIONS_PATH, index=False)
    rules.to_csv(RULES_PATH, index=False)
    np.savez_compressed(MODEL_PATH, **final_payload)

    print("metrics summary:")
    print(metrics_summary.to_string(index=False))
    print()
    print("rule summary:")
    print(
        rules
        .describe(include="all")
        .transpose()
        .to_string(
            float_format=lambda value: f"{value:.4f}",
        )
    )
    print()
    print("rules by seed:")
    print(rules.to_string(index=False))
    print()
    print("wrote:", PER_SEED_METRICS_PATH)
    print("wrote:", SUMMARY_METRICS_PATH)
    print("wrote:", PREDICTIONS_PATH)
    print("wrote:", RULES_PATH)
    print("wrote:", MODEL_PATH)
    print("TRAIN_SELECTIVE_RESCUE_RULE_MODEL_OK")


if __name__ == "__main__":
    main()