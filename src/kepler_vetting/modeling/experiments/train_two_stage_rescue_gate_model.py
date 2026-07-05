from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from kepler_vetting.modeling.lightcurve_common import (
    EVAL_SEEDS,
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


MODEL_NAME = "two_stage_rescue_gate"
GATE_TRAIN_SPLIT = "val"
GATE_THRESHOLD = 0.5
C_GRID = (0.05, 0.1, 0.25, 0.5, 1.0, 2.0)

METRICS_DIR = RUN_METRICS_DIR
MODEL_DIR = RUN_MODEL_DIR

TABULAR_LOCAL_FEATURES_PREDICTIONS_PATH = (
    METRICS_DIR / "tabular_local_features_predictions.csv"
)
FUSED_LOCAL_PREDICTIONS_PATH = METRICS_DIR / "fused_local_model_predictions.csv"
SOFT_LABEL_FUSED_LOCAL_PREDICTIONS_PATH = (
    METRICS_DIR / "soft_label_fused_local_model_predictions.csv"
)
CANDIDATE_WEIGHTED_FUSED_LOCAL_PREDICTIONS_PATH = (
    METRICS_DIR / "candidate_weighted_fused_local_model_predictions.csv"
)
THREE_CLASS_FUSED_LOCAL_PREDICTIONS_PATH = (
    METRICS_DIR / "three_class_fused_local_model_predictions.csv"
)
FUSED_TRANSIT_SET_PREDICTIONS_PATH = (
    METRICS_DIR / "fused_transit_set_model_predictions.csv"
)
FUSED_LOCAL_TRANSIT_SET_PREDICTIONS_PATH = (
    METRICS_DIR / "fused_local_transit_set_model_predictions.csv"
)
RESCUE_STACKED_PREDICTIONS_PATH = METRICS_DIR / "rescue_stacked_model_predictions.csv"
SELECTIVE_RESCUE_RULE_PREDICTIONS_PATH = (
    METRICS_DIR / "selective_rescue_rule_model_predictions.csv"
)

PER_SEED_METRICS_PATH = METRICS_DIR / "two_stage_rescue_gate_metrics_by_seed.csv"
SUMMARY_METRICS_PATH = METRICS_DIR / "two_stage_rescue_gate_metrics_summary.csv"
PREDICTIONS_PATH = METRICS_DIR / "two_stage_rescue_gate_predictions.csv"
GATE_SUMMARY_PATH = METRICS_DIR / "two_stage_rescue_gate_summary_by_seed.csv"
GATE_COEFFICIENTS_PATH = METRICS_DIR / "two_stage_rescue_gate_coefficients.csv"
MODEL_PATH = MODEL_DIR / "two_stage_rescue_gate.pkl"

KEY_COLUMNS = [
    "seed",
    "split",
    "row_index",
    "kepid",
    "kepoi_name",
    "disposition",
    "y_true",
]

REQUIRED_PREDICTION_COLUMNS = set(KEY_COLUMNS) | {
    "model",
    "planet_like_score",
}


@dataclass(frozen=True)
class PredictionSpec:
    display_model: str
    model_name: str
    predictions_path: Path


BASE_SPEC = PredictionSpec(
    display_model="fused_tabular_local_cnn",
    model_name="fused_tabular_local_cnn",
    predictions_path=FUSED_LOCAL_PREDICTIONS_PATH,
)

EXPERT_SPECS = [
    PredictionSpec(
        display_model="tabular_local_features_logistic_regression",
        model_name="tabular_local_features_logistic_regression",
        predictions_path=TABULAR_LOCAL_FEATURES_PREDICTIONS_PATH,
    ),
    PredictionSpec(
        display_model="soft_label_fused_tabular_local_cnn",
        model_name="soft_label_fused_tabular_local_cnn",
        predictions_path=SOFT_LABEL_FUSED_LOCAL_PREDICTIONS_PATH,
    ),
    PredictionSpec(
        display_model="candidate_weighted_fused_tabular_local_cnn",
        model_name="candidate_weighted_fused_tabular_local_cnn",
        predictions_path=CANDIDATE_WEIGHTED_FUSED_LOCAL_PREDICTIONS_PATH,
    ),
    PredictionSpec(
        display_model="three_class_fused_tabular_local_cnn",
        model_name="three_class_fused_tabular_local_cnn",
        predictions_path=THREE_CLASS_FUSED_LOCAL_PREDICTIONS_PATH,
    ),
    PredictionSpec(
        display_model="fused_tabular_transit_set_cnn",
        model_name="fused_tabular_transit_set_cnn",
        predictions_path=FUSED_TRANSIT_SET_PREDICTIONS_PATH,
    ),
    PredictionSpec(
        display_model="fused_tabular_local_transit_set_cnn",
        model_name="fused_tabular_local_transit_set_cnn",
        predictions_path=FUSED_LOCAL_TRANSIT_SET_PREDICTIONS_PATH,
    ),
    PredictionSpec(
        display_model="rescue_stacked_logistic_regression",
        model_name="rescue_stacked_logistic_regression",
        predictions_path=RESCUE_STACKED_PREDICTIONS_PATH,
    ),
    PredictionSpec(
        display_model="selective_rescue_rule_model",
        model_name="selective_rescue_rule_model",
        predictions_path=SELECTIVE_RESCUE_RULE_PREDICTIONS_PATH,
    ),
]


def score_column_name(display_model: str) -> str:
    return f"score__{display_model}"


BASE_SCORE_COLUMN = score_column_name(BASE_SPEC.display_model)
EXPERT_SCORE_COLUMNS = [score_column_name(spec.display_model) for spec in EXPERT_SPECS]


def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"missing required file: {path}")


def load_prediction_source(spec: PredictionSpec) -> pd.DataFrame:
    require_file(spec.predictions_path)

    frame = pd.read_csv(spec.predictions_path)

    missing = REQUIRED_PREDICTION_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(
            f"{spec.predictions_path} is missing columns: {sorted(missing)}"
        )

    frame = frame[frame["model"] == spec.model_name].copy()

    if frame.empty:
        raise ValueError(
            f"{spec.predictions_path} has no rows for model={spec.model_name}"
        )

    output = frame[KEY_COLUMNS + ["planet_like_score"]].copy()
    output = output.rename(
        columns={
            "planet_like_score": score_column_name(spec.display_model),
        }
    )

    return output


def load_merged_predictions() -> pd.DataFrame:
    merged = load_prediction_source(BASE_SPEC)

    for spec in EXPERT_SPECS:
        expert = load_prediction_source(spec)
        merged = merged.merge(
            expert,
            on=KEY_COLUMNS,
            how="inner",
            validate="one_to_one",
        )

    expected_rows = load_prediction_source(BASE_SPEC).shape[0]

    if merged.shape[0] != expected_rows:
        raise ValueError(
            "merged prediction row count changed; "
            f"expected={expected_rows}, got={merged.shape[0]}"
        )

    return merged


def add_gate_features(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()

    expert_scores = frame[EXPERT_SCORE_COLUMNS]

    frame["base_pred"] = (frame[BASE_SCORE_COLUMN] >= 0.5).astype(int)
    frame["base_margin_abs"] = np.abs(frame[BASE_SCORE_COLUMN] - 0.5)

    frame["expert_max_score"] = expert_scores.max(axis=1)
    frame["expert_min_score"] = expert_scores.min(axis=1)
    frame["expert_mean_score"] = expert_scores.mean(axis=1)
    frame["expert_score_range"] = frame["expert_max_score"] - frame["expert_min_score"]
    frame["expert_positive_vote_count"] = (
        (expert_scores >= 0.5).sum(axis=1).astype(float)
    )
    frame["expert_negative_vote_count"] = (
        (expert_scores < 0.5).sum(axis=1).astype(float)
    )

    for column in EXPERT_SCORE_COLUMNS:
        suffix = column.removeprefix("score__")
        frame[f"delta__{suffix}_minus_base"] = frame[column] - frame[BASE_SCORE_COLUMN]
        frame[f"abs_delta__{suffix}_minus_base"] = np.abs(
            frame[f"delta__{suffix}_minus_base"]
        )

    rescue_mask = (frame["base_pred"] == 0) & (frame["expert_max_score"] >= 0.5)
    veto_mask = (frame["base_pred"] == 1) & (frame["expert_min_score"] < 0.5)

    frame["proposal_kind"] = "none"
    frame.loc[rescue_mask, "proposal_kind"] = "rescue"
    frame.loc[veto_mask, "proposal_kind"] = "veto"

    frame["proposal_pred"] = frame["base_pred"]
    frame.loc[rescue_mask, "proposal_pred"] = 1
    frame.loc[veto_mask, "proposal_pred"] = 0

    frame["proposal_score"] = frame[BASE_SCORE_COLUMN]
    frame.loc[rescue_mask, "proposal_score"] = frame.loc[
        rescue_mask,
        "expert_max_score",
    ]
    frame.loc[veto_mask, "proposal_score"] = frame.loc[
        veto_mask,
        "expert_min_score",
    ]

    frame["proposal_is_rescue"] = rescue_mask.astype(float)
    frame["proposal_is_veto"] = veto_mask.astype(float)
    frame["proposal_abs_delta_from_base"] = np.abs(
        frame["proposal_score"] - frame[BASE_SCORE_COLUMN]
    )
    frame["proposal_disagrees_with_base"] = frame["proposal_pred"] != frame["base_pred"]

    frame["gate_target"] = (
        frame["proposal_pred"] == frame["y_true"].astype(int)
    ).astype(int)

    return frame


FEATURE_COLUMNS = (
    [
        BASE_SCORE_COLUMN,
        "base_margin_abs",
        "expert_max_score",
        "expert_min_score",
        "expert_mean_score",
        "expert_score_range",
        "expert_positive_vote_count",
        "expert_negative_vote_count",
        "proposal_score",
        "proposal_is_rescue",
        "proposal_is_veto",
        "proposal_abs_delta_from_base",
    ]
    + EXPERT_SCORE_COLUMNS
    + [
        f"delta__{column.removeprefix('score__')}_minus_base"
        for column in EXPERT_SCORE_COLUMNS
    ]
    + [
        f"abs_delta__{column.removeprefix('score__')}_minus_base"
        for column in EXPERT_SCORE_COLUMNS
    ]
)


def fused_accuracy(frame: pd.DataFrame) -> float:
    return float((frame["base_pred"] == frame["y_true"].astype(int)).mean())


def final_predictions_for_model(
    frame: pd.DataFrame,
    model,
) -> pd.DataFrame:
    output = frame.copy()

    output["gate_probability"] = 0.0
    output["gate_accept_override"] = False

    disagreement_mask = output["proposal_disagrees_with_base"]

    if disagreement_mask.any():
        probabilities = model.predict_proba(
            output.loc[disagreement_mask, FEATURE_COLUMNS]
        )[:, 1]
        output.loc[disagreement_mask, "gate_probability"] = probabilities
        output.loc[disagreement_mask, "gate_accept_override"] = (
            probabilities >= GATE_THRESHOLD
        )

    output["planet_like_score"] = output[BASE_SCORE_COLUMN].to_numpy()
    accept_mask = output["gate_accept_override"]

    output.loc[accept_mask, "planet_like_score"] = output.loc[
        accept_mask,
        "proposal_score",
    ]

    output["final_pred"] = (output["planet_like_score"] >= 0.5).astype(int)

    return output


def score_candidate_gate(
    frame: pd.DataFrame,
    model,
) -> dict[str, float | int]:
    scored = final_predictions_for_model(
        frame=frame,
        model=model,
    )

    base_correct = frame["base_pred"] == frame["y_true"].astype(int)
    final_correct = scored["final_pred"] == scored["y_true"].astype(int)

    return {
        "accuracy": float(final_correct.mean()),
        "base_accuracy": float(base_correct.mean()),
        "net_gain_vs_base": int(final_correct.sum() - base_correct.sum()),
        "accepted_override_count": int(scored["gate_accept_override"].sum()),
        "disagreement_count": int(frame["proposal_disagrees_with_base"].sum()),
    }


def fit_gate_for_seed(
    frame: pd.DataFrame,
    seed: int,
):
    val = frame[
        (frame["seed"] == seed)
        & (frame["split"] == GATE_TRAIN_SPLIT)
        & (frame["proposal_disagrees_with_base"])
    ].copy()

    if val.empty:
        raise ValueError(f"seed={seed} has no val disagreement rows")

    if val["gate_target"].nunique() < 2:
        raise ValueError(
            f"seed={seed} gate target has one class only: "
            f"{sorted(val['gate_target'].unique().tolist())}"
        )

    best_model = None
    best_result = None

    for c_value in C_GRID:
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=c_value,
                class_weight="balanced",
                max_iter=1000,
                random_state=seed,
            ),
        )
        model.fit(
            val[FEATURE_COLUMNS],
            val["gate_target"].astype(int),
        )

        full_val = frame[
            (frame["seed"] == seed) & (frame["split"] == GATE_TRAIN_SPLIT)
        ].copy()

        result = score_candidate_gate(
            frame=full_val,
            model=model,
        )
        result["c_value"] = c_value

        if best_result is None:
            best_model = model
            best_result = result
            continue

        current_key = (
            result["accuracy"],
            result["net_gain_vs_base"],
            -result["accepted_override_count"],
            -c_value,
        )
        best_key = (
            best_result["accuracy"],
            best_result["net_gain_vs_base"],
            -best_result["accepted_override_count"],
            -best_result["c_value"],
        )

        if current_key > best_key:
            best_model = model
            best_result = result

    if best_model is None or best_result is None:
        raise RuntimeError(f"seed={seed} failed to fit a gate")

    return best_model, best_result


def coefficients_for_seed(
    model,
    seed: int,
    c_value: float,
) -> pd.DataFrame:
    logistic = model.named_steps["logisticregression"]

    rows = [
        {
            "seed": seed,
            "c_value": c_value,
            "feature": "intercept",
            "coefficient": float(logistic.intercept_[0]),
        }
    ]

    rows.extend(
        {
            "seed": seed,
            "c_value": c_value,
            "feature": feature,
            "coefficient": float(coefficient),
        }
        for feature, coefficient in zip(
            FEATURE_COLUMNS,
            logistic.coef_[0],
            strict=True,
        )
    )

    return pd.DataFrame(rows)


def main() -> None:
    if not MODEL_READY_NPZ_PATH.exists():
        raise FileNotFoundError(f"missing model-ready dataset: {MODEL_READY_NPZ_PATH}")

    data = np.load(MODEL_READY_NPZ_PATH)

    kepid = data["kepid"]
    kepoi_name = data["kepoi_name"]
    disposition = data["disposition"]

    merged = load_merged_predictions()
    merged = add_gate_features(merged)

    print("dataset:", MODEL_READY_NPZ_PATH)
    print("model:", MODEL_NAME)
    print("split_mode:", SPLIT_MODE)
    print("base_model:", BASE_SPEC.display_model)
    print(
        "expert_models:",
        [spec.display_model for spec in EXPERT_SPECS],
    )
    print("gate_train_split:", GATE_TRAIN_SPLIT)
    print("gate_threshold:", GATE_THRESHOLD)
    print("c_grid:", list(C_GRID))
    print("rows:", merged.shape[0])
    print()

    metrics_rows = []
    prediction_frames = []
    gate_summary_rows = []
    coefficient_frames = []
    models_by_seed = {}

    for seed in EVAL_SEEDS:
        seed_frame = merged[merged["seed"] == seed].copy()
        gate_model, val_result = fit_gate_for_seed(
            frame=merged,
            seed=seed,
        )
        models_by_seed[seed] = gate_model

        c_value = float(val_result["c_value"])

        coefficient_frames.append(
            coefficients_for_seed(
                model=gate_model,
                seed=seed,
                c_value=c_value,
            )
        )

        for split_name in ["train", "val", "test"]:
            split_frame = seed_frame[seed_frame["split"] == split_name].copy()

            scored = final_predictions_for_model(
                frame=split_frame,
                model=gate_model,
            )

            targets_int = scored["y_true"].astype(int).to_numpy()
            scores = scored["planet_like_score"].astype(float).to_numpy()

            record = evaluate_predictions(
                model_name=MODEL_NAME,
                seed=seed,
                split_name=split_name,
                y_true=targets_int,
                y_score=scores,
            )

            base_correct = scored["base_pred"].astype(int) == targets_int
            final_correct = scored["final_pred"].astype(int) == targets_int

            record["base_accuracy"] = float(base_correct.mean())
            record["net_gain_vs_base"] = int(final_correct.sum() - base_correct.sum())
            record["gate_disagreement_count"] = int(
                scored["proposal_disagrees_with_base"].sum()
            )
            record["gate_accepted_override_count"] = int(
                scored["gate_accept_override"].sum()
            )
            record["gate_rescue_count"] = int(
                (
                    scored["gate_accept_override"]
                    & (scored["proposal_kind"] == "rescue")
                ).sum()
            )
            record["gate_veto_count"] = int(
                (
                    scored["gate_accept_override"] & (scored["proposal_kind"] == "veto")
                ).sum()
            )
            record["selected_c"] = c_value

            metrics_rows.append(record)

            predictions = make_predictions_frame(
                model_name=MODEL_NAME,
                seed=seed,
                split_name=split_name,
                indices=scored["row_index"].astype(int).to_numpy(),
                y_true=targets_int,
                y_score=scores,
                kepid=kepid,
                kepoi_name=kepoi_name,
                disposition=disposition,
            )

            predictions["base_score"] = scored[BASE_SCORE_COLUMN].to_numpy()
            predictions["base_pred"] = scored["base_pred"].astype(int).to_numpy()
            predictions["proposal_score"] = scored["proposal_score"].to_numpy()
            predictions["proposal_pred"] = (
                scored["proposal_pred"].astype(int).to_numpy()
            )
            predictions["proposal_kind"] = scored["proposal_kind"].to_numpy()
            predictions["gate_probability"] = scored["gate_probability"].to_numpy()
            predictions["gate_accept_override"] = (
                scored["gate_accept_override"].astype(bool).to_numpy()
            )
            predictions["final_pred"] = scored["final_pred"].astype(int).to_numpy()
            predictions["selected_c"] = c_value

            for column in EXPERT_SCORE_COLUMNS:
                predictions[column] = scored[column].to_numpy()

            prediction_frames.append(predictions)

        test_frame = seed_frame[seed_frame["split"] == "test"].copy()
        test_result = score_candidate_gate(
            frame=test_frame,
            model=gate_model,
        )

        gate_summary_rows.append(
            {
                "seed": seed,
                "selected_c": c_value,
                "val_accuracy": val_result["accuracy"],
                "val_base_accuracy": val_result["base_accuracy"],
                "val_net_gain_vs_base": val_result["net_gain_vs_base"],
                "val_disagreement_count": val_result["disagreement_count"],
                "val_accepted_override_count": val_result["accepted_override_count"],
                "test_accuracy": test_result["accuracy"],
                "test_base_accuracy": test_result["base_accuracy"],
                "test_net_gain_vs_base": test_result["net_gain_vs_base"],
                "test_disagreement_count": test_result["disagreement_count"],
                "test_accepted_override_count": test_result["accepted_override_count"],
            }
        )

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    metrics = pd.DataFrame(metrics_rows)
    metrics_summary = summarize_metrics(metrics)
    predictions = pd.concat(prediction_frames, ignore_index=True)
    gate_summary = pd.DataFrame(gate_summary_rows)
    coefficients = pd.concat(coefficient_frames, ignore_index=True)

    metrics.to_csv(PER_SEED_METRICS_PATH, index=False)
    metrics_summary.to_csv(SUMMARY_METRICS_PATH, index=False)
    predictions.to_csv(PREDICTIONS_PATH, index=False)
    gate_summary.to_csv(GATE_SUMMARY_PATH, index=False)
    coefficients.to_csv(GATE_COEFFICIENTS_PATH, index=False)

    with MODEL_PATH.open("wb") as f:
        pickle.dump(
            {
                "model_name": MODEL_NAME,
                "split_mode": SPLIT_MODE,
                "base_model": BASE_SPEC.display_model,
                "expert_models": [spec.display_model for spec in EXPERT_SPECS],
                "feature_columns": FEATURE_COLUMNS,
                "gate_train_split": GATE_TRAIN_SPLIT,
                "gate_threshold": GATE_THRESHOLD,
                "c_grid": C_GRID,
                "models_by_seed": models_by_seed,
            },
            f,
        )

    print("metrics summary:")
    print(metrics_summary.to_string(index=False))
    print()
    print("gate summary:")
    print(gate_summary.to_string(index=False))
    print()
    print("wrote:", PER_SEED_METRICS_PATH)
    print("wrote:", SUMMARY_METRICS_PATH)
    print("wrote:", PREDICTIONS_PATH)
    print("wrote:", GATE_SUMMARY_PATH)
    print("wrote:", GATE_COEFFICIENTS_PATH)
    print("wrote:", MODEL_PATH)
    print("TRAIN_TWO_STAGE_RESCUE_GATE_MODEL_OK")


if __name__ == "__main__":
    main()
