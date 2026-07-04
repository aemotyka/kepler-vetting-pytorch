from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from kepler_vetting.modeling.local_derived_features import (
    build_local_derived_feature_matrix,
    build_quality_feature_matrix,
    combine_feature_blocks,
)
from kepler_vetting.modeling.train_tabular_baseline import (
    load_unstandardized_tabular_features,
)
from kepler_vetting.processing.common import (
    MODEL_READY_MANIFEST_PATH,
    MODEL_READY_NPZ_PATH,
)


MODEL_NAME = "fused_tabular_local_cnn"

METRICS_DIR = Path("outputs/metrics")
FIGURES_DIR = Path("outputs/figures/fused_positive_failures")

FUSED_PREDICTIONS_PATH = METRICS_DIR / "fused_local_model_predictions.csv"

POSITIVE_DECISIONS_PATH = METRICS_DIR / "fused_positive_failure_decisions.csv"
RECURRING_FAILURES_PATH = METRICS_DIR / "fused_positive_failure_recurring_rows.csv"
DISPOSITION_SUMMARY_PATH = METRICS_DIR / "fused_positive_failure_disposition_summary.csv"
SCORE_BUCKET_SUMMARY_PATH = METRICS_DIR / "fused_positive_failure_score_bucket_summary.csv"
FEATURE_SUMMARY_PATH = METRICS_DIR / "fused_positive_failure_feature_summary.csv"
FEATURE_DIFFERENCES_PATH = METRICS_DIR / "fused_positive_failure_feature_differences.csv"
OTHER_MODEL_CORRECTIONS_PATH = METRICS_DIR / "fused_positive_failure_other_model_corrections.csv"
PLOT_TARGETS_PATH = METRICS_DIR / "fused_positive_failure_plot_targets.csv"


OTHER_MODEL_SPECS = [
    {
        "display_model": "tabular_logistic_regression",
        "model_name": "logistic_regression",
        "path": METRICS_DIR / "tabular_baseline_predictions.csv",
    },
    {
        "display_model": "tabular_local_features_logistic_regression",
        "model_name": "tabular_local_features_logistic_regression",
        "path": METRICS_DIR / "tabular_local_features_predictions.csv",
    },
    {
        "display_model": "local_view_cnn",
        "model_name": "local_view_cnn",
        "path": METRICS_DIR / "lightcurve_cnn_predictions.csv",
    },
    {
        "display_model": "global_view_cnn",
        "model_name": "global_view_cnn",
        "path": METRICS_DIR / "global_lightcurve_cnn_predictions.csv",
    },
    {
        "display_model": "fused_tabular_local_features_cnn",
        "model_name": "fused_tabular_local_features_cnn",
        "path": METRICS_DIR / "fused_local_features_model_predictions.csv",
    },
    {
        "display_model": "stacked_score_logistic_regression",
        "model_name": "stacked_score_logistic_regression",
        "path": METRICS_DIR / "stacked_score_model_predictions.csv",
    },
    {
        "display_model": "fused_tabular_residual_local_cnn",
        "model_name": "fused_tabular_residual_local_cnn",
        "path": METRICS_DIR / "fused_residual_local_model_predictions.csv",
    },
    {
        "display_model": "fused_tabular_multiscale_local_cnn",
        "model_name": "fused_tabular_multiscale_local_cnn",
        "path": METRICS_DIR / "fused_multiscale_local_model_predictions.csv",
    },
    {
        "display_model": "fused_tabular_transit_set_cnn",
        "model_name": "fused_tabular_transit_set_cnn",
        "path": METRICS_DIR / "fused_transit_set_model_predictions.csv",
    },
    {
        "display_model": "fused_tabular_local_transit_set_cnn",
        "model_name": "fused_tabular_local_transit_set_cnn",
        "path": METRICS_DIR / "fused_local_transit_set_model_predictions.csv",
    },
]

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

SCORE_BUCKETS = [
    -0.001,
    0.10,
    0.20,
    0.30,
    0.40,
    0.50,
    0.60,
    0.70,
    0.80,
    0.90,
    1.001,
]

SCORE_BUCKET_LABELS = [
    "0.00-0.10",
    "0.10-0.20",
    "0.20-0.30",
    "0.30-0.40",
    "0.40-0.50",
    "0.50-0.60",
    "0.60-0.70",
    "0.70-0.80",
    "0.80-0.90",
    "0.90-1.00",
]


def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"missing required file: {path}")


def safe_filename(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value).strip())
    return value.strip("_") or "row"


def prediction_columns(display_model: str) -> dict[str, str]:
    return {
        "score": f"score__{display_model}",
        "pred": f"pred__{display_model}",
        "correct": f"correct__{display_model}",
    }


def load_predictions(
    path: Path,
    model_name: str,
    display_model: str,
    require: bool,
) -> pd.DataFrame | None:
    if not path.exists():
        if require:
            raise FileNotFoundError(f"missing required predictions file: {path}")
        return None

    frame = pd.read_csv(path)

    missing = REQUIRED_PREDICTION_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")

    frame = frame[frame["model"] == model_name].copy()

    if frame.empty:
        if require:
            raise ValueError(f"{path} has no rows for model={model_name}")
        return None

    frame["seed"] = frame["seed"].astype(int)
    frame["row_index"] = frame["row_index"].astype(int)
    frame["kepid"] = frame["kepid"].astype(int)
    frame["kepoi_name"] = frame["kepoi_name"].astype(str)
    frame["y_true"] = frame["y_true"].astype(int)
    frame["planet_like_score"] = frame["planet_like_score"].astype(float)

    if "y_pred" in frame.columns:
        frame["y_pred"] = frame["y_pred"].astype(int)
    else:
        frame["y_pred"] = (frame["planet_like_score"] >= 0.5).astype(int)

    columns = prediction_columns(display_model)

    frame = frame.rename(
        columns={
            "planet_like_score": columns["score"],
            "y_pred": columns["pred"],
        }
    )

    frame[columns["correct"]] = frame[columns["pred"]] == frame["y_true"]

    keep_columns = [
        "seed",
        "split",
        "row_index",
        "kepid",
        "kepoi_name",
        "disposition",
        "y_true",
        columns["score"],
        columns["pred"],
        columns["correct"],
    ]

    return frame[keep_columns]


def validate_prediction_rows_against_npz(
    frame: pd.DataFrame,
    labels: np.ndarray,
    kepid: np.ndarray,
    kepoi_name: np.ndarray,
    display_model: str,
) -> None:
    n_rows = labels.shape[0]
    expected = np.arange(n_rows)

    for seed, seed_frame in frame.groupby("seed"):
        if len(seed_frame) != n_rows:
            raise ValueError(
                f"{display_model} seed={seed} has {len(seed_frame)} rows, expected {n_rows}"
            )

        row_index = seed_frame["row_index"].to_numpy(dtype=int)

        if not np.array_equal(np.sort(row_index), expected):
            raise ValueError(
                f"{display_model} seed={seed} does not cover all model-ready rows"
            )

        if len(np.unique(row_index)) != n_rows:
            raise ValueError(f"{display_model} seed={seed} has duplicate row_index values")

        if not np.array_equal(seed_frame["y_true"].to_numpy(dtype=int), labels[row_index]):
            raise ValueError(f"{display_model} seed={seed} y_true does not match labels")

        if not np.array_equal(seed_frame["kepid"].to_numpy(dtype=int), kepid[row_index]):
            raise ValueError(f"{display_model} seed={seed} kepid mismatch")

        if not np.array_equal(
            seed_frame["kepoi_name"].astype(str).to_numpy(),
            kepoi_name[row_index],
        ):
            raise ValueError(f"{display_model} seed={seed} kepoi_name mismatch")


def load_context(data: np.lib.npyio.NpzFile) -> pd.DataFrame:
    require_file(MODEL_READY_MANIFEST_PATH)

    manifest = pd.read_csv(MODEL_READY_MANIFEST_PATH).reset_index(
        names="row_index"
    )

    n_rows = data["labels"].shape[0]

    if len(manifest) != n_rows:
        raise ValueError(
            "model-ready manifest row count does not match model-ready .npz: "
            f"{len(manifest)} vs {n_rows}"
        )

    tabular_feature_names = data["feature_names"].astype(str).tolist()
    tabular_matrix = load_unstandardized_tabular_features(data)

    tabular = pd.DataFrame(
        tabular_matrix,
        columns=tabular_feature_names,
    )
    tabular.insert(0, "row_index", np.arange(n_rows))

    local_matrix, local_names = build_local_derived_feature_matrix(data)
    quality_matrix, quality_names = build_quality_feature_matrix(manifest)

    aux_matrix, aux_names = combine_feature_blocks(
        [
            (local_matrix, local_names),
            (quality_matrix, quality_names),
        ]
    )

    aux = pd.DataFrame(
        aux_matrix,
        columns=aux_names.astype(str).tolist(),
    )
    aux.insert(0, "row_index", np.arange(n_rows))

    context = manifest.merge(
        tabular,
        on="row_index",
        how="left",
        validate="one_to_one",
    ).merge(
        aux,
        on="row_index",
        how="left",
        validate="one_to_one",
    )

    return context


def merge_other_model_predictions(
    decisions: pd.DataFrame,
    labels: np.ndarray,
    kepid: np.ndarray,
    kepoi_name: np.ndarray,
) -> tuple[pd.DataFrame, list[str]]:
    available_models = []

    for spec in OTHER_MODEL_SPECS:
        display_model = spec["display_model"]
        model_name = spec["model_name"]
        path = spec["path"]

        other = load_predictions(
            path=path,
            model_name=model_name,
            display_model=display_model,
            require=False,
        )

        if other is None:
            print(f"skipping optional model, missing predictions: {path}")
            continue

        validate_prediction_rows_against_npz(
            frame=other,
            labels=labels,
            kepid=kepid,
            kepoi_name=kepoi_name,
            display_model=display_model,
        )

        metadata_columns = [
            "kepid",
            "kepoi_name",
            "disposition",
            "y_true",
        ]

        other = other.drop(columns=metadata_columns)

        decisions = decisions.merge(
            other,
            on=[
                "seed",
                "split",
                "row_index",
            ],
            how="left",
            validate="one_to_one",
        )

        available_models.append(display_model)

    return decisions, available_models


def build_positive_decisions(
    fused: pd.DataFrame,
    context: pd.DataFrame,
    target_split: str,
) -> pd.DataFrame:
    columns = prediction_columns(MODEL_NAME)

    positives = fused[
        (fused["split"] == target_split)
        & (fused["y_true"] == 1)
    ].copy()

    if positives.empty:
        raise ValueError(f"no positive fused predictions found for split={target_split}")

    positives = positives.rename(
        columns={
            columns["score"]: "fused_score",
            columns["pred"]: "fused_pred",
            columns["correct"]: "fused_correct",
        }
    )

    positives["fused_false_negative"] = positives["fused_pred"] == 0
    positives["fused_true_positive"] = positives["fused_pred"] == 1
    positives["fused_error_type"] = np.where(
        positives["fused_false_negative"],
        "false_negative",
        "true_positive",
    )

    positives["score_bucket"] = pd.cut(
        positives["fused_score"],
        bins=SCORE_BUCKETS,
        labels=SCORE_BUCKET_LABELS,
        include_lowest=True,
    ).astype(str)

    positives = positives.merge(
        context,
        on="row_index",
        how="left",
        validate="many_to_one",
        suffixes=("", "_context"),
    )

    if "koi_disposition" in positives.columns:
        positives["display_disposition"] = positives["koi_disposition"].astype(str)
    else:
        positives["display_disposition"] = positives["disposition"].astype(str)

    return positives


def add_other_model_diagnostics(
    decisions: pd.DataFrame,
    available_models: list[str],
) -> pd.DataFrame:
    decisions = decisions.copy()

    pred_columns = [
        prediction_columns(model)["pred"]
        for model in available_models
        if prediction_columns(model)["pred"] in decisions.columns
    ]

    for model in available_models:
        columns = prediction_columns(model)

        if columns["pred"] not in decisions.columns:
            continue

        decisions[f"{model}__corrects_fused_fn"] = (
            decisions["fused_false_negative"]
            & (decisions[columns["pred"]] == 1)
        )

        decisions[f"{model}__also_misses_fused_fn"] = (
            decisions["fused_false_negative"]
            & (decisions[columns["pred"]] == 0)
        )

    if pred_columns:
        decisions["any_other_model_corrects_fused_fn"] = (
            decisions.loc[decisions["fused_false_negative"], pred_columns]
            .eq(1)
            .any(axis=1)
        )
        decisions["all_available_models_miss_positive"] = (
            decisions[["fused_pred", *pred_columns]]
            .eq(0)
            .all(axis=1)
        )
    else:
        decisions["any_other_model_corrects_fused_fn"] = False
        decisions["all_available_models_miss_positive"] = decisions["fused_false_negative"]

    decisions["any_other_model_corrects_fused_fn"] = (
        decisions["any_other_model_corrects_fused_fn"].fillna(False)
    )

    return decisions


def summarize_by_disposition(decisions: pd.DataFrame) -> pd.DataFrame:
    summary = (
        decisions
        .groupby("display_disposition", as_index=False)
        .agg(
            positive_decisions=("row_index", "count"),
            unique_positive_rows=("row_index", "nunique"),
            fused_false_negative_decisions=("fused_false_negative", "sum"),
            fused_true_positive_decisions=("fused_true_positive", "sum"),
            mean_fused_score=("fused_score", "mean"),
            median_fused_score=("fused_score", "median"),
            min_fused_score=("fused_score", "min"),
            max_fused_score=("fused_score", "max"),
            any_other_model_corrections=(
                "any_other_model_corrects_fused_fn",
                "sum",
            ),
            all_available_models_miss_positive=(
                "all_available_models_miss_positive",
                "sum",
            ),
        )
    )

    summary["fused_false_negative_rate"] = (
        summary["fused_false_negative_decisions"]
        / summary["positive_decisions"]
    )

    return summary.sort_values(
        [
            "fused_false_negative_rate",
            "positive_decisions",
        ],
        ascending=[
            False,
            False,
        ],
    ).reset_index(drop=True)


def summarize_score_buckets(decisions: pd.DataFrame) -> pd.DataFrame:
    summary = (
        decisions
        .groupby(
            [
                "display_disposition",
                "score_bucket",
            ],
            as_index=False,
        )
        .agg(
            positive_decisions=("row_index", "count"),
            fused_false_negative_decisions=("fused_false_negative", "sum"),
            mean_fused_score=("fused_score", "mean"),
        )
    )

    summary["fused_false_negative_rate"] = (
        summary["fused_false_negative_decisions"]
        / summary["positive_decisions"].clip(lower=1)
    )

    return summary


def numeric_feature_columns(decisions: pd.DataFrame) -> list[str]:
    blocked = {
        "seed",
        "row_index",
        "kepid",
        "y_true",
        "fused_pred",
        "fused_score",
        "fused_correct",
        "binary_label",
    }

    columns = []

    for column in decisions.columns:
        if column in blocked:
            continue

        if column.startswith("score__"):
            continue

        if column.startswith("pred__"):
            continue

        if column.startswith("correct__"):
            continue

        if column.endswith("__corrects_fused_fn"):
            continue

        if column.endswith("__also_misses_fused_fn"):
            continue

        if pd.api.types.is_numeric_dtype(decisions[column]):
            columns.append(column)

    return columns


def summarize_features(decisions: pd.DataFrame) -> pd.DataFrame:
    features = numeric_feature_columns(decisions)

    rows = []

    for (disposition, error_type), group in decisions.groupby(
        [
            "display_disposition",
            "fused_error_type",
        ]
    ):
        for feature in features:
            values = pd.to_numeric(group[feature], errors="coerce").dropna()

            if values.empty:
                continue

            rows.append(
                {
                    "display_disposition": disposition,
                    "fused_error_type": error_type,
                    "feature": feature,
                    "n": int(values.shape[0]),
                    "mean": float(values.mean()),
                    "std": float(values.std()),
                    "min": float(values.min()),
                    "median": float(values.median()),
                    "max": float(values.max()),
                }
            )

    return pd.DataFrame(rows)


def build_feature_differences(feature_summary: pd.DataFrame) -> pd.DataFrame:
    if feature_summary.empty:
        return pd.DataFrame()

    pivot = (
        feature_summary
        .pivot_table(
            index=[
                "display_disposition",
                "feature",
            ],
            columns="fused_error_type",
            values="mean",
            aggfunc="first",
        )
        .reset_index()
    )

    for column in [
        "false_negative",
        "true_positive",
    ]:
        if column not in pivot.columns:
            pivot[column] = np.nan

    pivot = pivot.rename(
        columns={
            "false_negative": "false_negative_mean",
            "true_positive": "true_positive_mean",
        }
    )

    pivot["mean_diff_false_negative_minus_true_positive"] = (
        pivot["false_negative_mean"] - pivot["true_positive_mean"]
    )
    pivot["abs_mean_diff_false_negative_vs_true_positive"] = (
        pivot["mean_diff_false_negative_minus_true_positive"].abs()
    )

    return pivot.sort_values(
        [
            "display_disposition",
            "abs_mean_diff_false_negative_vs_true_positive",
        ],
        ascending=[
            True,
            False,
        ],
    ).reset_index(drop=True)


def summarize_other_model_corrections(
    decisions: pd.DataFrame,
    available_models: list[str],
) -> pd.DataFrame:
    fused_false_negatives = decisions[decisions["fused_false_negative"]].copy()

    rows = []

    for model in available_models:
        columns = prediction_columns(model)

        if columns["pred"] not in fused_false_negatives.columns:
            continue

        corrected = fused_false_negatives[columns["pred"]] == 1
        also_missed = fused_false_negatives[columns["pred"]] == 0

        rows.append(
            {
                "model": model,
                "fused_false_negative_decisions": int(fused_false_negatives.shape[0]),
                "corrected_decisions": int(corrected.sum()),
                "also_missed_decisions": int(also_missed.sum()),
                "correction_rate": float(corrected.mean())
                if fused_false_negatives.shape[0]
                else np.nan,
                "mean_score_on_fused_false_negatives": float(
                    fused_false_negatives[columns["score"]].mean()
                ),
            }
        )

    return pd.DataFrame(rows).sort_values(
        [
            "corrected_decisions",
            "correction_rate",
        ],
        ascending=[
            False,
            False,
        ],
    ).reset_index(drop=True)


def summarize_recurring_failures(
    decisions: pd.DataFrame,
    available_models: list[str],
) -> pd.DataFrame:
    aggregations = {
        "test_appearances": ("seed", "nunique"),
        "fused_false_negative_count": ("fused_false_negative", "sum"),
        "fused_true_positive_count": ("fused_true_positive", "sum"),
        "mean_fused_score": ("fused_score", "mean"),
        "median_fused_score": ("fused_score", "median"),
        "min_fused_score": ("fused_score", "min"),
        "max_fused_score": ("fused_score", "max"),
        "any_other_model_corrections": ("any_other_model_corrects_fused_fn", "sum"),
        "all_available_models_miss_positive_count": (
            "all_available_models_miss_positive",
            "sum",
        ),
    }

    for model in available_models:
        columns = prediction_columns(model)

        if columns["score"] in decisions.columns:
            aggregations[f"mean_score__{model}"] = (columns["score"], "mean")

        if columns["pred"] in decisions.columns:
            aggregations[f"positive_pred_count__{model}"] = (columns["pred"], "sum")

    recurring = (
        decisions
        .groupby(
            [
                "row_index",
                "kepid",
                "kepoi_name",
                "display_disposition",
            ],
            as_index=False,
        )
        .agg(**aggregations)
    )

    recurring["fused_false_negative_rate"] = (
        recurring["fused_false_negative_count"]
        / recurring["test_appearances"].clip(lower=1)
    )

    return recurring.sort_values(
        [
            "fused_false_negative_count",
            "fused_false_negative_rate",
            "min_fused_score",
            "mean_fused_score",
        ],
        ascending=[
            False,
            False,
            True,
            True,
        ],
    ).reset_index(drop=True)


def phase_for_row(phase: np.ndarray, row_index: int) -> np.ndarray:
    if phase.ndim == 1:
        return phase

    if phase.ndim == 2:
        return phase[row_index]

    raise ValueError(f"phase must be 1D or 2D; got shape {phase.shape}")


def plot_failure_row(
    row: pd.Series,
    data: np.lib.npyio.NpzFile,
    output_path: Path,
) -> None:
    row_index = int(row["row_index"])

    global_phase = phase_for_row(data["global_phase"], row_index)
    global_view = data["global_view"][row_index]
    local_phase = phase_for_row(data["local_phase"], row_index)
    local_view = data["local_view"][row_index]

    title_base = (
        f'{row["kepoi_name"]} / KIC {row["kepid"]} / '
        f'{row["display_disposition"]} / '
        f'FN count={int(row["fused_false_negative_count"])} / '
        f'mean fused score={float(row["mean_fused_score"]):.3f}'
    )

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharey=False)

    axes[0].plot(global_phase, global_view, linewidth=1)
    axes[0].axvline(0.0, linestyle="--", linewidth=1)
    axes[0].axhline(0.0, linestyle=":", linewidth=1)
    axes[0].set_title(f"Global view: {title_base}")
    axes[0].set_xlabel("Orbital phase")
    axes[0].set_ylabel("Relative flux")

    axes[1].plot(local_phase, local_view, linewidth=1)
    axes[1].axvline(0.0, linestyle="--", linewidth=1)
    axes[1].axhline(0.0, linestyle=":", linewidth=1)
    axes[1].set_title(f"Local view: {title_base}")
    axes[1].set_xlabel("Orbital phase")
    axes[1].set_ylabel("Relative flux")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def write_failure_plots(
    recurring: pd.DataFrame,
    data: np.lib.npyio.NpzFile,
    top_n: int,
) -> pd.DataFrame:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    targets = (
        recurring[recurring["fused_false_negative_count"] > 0]
        .head(top_n)
        .copy()
        .reset_index(drop=True)
    )

    plot_rows = []

    for rank, (_, row) in enumerate(targets.iterrows(), start=1):
        name = safe_filename(
            f'{rank:02d}_{row["kepoi_name"]}_{row["kepid"]}_{row["display_disposition"]}'
        )
        output_path = FIGURES_DIR / f"{name}.png"

        plot_failure_row(
            row=row,
            data=data,
            output_path=output_path,
        )

        plot_record = row.to_dict()
        plot_record["rank"] = rank
        plot_record["plot_path"] = str(output_path)
        plot_rows.append(plot_record)

    return pd.DataFrame(plot_rows)


def print_table(
    title: str,
    frame: pd.DataFrame,
    columns: list[str] | None = None,
    top_n: int | None = None,
) -> None:
    print(title)

    if frame.empty:
        print("(no rows)")
        print()
        return

    display = frame.copy()

    if top_n is not None:
        display = display.head(top_n)

    if columns is not None:
        display = display[
            [
                column
                for column in columns
                if column in display.columns
            ]
        ]

    print(
        display.to_string(
            index=False,
            float_format=lambda value: f"{value:.3f}",
        )
    )
    print()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze positive failures for the fused tabular/local CNN."
    )

    parser.add_argument(
        "--split",
        default="test",
        choices=[
            "train",
            "val",
            "test",
        ],
        help="Prediction split to analyze. Default: test.",
    )

    parser.add_argument(
        "--top-n",
        type=int,
        default=20,
        help="Number of recurring failure rows to print and plot.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    require_file(MODEL_READY_NPZ_PATH)
    require_file(FUSED_PREDICTIONS_PATH)

    data = np.load(MODEL_READY_NPZ_PATH)

    labels = data["labels"].astype(int)
    kepid = data["kepid"].astype(int)
    kepoi_name = data["kepoi_name"].astype(str)

    context = load_context(data)

    fused = load_predictions(
        path=FUSED_PREDICTIONS_PATH,
        model_name=MODEL_NAME,
        display_model=MODEL_NAME,
        require=True,
    )
    if fused is None:
        raise RuntimeError("fused predictions unexpectedly failed to load")

    validate_prediction_rows_against_npz(
        frame=fused,
        labels=labels,
        kepid=kepid,
        kepoi_name=kepoi_name,
        display_model=MODEL_NAME,
    )

    decisions = build_positive_decisions(
        fused=fused,
        context=context,
        target_split=args.split,
    )

    decisions, available_models = merge_other_model_predictions(
        decisions=decisions,
        labels=labels,
        kepid=kepid,
        kepoi_name=kepoi_name,
    )

    decisions = add_other_model_diagnostics(
        decisions=decisions,
        available_models=available_models,
    )

    disposition_summary = summarize_by_disposition(decisions)
    score_bucket_summary = summarize_score_buckets(decisions)
    feature_summary = summarize_features(decisions)
    feature_differences = build_feature_differences(feature_summary)
    other_model_corrections = summarize_other_model_corrections(
        decisions=decisions,
        available_models=available_models,
    )
    recurring = summarize_recurring_failures(
        decisions=decisions,
        available_models=available_models,
    )
    plot_targets = write_failure_plots(
        recurring=recurring,
        data=data,
        top_n=args.top_n,
    )

    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    decisions.to_csv(POSITIVE_DECISIONS_PATH, index=False)
    recurring.to_csv(RECURRING_FAILURES_PATH, index=False)
    disposition_summary.to_csv(DISPOSITION_SUMMARY_PATH, index=False)
    score_bucket_summary.to_csv(SCORE_BUCKET_SUMMARY_PATH, index=False)
    feature_summary.to_csv(FEATURE_SUMMARY_PATH, index=False)
    feature_differences.to_csv(FEATURE_DIFFERENCES_PATH, index=False)
    other_model_corrections.to_csv(OTHER_MODEL_CORRECTIONS_PATH, index=False)
    plot_targets.to_csv(PLOT_TARGETS_PATH, index=False)

    total_decisions = int(decisions.shape[0])
    false_negative_decisions = int(decisions["fused_false_negative"].sum())
    unique_positive_rows = int(decisions["row_index"].nunique())
    unique_false_negative_rows = int(
        decisions.loc[decisions["fused_false_negative"], "row_index"].nunique()
    )

    print("model:", MODEL_NAME)
    print("split:", args.split)
    print("positive_decisions:", total_decisions)
    print("unique_positive_rows:", unique_positive_rows)
    print("fused_false_negative_decisions:", false_negative_decisions)
    print("unique_fused_false_negative_rows:", unique_false_negative_rows)
    print(
        "fused_false_negative_rate:",
        f"{false_negative_decisions / max(total_decisions, 1):.3f}",
    )
    print("available_comparison_models:", available_models)
    print()

    print_table(
        title="false negatives by disposition:",
        frame=disposition_summary,
        columns=[
            "display_disposition",
            "positive_decisions",
            "unique_positive_rows",
            "fused_false_negative_decisions",
            "fused_false_negative_rate",
            "mean_fused_score",
            "any_other_model_corrections",
            "all_available_models_miss_positive",
        ],
    )

    print_table(
        title="other models on fused false negatives:",
        frame=other_model_corrections,
        columns=[
            "model",
            "fused_false_negative_decisions",
            "corrected_decisions",
            "also_missed_decisions",
            "correction_rate",
            "mean_score_on_fused_false_negatives",
        ],
    )

    print_table(
        title=f"top {args.top_n} recurring fused positive failures:",
        frame=recurring,
        columns=[
            "row_index",
            "kepid",
            "kepoi_name",
            "display_disposition",
            "test_appearances",
            "fused_false_negative_count",
            "fused_false_negative_rate",
            "mean_fused_score",
            "min_fused_score",
            "max_fused_score",
            "any_other_model_corrections",
            "all_available_models_miss_positive_count",
        ],
        top_n=args.top_n,
    )

    print_table(
        title=f"top {args.top_n} feature differences: false negatives vs true positives",
        frame=feature_differences,
        columns=[
            "display_disposition",
            "feature",
            "false_negative_mean",
            "true_positive_mean",
            "mean_diff_false_negative_minus_true_positive",
            "abs_mean_diff_false_negative_vs_true_positive",
        ],
        top_n=args.top_n,
    )

    print("wrote:", POSITIVE_DECISIONS_PATH)
    print("wrote:", RECURRING_FAILURES_PATH)
    print("wrote:", DISPOSITION_SUMMARY_PATH)
    print("wrote:", SCORE_BUCKET_SUMMARY_PATH)
    print("wrote:", FEATURE_SUMMARY_PATH)
    print("wrote:", FEATURE_DIFFERENCES_PATH)
    print("wrote:", OTHER_MODEL_CORRECTIONS_PATH)
    print("wrote:", PLOT_TARGETS_PATH)
    print("wrote plots under:", FIGURES_DIR)
    print("ANALYZE_FUSED_POSITIVE_FAILURES_OK")


if __name__ == "__main__":
    main()