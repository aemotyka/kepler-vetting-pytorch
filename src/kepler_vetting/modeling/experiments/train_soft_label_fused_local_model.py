from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch import nn
from tqdm import tqdm

from kepler_vetting.modeling.lightcurve_common import (
    BATCH_SIZE,
    EVAL_SEEDS,
    FINAL_MODEL_SEED,
    LEARNING_RATE,
    MAX_EPOCHS,
    PATIENCE,
    SPLIT_MODE,
    WEIGHT_DECAY,
    describe_split,
    evaluate_predictions,
    get_device,
    make_predictions_frame,
    normalize_from_train,
    set_seed,
    sigmoid_np,
    summarize_metrics,
)
from kepler_vetting.modeling.splits import split_indices
from kepler_vetting.modeling.train_fused_local_model import (
    FusedTabularLocalCNN,
    make_fused_loader,
    predict_fused_model,
    train_one_epoch_fused,
)
from kepler_vetting.modeling.train_tabular_baseline import (
    load_unstandardized_tabular_features,
)
from kepler_vetting.processing.common import (
    MODEL_READY_NPZ_PATH,
    RUN_METRICS_DIR,
    RUN_MODEL_DIR,
)


MODEL_NAME = "soft_label_fused_tabular_local_cnn"
VIEW_NAME = "local_view"

CANDIDATE_SOFT_LABEL = 0.7
CONFIRMED_SOFT_LABEL = 1.0
FALSE_POSITIVE_SOFT_LABEL = 0.0

METRICS_DIR = RUN_METRICS_DIR
MODEL_DIR = RUN_MODEL_DIR

PER_SEED_METRICS_PATH = METRICS_DIR / "soft_label_fused_local_model_metrics_by_seed.csv"
SUMMARY_METRICS_PATH = METRICS_DIR / "soft_label_fused_local_model_metrics_summary.csv"
PREDICTIONS_PATH = METRICS_DIR / "soft_label_fused_local_model_predictions.csv"
TRAINING_HISTORY_PATH = (
    METRICS_DIR / "soft_label_fused_local_model_training_history.csv"
)
MODEL_PATH = MODEL_DIR / "soft_label_fused_tabular_local_cnn.pt"


def build_soft_targets(
    labels: np.ndarray,
    disposition: np.ndarray,
) -> np.ndarray:
    labels = labels.astype(np.int64)
    disposition_str = np.asarray(disposition).astype(str)

    if set(labels.tolist()) - {0, 1}:
        raise ValueError(
            f"labels must only contain 0/1 values; got {sorted(set(labels.tolist()))}"
        )

    soft = np.empty(labels.shape[0], dtype=np.float32)

    false_positive_mask = disposition_str == "FALSE POSITIVE"
    candidate_mask = disposition_str == "CANDIDATE"
    confirmed_mask = disposition_str == "CONFIRMED"

    known_mask = false_positive_mask | candidate_mask | confirmed_mask

    if not known_mask.all():
        unknown = sorted(set(disposition_str[~known_mask].tolist()))
        raise ValueError(f"unexpected disposition values: {unknown}")

    soft[false_positive_mask] = FALSE_POSITIVE_SOFT_LABEL
    soft[candidate_mask] = CANDIDATE_SOFT_LABEL
    soft[confirmed_mask] = CONFIRMED_SOFT_LABEL

    expected_binary = (soft > 0.0).astype(np.int64)

    if not np.array_equal(expected_binary, labels):
        mismatch_count = int((expected_binary != labels).sum())
        raise ValueError(
            "soft-label mapping does not match binary labels; "
            f"mismatch_count={mismatch_count}"
        )

    return soft


def describe_soft_targets(
    labels: np.ndarray,
    disposition: np.ndarray,
    soft_targets: np.ndarray,
) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "binary_label": labels.astype(int),
            "disposition": np.asarray(disposition).astype(str),
            "soft_target": soft_targets.astype(float),
        }
    )

    return (
        frame.groupby(
            [
                "binary_label",
                "disposition",
                "soft_target",
            ],
            as_index=False,
        )
        .size()
        .rename(columns={"size": "n"})
        .sort_values(
            [
                "binary_label",
                "disposition",
                "soft_target",
            ]
        )
        .reset_index(drop=True)
    )


def main() -> None:
    if not MODEL_READY_NPZ_PATH.exists():
        raise FileNotFoundError(
            f"missing model-ready dataset: {MODEL_READY_NPZ_PATH}. "
            "Run kepler_vetting.processing.filter_model_ready_dataset first."
        )

    data = np.load(MODEL_READY_NPZ_PATH)

    x_tabular_unscaled = load_unstandardized_tabular_features(data)
    x_local_raw = data[VIEW_NAME].astype(np.float32)
    y_binary = data["labels"].astype(np.int64)

    kepid = data["kepid"]
    kepoi_name = data["kepoi_name"]
    disposition = data["disposition"]
    feature_names = data["feature_names"].astype(str)

    y_soft = build_soft_targets(
        labels=y_binary,
        disposition=disposition,
    )

    if x_local_raw.ndim != 2:
        raise ValueError(f"{VIEW_NAME} must be 2D; got shape {x_local_raw.shape}")

    if x_tabular_unscaled.ndim != 2:
        raise ValueError(
            f"tabular features must be 2D; got shape {x_tabular_unscaled.shape}"
        )

    if x_local_raw.shape[0] != y_binary.shape[0]:
        raise ValueError(
            f"{VIEW_NAME} and labels row counts differ: "
            f"{x_local_raw.shape[0]} vs {y_binary.shape[0]}"
        )

    if x_tabular_unscaled.shape[0] != y_binary.shape[0]:
        raise ValueError(
            "tabular features and labels row counts differ: "
            f"{x_tabular_unscaled.shape[0]} vs {y_binary.shape[0]}"
        )

    device = get_device()

    print("dataset:", MODEL_READY_NPZ_PATH)
    print("model:", MODEL_NAME)
    print("view:", VIEW_NAME)
    print("n_rows:", y_binary.shape[0])
    print("local_input_length:", x_local_raw.shape[1])
    print("n_tabular_features:", x_tabular_unscaled.shape[1])
    print("feature_names:", feature_names.tolist())
    print(
        "soft_label_map:",
        {
            "FALSE POSITIVE": FALSE_POSITIVE_SOFT_LABEL,
            "CANDIDATE": CANDIDATE_SOFT_LABEL,
            "CONFIRMED": CONFIRMED_SOFT_LABEL,
        },
    )
    print("soft target summary:")
    print(
        describe_soft_targets(
            labels=y_binary,
            disposition=disposition,
            soft_targets=y_soft,
        ).to_string(index=False)
    )
    print("eval_seeds:", list(EVAL_SEEDS))
    print("split_mode:", SPLIT_MODE)
    print("device:", device)
    print()

    metrics_rows = []
    prediction_frames = []
    training_history_rows = []

    final_model_payload = None

    for seed in tqdm(EVAL_SEEDS, desc="soft-label fused local model seeds"):
        set_seed(seed)

        splits = split_indices(
            labels=y_binary,
            groups=kepid,
            seed=seed,
        )

        if seed == FINAL_MODEL_SEED:
            print("split summary:")
            print(
                pd.DataFrame(
                    describe_split(
                        labels=y_binary,
                        groups=kepid,
                        splits=splits,
                    )
                ).to_string(index=False)
            )
            print()

        x_local_normalized, local_train_mean, local_train_std = normalize_from_train(
            x=x_local_raw,
            train_idx=splits["train"],
        )

        x_local_for_torch = x_local_normalized[:, None, :]

        tabular_scaler = StandardScaler()
        x_tabular_train = tabular_scaler.fit_transform(
            x_tabular_unscaled[splits["train"]]
        )

        split_features = {
            "train": {
                "local": x_local_for_torch[splits["train"]],
                "tabular": x_tabular_train.astype(np.float32),
            },
            "val": {
                "local": x_local_for_torch[splits["val"]],
                "tabular": tabular_scaler.transform(
                    x_tabular_unscaled[splits["val"]]
                ).astype(np.float32),
            },
            "test": {
                "local": x_local_for_torch[splits["test"]],
                "tabular": tabular_scaler.transform(
                    x_tabular_unscaled[splits["test"]]
                ).astype(np.float32),
            },
        }

        train_loader = make_fused_loader(
            local_view=split_features["train"]["local"],
            tabular_features=split_features["train"]["tabular"],
            y=y_soft[splits["train"]],
            batch_size=BATCH_SIZE,
            shuffle=True,
        )

        eval_loaders = {
            split_name: make_fused_loader(
                local_view=features["local"],
                tabular_features=features["tabular"],
                y=y_soft[splits[split_name]],
                batch_size=BATCH_SIZE,
                shuffle=False,
            )
            for split_name, features in split_features.items()
        }

        y_soft_train = y_soft[splits["train"]]
        effective_pos = float(y_soft_train.sum())
        effective_neg = float((1.0 - y_soft_train).sum())

        if effective_pos <= 0.0 or effective_neg <= 0.0:
            raise ValueError(f"seed={seed} has invalid soft-label class mass")

        pos_weight = torch.tensor(
            [effective_neg / effective_pos],
            dtype=torch.float32,
            device=device,
        )

        model = FusedTabularLocalCNN(
            local_input_length=x_local_raw.shape[1],
            n_tabular_features=x_tabular_unscaled.shape[1],
        ).to(device)

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY,
        )
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        best_val_loss = float("inf")
        best_epoch = -1
        best_state = None
        epochs_without_improvement = 0

        epoch_iter = tqdm(
            range(1, MAX_EPOCHS + 1),
            desc=f"seed {seed} epochs",
            leave=False,
        )

        for epoch in epoch_iter:
            train_loss = train_one_epoch_fused(
                model=model,
                loader=train_loader,
                optimizer=optimizer,
                criterion=criterion,
                device=device,
            )

            val_loss, val_logits, _ = predict_fused_model(
                model=model,
                loader=eval_loaders["val"],
                criterion=criterion,
                device=device,
            )

            val_scores = sigmoid_np(val_logits)
            val_binary_targets = y_binary[splits["val"]]

            val_auc = evaluate_predictions(
                model_name=MODEL_NAME,
                seed=seed,
                split_name="val",
                y_true=val_binary_targets,
                y_score=val_scores,
            )["roc_auc"]

            training_history_rows.append(
                {
                    "seed": seed,
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "val_binary_roc_auc": val_auc,
                    "effective_train_positive_mass": effective_pos,
                    "effective_train_negative_mass": effective_neg,
                    "pos_weight": float(pos_weight.detach().cpu()[0]),
                }
            )

            epoch_iter.set_postfix(
                {
                    "train_loss": f"{train_loss:.4f}",
                    "val_loss": f"{val_loss:.4f}",
                    "val_auc": f"{val_auc:.3f}",
                }
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                best_state = copy.deepcopy(model.state_dict())
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= PATIENCE:
                break

        if best_state is None:
            raise RuntimeError(f"seed={seed} did not produce a best model")

        model.load_state_dict(best_state)

        for split_name, indices in splits.items():
            split_loss, logits, _ = predict_fused_model(
                model=model,
                loader=eval_loaders[split_name],
                criterion=criterion,
                device=device,
            )

            scores = sigmoid_np(logits)
            targets_int = y_binary[indices].astype(np.int64)

            record = evaluate_predictions(
                model_name=MODEL_NAME,
                seed=seed,
                split_name=split_name,
                y_true=targets_int,
                y_score=scores,
            )
            record["loss"] = split_loss
            record["best_epoch"] = best_epoch
            record["best_val_loss"] = best_val_loss
            record["local_train_mean"] = local_train_mean
            record["local_train_std"] = local_train_std
            record["candidate_soft_label"] = CANDIDATE_SOFT_LABEL
            record["effective_train_positive_mass"] = effective_pos
            record["effective_train_negative_mass"] = effective_neg
            record["pos_weight"] = float(pos_weight.detach().cpu()[0])

            metrics_rows.append(record)

            predictions = make_predictions_frame(
                model_name=MODEL_NAME,
                seed=seed,
                split_name=split_name,
                indices=indices,
                y_true=targets_int,
                y_score=scores,
                kepid=kepid,
                kepoi_name=kepoi_name,
                disposition=disposition,
            )
            predictions["soft_target"] = y_soft[indices]
            predictions["candidate_soft_label"] = CANDIDATE_SOFT_LABEL

            prediction_frames.append(predictions)

        if seed == FINAL_MODEL_SEED:
            final_model_payload = {
                "model_state_dict": copy.deepcopy(model.state_dict()),
                "local_input_length": x_local_raw.shape[1],
                "n_tabular_features": x_tabular_unscaled.shape[1],
                "feature_names": feature_names,
                "local_train_mean": local_train_mean,
                "local_train_std": local_train_std,
                "tabular_scaler": tabular_scaler,
                "seed": seed,
                "best_epoch": best_epoch,
                "best_val_loss": best_val_loss,
                "source_dataset": str(MODEL_READY_NPZ_PATH),
                "split_mode": SPLIT_MODE,
                "view": VIEW_NAME,
                "architecture": "FusedTabularLocalCNN",
                "training_target": "soft_binary",
                "false_positive_soft_label": FALSE_POSITIVE_SOFT_LABEL,
                "candidate_soft_label": CANDIDATE_SOFT_LABEL,
                "confirmed_soft_label": CONFIRMED_SOFT_LABEL,
            }

    if final_model_payload is None:
        raise RuntimeError(f"did not capture final model for seed={FINAL_MODEL_SEED}")

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    metrics = pd.DataFrame(metrics_rows)
    metrics_summary = summarize_metrics(metrics)
    predictions = pd.concat(prediction_frames, ignore_index=True)
    training_history = pd.DataFrame(training_history_rows)

    metrics.to_csv(PER_SEED_METRICS_PATH, index=False)
    metrics_summary.to_csv(SUMMARY_METRICS_PATH, index=False)
    predictions.to_csv(PREDICTIONS_PATH, index=False)
    training_history.to_csv(TRAINING_HISTORY_PATH, index=False)

    torch.save(final_model_payload, MODEL_PATH)

    print("metrics summary:")
    print(metrics_summary.to_string(index=False))
    print()
    print("training summary:")
    print(
        metrics[
            [
                "seed",
                "split",
                "loss",
                "best_epoch",
                "best_val_loss",
                "candidate_soft_label",
                "pos_weight",
            ]
        ].to_string(index=False)
    )
    print()
    print("wrote:", PER_SEED_METRICS_PATH)
    print("wrote:", SUMMARY_METRICS_PATH)
    print("wrote:", PREDICTIONS_PATH)
    print("wrote:", TRAINING_HISTORY_PATH)
    print("wrote:", MODEL_PATH)
    print("TRAIN_SOFT_LABEL_FUSED_LOCAL_MODEL_OK")


if __name__ == "__main__":
    main()
