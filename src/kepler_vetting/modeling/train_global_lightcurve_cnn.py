from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.dummy import DummyClassifier
from torch import nn
from tqdm import tqdm

from kepler_vetting.modeling.train_lightcurve_cnn import (
    BATCH_SIZE,
    EVAL_SEEDS,
    FINAL_MODEL_SEED,
    LEARNING_RATE,
    MAX_EPOCHS,
    PATIENCE,
    WEIGHT_DECAY,
    LocalViewCNN as ViewCNN,
    evaluate_hard_predictions,
    evaluate_predictions,
    get_device,
    make_loader,
    make_predictions_frame,
    normalize_from_train,
    predict_model,
    set_seed,
    sigmoid_np,
    split_indices,
    summarize_metrics,
    train_one_epoch,
)
from kepler_vetting.processing.common import MODEL_READY_NPZ_PATH


VIEW_NAME = "global_view"
MODEL_NAME = "global_view_cnn"

METRICS_DIR = Path("outputs/metrics")
MODEL_DIR = Path("artifacts/models")

PER_SEED_METRICS_PATH = METRICS_DIR / "global_lightcurve_cnn_metrics_by_seed.csv"
SUMMARY_METRICS_PATH = METRICS_DIR / "global_lightcurve_cnn_metrics_summary.csv"
PREDICTIONS_PATH = METRICS_DIR / "global_lightcurve_cnn_predictions.csv"
TRAINING_HISTORY_PATH = METRICS_DIR / "global_lightcurve_cnn_training_history.csv"
MODEL_PATH = MODEL_DIR / "global_view_cnn.pt"


def main() -> None:
    if not MODEL_READY_NPZ_PATH.exists():
        raise FileNotFoundError(
            f"missing model-ready dataset: {MODEL_READY_NPZ_PATH}. "
            "Run kepler_vetting.processing.filter_model_ready_dataset first."
        )

    data = np.load(MODEL_READY_NPZ_PATH)

    x_raw = data[VIEW_NAME].astype(np.float32)
    y = data["labels"].astype(np.int64)
    kepid = data["kepid"]
    kepoi_name = data["kepoi_name"]
    disposition = data["disposition"]

    if x_raw.ndim != 2:
        raise ValueError(f"{VIEW_NAME} must be 2D; got shape {x_raw.shape}")

    if x_raw.shape[0] != y.shape[0]:
        raise ValueError(
            f"{VIEW_NAME} and labels row counts differ: "
            f"{x_raw.shape[0]} vs {y.shape[0]}"
        )

    if set(y.tolist()) - {0, 1}:
        raise ValueError(
            f"labels must only contain 0/1 values; got {sorted(set(y.tolist()))}"
        )

    device = get_device()

    print("dataset:", MODEL_READY_NPZ_PATH)
    print("view:", VIEW_NAME)
    print("n_rows:", y.shape[0])
    print("input_length:", x_raw.shape[1])
    print("eval_seeds:", list(EVAL_SEEDS))
    print("device:", device)
    print()

    metrics_rows = []
    prediction_frames = []
    training_history_rows = []

    final_model_payload = None

    for seed in tqdm(EVAL_SEEDS, desc="global CNN seeds"):
        set_seed(seed)

        splits = split_indices(y, seed=seed)

        x_normalized, train_mean, train_std = normalize_from_train(
            x=x_raw,
            train_idx=splits["train"],
        )

        x_for_torch = x_normalized[:, None, :]

        train_loader = make_loader(
            x=x_for_torch[splits["train"]],
            y=y[splits["train"]],
            batch_size=BATCH_SIZE,
            shuffle=True,
        )

        eval_loaders = {
            split_name: make_loader(
                x=x_for_torch[indices],
                y=y[indices],
                batch_size=BATCH_SIZE,
                shuffle=False,
            )
            for split_name, indices in splits.items()
        }

        y_train = y[splits["train"]]
        n_pos = int((y_train == 1).sum())
        n_neg = int((y_train == 0).sum())

        if n_pos == 0 or n_neg == 0:
            raise ValueError(f"seed={seed} has a single-class train split")

        pos_weight = torch.tensor([n_neg / n_pos], dtype=torch.float32, device=device)

        model = ViewCNN(input_length=x_raw.shape[1]).to(device)
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
            train_loss = train_one_epoch(
                model=model,
                loader=train_loader,
                optimizer=optimizer,
                criterion=criterion,
                device=device,
            )

            val_loss, val_logits, val_targets = predict_model(
                model=model,
                loader=eval_loaders["val"],
                criterion=criterion,
                device=device,
            )

            val_scores = sigmoid_np(val_logits)
            val_auc = (
                evaluate_predictions(
                    model_name=MODEL_NAME,
                    seed=seed,
                    split_name="val",
                    y_true=val_targets.astype(np.int64),
                    y_score=val_scores,
                )["roc_auc"]
            )

            training_history_rows.append(
                {
                    "seed": seed,
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "val_roc_auc": val_auc,
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

        dummy = DummyClassifier(strategy="most_frequent")
        dummy.fit(x_raw[splits["train"]], y[splits["train"]])

        for split_name, indices in splits.items():
            split_y = y[indices]

            dummy_pred = dummy.predict(x_raw[indices])
            dummy_score = np.full(split_y.shape[0], float(dummy_pred[0]), dtype=np.float64)

            metrics_rows.append(
                evaluate_hard_predictions(
                    model_name="dummy_most_frequent",
                    seed=seed,
                    split_name=split_name,
                    y_true=split_y,
                    y_pred=dummy_pred,
                    y_score=dummy_score,
                )
            )

            prediction_frames.append(
                pd.DataFrame(
                    {
                        "model": "dummy_most_frequent",
                        "seed": seed,
                        "split": split_name,
                        "row_index": indices,
                        "kepid": kepid[indices],
                        "kepoi_name": kepoi_name[indices],
                        "disposition": disposition[indices],
                        "y_true": split_y,
                        "y_pred": dummy_pred,
                        "planet_like_score": dummy_score,
                        "correct": split_y == dummy_pred,
                    }
                )
            )

            split_loss, logits, targets = predict_model(
                model=model,
                loader=eval_loaders[split_name],
                criterion=criterion,
                device=device,
            )

            scores = sigmoid_np(logits)
            targets_int = targets.astype(np.int64)

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
            record["train_mean"] = train_mean
            record["train_std"] = train_std

            metrics_rows.append(record)

            prediction_frames.append(
                make_predictions_frame(
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
            )

        if seed == FINAL_MODEL_SEED:
            final_model_payload = {
                "model_state_dict": copy.deepcopy(model.state_dict()),
                "input_length": x_raw.shape[1],
                "train_mean": train_mean,
                "train_std": train_std,
                "seed": seed,
                "best_epoch": best_epoch,
                "best_val_loss": best_val_loss,
                "source_dataset": str(MODEL_READY_NPZ_PATH),
                "view": VIEW_NAME,
                "architecture": "ViewCNN",
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
        metrics[metrics["model"] == MODEL_NAME][
            [
                "seed",
                "split",
                "loss",
                "best_epoch",
                "best_val_loss",
            ]
        ].to_string(index=False)
    )
    print()
    print("wrote:", PER_SEED_METRICS_PATH)
    print("wrote:", SUMMARY_METRICS_PATH)
    print("wrote:", PREDICTIONS_PATH)
    print("wrote:", TRAINING_HISTORY_PATH)
    print("wrote:", MODEL_PATH)
    print("TRAIN_GLOBAL_LIGHTCURVE_CNN_OK")


if __name__ == "__main__":
    main()