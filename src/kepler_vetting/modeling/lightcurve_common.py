from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from kepler_vetting.processing.common import MODEL_READY_NPZ_PATH


EVAL_SEEDS = tuple(range(10))
FINAL_MODEL_SEED = 0

BATCH_SIZE = 32
MAX_EPOCHS = 80
PATIENCE = 15
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4


@dataclass(frozen=True)
class LightcurveTrainingPaths:
    per_seed_metrics_path: Path
    summary_metrics_path: Path
    predictions_path: Path
    training_history_path: Path
    model_path: Path


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")

    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def split_indices(labels: np.ndarray, seed: int) -> dict[str, np.ndarray]:
    all_indices = np.arange(labels.shape[0])

    train_val_idx, test_idx = train_test_split(
        all_indices,
        test_size=0.20,
        random_state=seed,
        stratify=labels,
    )

    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=0.25,
        random_state=seed,
        stratify=labels[train_val_idx],
    )

    return {
        "train": train_idx,
        "val": val_idx,
        "test": test_idx,
    }


class PhaseViewCNN(nn.Module):
    def __init__(self, input_length: int) -> None:
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=7, padding=3),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),

            nn.Conv1d(16, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),

            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),

            nn.AdaptiveAvgPool1d(1),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=0.25),
            nn.Linear(64, 1),
        )

        self.input_length = input_length

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        logits = self.classifier(x)
        return logits.squeeze(1)


def make_loader(
    x: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    dataset = TensorDataset(
        torch.tensor(x, dtype=torch.float32),
        torch.tensor(y, dtype=torch.float32),
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=False,
    )


def normalize_from_train(
    x: np.ndarray,
    train_idx: np.ndarray,
) -> tuple[np.ndarray, float, float]:
    train_values = x[train_idx]

    mean = float(np.mean(train_values))
    std = float(np.std(train_values))

    if not np.isfinite(mean):
        raise ValueError("view train mean is not finite")

    if not np.isfinite(std) or std == 0.0:
        raise ValueError("view train std is invalid")

    normalized = (x - mean) / std

    return normalized.astype(np.float32), mean, std


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.train()

    total_loss = 0.0
    total_count = 0

    for batch_x, batch_y in loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        optimizer.zero_grad(set_to_none=True)

        logits = model(batch_x)
        loss = criterion(logits, batch_y)

        loss.backward()
        optimizer.step()

        batch_count = batch_x.shape[0]
        total_loss += float(loss.detach().cpu()) * batch_count
        total_count += batch_count

    return total_loss / max(total_count, 1)


@torch.no_grad()
def predict_model(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, np.ndarray, np.ndarray]:
    model.eval()

    total_loss = 0.0
    total_count = 0

    all_logits = []
    all_targets = []

    for batch_x, batch_y in loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        logits = model(batch_x)
        loss = criterion(logits, batch_y)

        batch_count = batch_x.shape[0]
        total_loss += float(loss.detach().cpu()) * batch_count
        total_count += batch_count

        all_logits.append(logits.detach().cpu().numpy())
        all_targets.append(batch_y.detach().cpu().numpy())

    logits_np = np.concatenate(all_logits)
    targets_np = np.concatenate(all_targets)

    return total_loss / max(total_count, 1), logits_np, targets_np


def sigmoid_np(logits: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-logits))


def evaluate_predictions(
    model_name: str,
    seed: int,
    split_name: str,
    y_true: np.ndarray,
    y_score: np.ndarray,
) -> dict[str, float | int | str]:
    y_pred = (y_score >= 0.5).astype(np.int64)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
    ).ravel()

    record: dict[str, float | int | str] = {
        "model": model_name,
        "seed": seed,
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

    if len(np.unique(y_true)) == 2:
        record["roc_auc"] = roc_auc_score(y_true, y_score)
    else:
        record["roc_auc"] = np.nan

    return record


def evaluate_hard_predictions(
    model_name: str,
    seed: int,
    split_name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: np.ndarray,
) -> dict[str, float | int | str]:
    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
    ).ravel()

    record: dict[str, float | int | str] = {
        "model": model_name,
        "seed": seed,
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

    if len(np.unique(y_true)) == 2:
        record["roc_auc"] = roc_auc_score(y_true, y_score)
    else:
        record["roc_auc"] = np.nan

    return record


def make_predictions_frame(
    model_name: str,
    seed: int,
    split_name: str,
    indices: np.ndarray,
    y_true: np.ndarray,
    y_score: np.ndarray,
    kepid: np.ndarray,
    kepoi_name: np.ndarray,
    disposition: np.ndarray,
) -> pd.DataFrame:
    y_pred = (y_score >= 0.5).astype(np.int64)

    return pd.DataFrame(
        {
            "model": model_name,
            "seed": seed,
            "split": split_name,
            "row_index": indices,
            "kepid": kepid[indices],
            "kepoi_name": kepoi_name[indices],
            "disposition": disposition[indices],
            "y_true": y_true,
            "y_pred": y_pred,
            "planet_like_score": y_score,
            "correct": y_true == y_pred,
        }
    )


def summarize_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    metric_columns = [
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "tn",
        "fp",
        "fn",
        "tp",
    ]

    summary = (
        metrics
        .groupby(["model", "split"])[metric_columns]
        .agg(["mean", "std", "min", "max"])
        .reset_index()
    )

    summary.columns = [
        "_".join(col).rstrip("_")
        for col in summary.columns.to_flat_index()
    ]

    return summary


def run_phase_view_cnn_baseline(
    view_name: str,
    model_name: str,
    progress_description: str,
    paths: LightcurveTrainingPaths,
    success_marker: str,
) -> None:
    if not MODEL_READY_NPZ_PATH.exists():
        raise FileNotFoundError(
            f"missing model-ready dataset: {MODEL_READY_NPZ_PATH}. "
            "Run kepler_vetting.processing.filter_model_ready_dataset first."
        )

    data = np.load(MODEL_READY_NPZ_PATH)

    x_raw = data[view_name].astype(np.float32)
    y = data["labels"].astype(np.int64)
    kepid = data["kepid"]
    kepoi_name = data["kepoi_name"]
    disposition = data["disposition"]

    if x_raw.ndim != 2:
        raise ValueError(f"{view_name} must be 2D; got shape {x_raw.shape}")

    if x_raw.shape[0] != y.shape[0]:
        raise ValueError(
            f"{view_name} and labels row counts differ: "
            f"{x_raw.shape[0]} vs {y.shape[0]}"
        )

    if set(y.tolist()) - {0, 1}:
        raise ValueError(
            f"labels must only contain 0/1 values; got {sorted(set(y.tolist()))}"
        )

    device = get_device()

    print("dataset:", MODEL_READY_NPZ_PATH)
    print("view:", view_name)
    print("n_rows:", y.shape[0])
    print("input_length:", x_raw.shape[1])
    print("eval_seeds:", list(EVAL_SEEDS))
    print("device:", device)
    print()

    metrics_rows = []
    prediction_frames = []
    training_history_rows = []

    final_model_payload = None

    for seed in tqdm(EVAL_SEEDS, desc=progress_description):
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

        model = PhaseViewCNN(input_length=x_raw.shape[1]).to(device)
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
            val_auc = evaluate_predictions(
                model_name=model_name,
                seed=seed,
                split_name="val",
                y_true=val_targets.astype(np.int64),
                y_score=val_scores,
            )["roc_auc"]

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
            dummy_score = np.full(
                split_y.shape[0],
                float(dummy_pred[0]),
                dtype=np.float64,
            )

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
                make_predictions_frame(
                    model_name="dummy_most_frequent",
                    seed=seed,
                    split_name=split_name,
                    indices=indices,
                    y_true=split_y,
                    y_score=dummy_score,
                    kepid=kepid,
                    kepoi_name=kepoi_name,
                    disposition=disposition,
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
                model_name=model_name,
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
                    model_name=model_name,
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
                "view": view_name,
                "architecture": "PhaseViewCNN",
            }

    if final_model_payload is None:
        raise RuntimeError(f"did not capture final model for seed={FINAL_MODEL_SEED}")

    paths.per_seed_metrics_path.parent.mkdir(parents=True, exist_ok=True)
    paths.model_path.parent.mkdir(parents=True, exist_ok=True)

    metrics = pd.DataFrame(metrics_rows)
    metrics_summary = summarize_metrics(metrics)
    predictions = pd.concat(prediction_frames, ignore_index=True)
    training_history = pd.DataFrame(training_history_rows)

    metrics.to_csv(paths.per_seed_metrics_path, index=False)
    metrics_summary.to_csv(paths.summary_metrics_path, index=False)
    predictions.to_csv(paths.predictions_path, index=False)
    training_history.to_csv(paths.training_history_path, index=False)

    torch.save(final_model_payload, paths.model_path)

    print("metrics summary:")
    print(metrics_summary.to_string(index=False))
    print()
    print("training summary:")
    print(
        metrics[metrics["model"] == model_name][
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
    print("wrote:", paths.per_seed_metrics_path)
    print("wrote:", paths.summary_metrics_path)
    print("wrote:", paths.predictions_path)
    print("wrote:", paths.training_history_path)
    print("wrote:", paths.model_path)
    print(success_marker)