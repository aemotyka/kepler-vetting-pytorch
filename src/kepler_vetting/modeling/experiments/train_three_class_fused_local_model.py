from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
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
    PhaseViewCNN,
    describe_split,
    evaluate_predictions,
    get_device,
    make_predictions_frame,
    normalize_from_train,
    set_seed,
    summarize_metrics,
)
from kepler_vetting.modeling.splits import split_indices
from kepler_vetting.modeling.train_tabular_baseline import (
    load_unstandardized_tabular_features,
)
from kepler_vetting.processing.common import (
    MODEL_READY_NPZ_PATH,
    RUN_METRICS_DIR,
    RUN_MODEL_DIR,
)


MODEL_NAME = "three_class_fused_tabular_local_cnn"
VIEW_NAME = "local_view"

FALSE_POSITIVE_CLASS = 0
CANDIDATE_CLASS = 1
CONFIRMED_CLASS = 2

CLASS_NAMES = {
    FALSE_POSITIVE_CLASS: "FALSE POSITIVE",
    CANDIDATE_CLASS: "CANDIDATE",
    CONFIRMED_CLASS: "CONFIRMED",
}

METRICS_DIR = RUN_METRICS_DIR
MODEL_DIR = RUN_MODEL_DIR

PER_SEED_METRICS_PATH = METRICS_DIR / "three_class_fused_local_model_metrics_by_seed.csv"
SUMMARY_METRICS_PATH = METRICS_DIR / "three_class_fused_local_model_metrics_summary.csv"
PREDICTIONS_PATH = METRICS_DIR / "three_class_fused_local_model_predictions.csv"
TRAINING_HISTORY_PATH = METRICS_DIR / "three_class_fused_local_model_training_history.csv"
MODEL_PATH = MODEL_DIR / "three_class_fused_tabular_local_cnn.pt"


class ThreeClassFusedTabularLocalCNN(nn.Module):
    def __init__(
        self,
        local_input_length: int,
        n_tabular_features: int,
    ) -> None:
        super().__init__()

        self.local_encoder = PhaseViewCNN(
            input_length=local_input_length,
        ).features

        self.tabular_encoder = nn.Sequential(
            nn.Linear(n_tabular_features, 32),
            nn.ReLU(),
            nn.Dropout(p=0.10),
            nn.Linear(32, 16),
            nn.ReLU(),
        )

        self.classifier = nn.Sequential(
            nn.Dropout(p=0.25),
            nn.Linear(64 + 16, 32),
            nn.ReLU(),
            nn.Dropout(p=0.10),
            nn.Linear(32, 3),
        )

        self.local_input_length = local_input_length
        self.n_tabular_features = n_tabular_features

    def forward(
        self,
        local_view: torch.Tensor,
        tabular_features: torch.Tensor,
    ) -> torch.Tensor:
        local_embedding = self.local_encoder(local_view).flatten(start_dim=1)
        tabular_embedding = self.tabular_encoder(tabular_features)

        fused = torch.cat(
            [
                local_embedding,
                tabular_embedding,
            ],
            dim=1,
        )

        return self.classifier(fused)


def build_three_class_targets(
    labels: np.ndarray,
    disposition: np.ndarray,
) -> np.ndarray:
    labels = labels.astype(np.int64)
    disposition_str = np.asarray(disposition).astype(str)

    if set(labels.tolist()) - {0, 1}:
        raise ValueError(
            f"labels must only contain 0/1 values; got {sorted(set(labels.tolist()))}"
        )

    targets = np.empty(labels.shape[0], dtype=np.int64)

    false_positive_mask = disposition_str == "FALSE POSITIVE"
    candidate_mask = disposition_str == "CANDIDATE"
    confirmed_mask = disposition_str == "CONFIRMED"

    known_mask = false_positive_mask | candidate_mask | confirmed_mask

    if not known_mask.all():
        unknown = sorted(set(disposition_str[~known_mask].tolist()))
        raise ValueError(f"unexpected disposition values: {unknown}")

    targets[false_positive_mask] = FALSE_POSITIVE_CLASS
    targets[candidate_mask] = CANDIDATE_CLASS
    targets[confirmed_mask] = CONFIRMED_CLASS

    expected_binary = (targets != FALSE_POSITIVE_CLASS).astype(np.int64)

    if not np.array_equal(expected_binary, labels):
        mismatch_count = int((expected_binary != labels).sum())
        raise ValueError(
            "three-class mapping does not match binary labels; "
            f"mismatch_count={mismatch_count}"
        )

    return targets


def describe_three_class_targets(
    labels: np.ndarray,
    disposition: np.ndarray,
    three_class_targets: np.ndarray,
) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "binary_label": labels.astype(int),
            "disposition": np.asarray(disposition).astype(str),
            "three_class_target": three_class_targets.astype(int),
        }
    )
    frame["class_name"] = frame["three_class_target"].map(CLASS_NAMES)

    return (
        frame
        .groupby(
            [
                "binary_label",
                "disposition",
                "three_class_target",
                "class_name",
            ],
            as_index=False,
        )
        .size()
        .rename(columns={"size": "n"})
        .sort_values(
            [
                "three_class_target",
                "disposition",
            ]
        )
        .reset_index(drop=True)
    )


def make_three_class_fused_loader(
    local_view: np.ndarray,
    tabular_features: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    dataset = TensorDataset(
        torch.tensor(local_view, dtype=torch.float32),
        torch.tensor(tabular_features, dtype=torch.float32),
        torch.tensor(y, dtype=torch.long),
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=False,
    )


def class_weights_from_train_targets(
    y_train: np.ndarray,
    device: torch.device,
) -> torch.Tensor:
    counts = np.bincount(y_train, minlength=3).astype(np.float32)

    if (counts <= 0).any():
        raise ValueError(f"train split is missing classes: counts={counts.tolist()}")

    weights = counts.sum() / (3.0 * counts)

    return torch.tensor(weights, dtype=torch.float32, device=device)


def train_one_epoch_three_class(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.train()

    total_loss = 0.0
    total_count = 0

    for batch_local, batch_tabular, batch_y in loader:
        batch_local = batch_local.to(device)
        batch_tabular = batch_tabular.to(device)
        batch_y = batch_y.to(device)

        optimizer.zero_grad(set_to_none=True)

        logits = model(
            local_view=batch_local,
            tabular_features=batch_tabular,
        )
        loss = criterion(logits, batch_y)

        loss.backward()
        optimizer.step()

        batch_count = batch_local.shape[0]
        total_loss += float(loss.detach().cpu()) * batch_count
        total_count += batch_count

    return total_loss / max(total_count, 1)


@torch.no_grad()
def predict_three_class_model(
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

    for batch_local, batch_tabular, batch_y in loader:
        batch_local = batch_local.to(device)
        batch_tabular = batch_tabular.to(device)
        batch_y = batch_y.to(device)

        logits = model(
            local_view=batch_local,
            tabular_features=batch_tabular,
        )
        loss = criterion(logits, batch_y)

        batch_count = batch_local.shape[0]
        total_loss += float(loss.detach().cpu()) * batch_count
        total_count += batch_count

        all_logits.append(logits.detach().cpu().numpy())
        all_targets.append(batch_y.detach().cpu().numpy())

    logits_np = np.concatenate(all_logits)
    targets_np = np.concatenate(all_targets)

    return total_loss / max(total_count, 1), logits_np, targets_np


def softmax_np(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)

    return exp / exp.sum(axis=1, keepdims=True)


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

    y_three_class = build_three_class_targets(
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
    print("class_names:", CLASS_NAMES)
    print("three-class target summary:")
    print(
        describe_three_class_targets(
            labels=y_binary,
            disposition=disposition,
            three_class_targets=y_three_class,
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

    for seed in tqdm(EVAL_SEEDS, desc="three-class fused local model seeds"):
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

        train_loader = make_three_class_fused_loader(
            local_view=split_features["train"]["local"],
            tabular_features=split_features["train"]["tabular"],
            y=y_three_class[splits["train"]],
            batch_size=BATCH_SIZE,
            shuffle=True,
        )

        eval_loaders = {
            split_name: make_three_class_fused_loader(
                local_view=features["local"],
                tabular_features=features["tabular"],
                y=y_three_class[splits[split_name]],
                batch_size=BATCH_SIZE,
                shuffle=False,
            )
            for split_name, features in split_features.items()
        }

        class_weights = class_weights_from_train_targets(
            y_train=y_three_class[splits["train"]],
            device=device,
        )

        model = ThreeClassFusedTabularLocalCNN(
            local_input_length=x_local_raw.shape[1],
            n_tabular_features=x_tabular_unscaled.shape[1],
        ).to(device)

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY,
        )
        criterion = nn.CrossEntropyLoss(weight=class_weights)

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
            train_loss = train_one_epoch_three_class(
                model=model,
                loader=train_loader,
                optimizer=optimizer,
                criterion=criterion,
                device=device,
            )

            val_loss, val_logits, _ = predict_three_class_model(
                model=model,
                loader=eval_loaders["val"],
                criterion=criterion,
                device=device,
            )

            val_probs = softmax_np(val_logits)
            val_planet_like_score = (
                val_probs[:, CANDIDATE_CLASS]
                + val_probs[:, CONFIRMED_CLASS]
            )

            val_auc = evaluate_predictions(
                model_name=MODEL_NAME,
                seed=seed,
                split_name="val",
                y_true=y_binary[splits["val"]],
                y_score=val_planet_like_score,
            )["roc_auc"]

            training_history_rows.append(
                {
                    "seed": seed,
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "val_binary_roc_auc": val_auc,
                    "false_positive_class_weight": float(class_weights[FALSE_POSITIVE_CLASS].detach().cpu()),
                    "candidate_class_weight": float(class_weights[CANDIDATE_CLASS].detach().cpu()),
                    "confirmed_class_weight": float(class_weights[CONFIRMED_CLASS].detach().cpu()),
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
            split_loss, logits, three_class_targets = predict_three_class_model(
                model=model,
                loader=eval_loaders[split_name],
                criterion=criterion,
                device=device,
            )

            probs = softmax_np(logits)
            planet_like_score = (
                probs[:, CANDIDATE_CLASS]
                + probs[:, CONFIRMED_CLASS]
            )
            predicted_class = probs.argmax(axis=1).astype(np.int64)
            targets_int = y_binary[indices].astype(np.int64)

            record = evaluate_predictions(
                model_name=MODEL_NAME,
                seed=seed,
                split_name=split_name,
                y_true=targets_int,
                y_score=planet_like_score,
            )
            record["loss"] = split_loss
            record["best_epoch"] = best_epoch
            record["best_val_loss"] = best_val_loss
            record["local_train_mean"] = local_train_mean
            record["local_train_std"] = local_train_std
            record["false_positive_class_weight"] = float(class_weights[FALSE_POSITIVE_CLASS].detach().cpu())
            record["candidate_class_weight"] = float(class_weights[CANDIDATE_CLASS].detach().cpu())
            record["confirmed_class_weight"] = float(class_weights[CONFIRMED_CLASS].detach().cpu())

            metrics_rows.append(record)

            predictions = make_predictions_frame(
                model_name=MODEL_NAME,
                seed=seed,
                split_name=split_name,
                indices=indices,
                y_true=targets_int,
                y_score=planet_like_score,
                kepid=kepid,
                kepoi_name=kepoi_name,
                disposition=disposition,
            )
            predictions["false_positive_score"] = probs[:, FALSE_POSITIVE_CLASS]
            predictions["candidate_score"] = probs[:, CANDIDATE_CLASS]
            predictions["confirmed_score"] = probs[:, CONFIRMED_CLASS]
            predictions["three_class_target"] = three_class_targets
            predictions["three_class_pred"] = predicted_class
            predictions["three_class_target_name"] = [
                CLASS_NAMES[int(value)]
                for value in three_class_targets
            ]
            predictions["three_class_pred_name"] = [
                CLASS_NAMES[int(value)]
                for value in predicted_class
            ]

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
                "architecture": "ThreeClassFusedTabularLocalCNN",
                "training_target": "three_class_disposition",
                "class_names": CLASS_NAMES,
                "false_positive_class_weight": float(class_weights[FALSE_POSITIVE_CLASS].detach().cpu()),
                "candidate_class_weight": float(class_weights[CANDIDATE_CLASS].detach().cpu()),
                "confirmed_class_weight": float(class_weights[CONFIRMED_CLASS].detach().cpu()),
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
                "false_positive_class_weight",
                "candidate_class_weight",
                "confirmed_class_weight",
            ]
        ].to_string(index=False)
    )
    print()
    print("wrote:", PER_SEED_METRICS_PATH)
    print("wrote:", SUMMARY_METRICS_PATH)
    print("wrote:", PREDICTIONS_PATH)
    print("wrote:", TRAINING_HISTORY_PATH)
    print("wrote:", MODEL_PATH)
    print("TRAIN_THREE_CLASS_FUSED_LOCAL_MODEL_OK")


if __name__ == "__main__":
    main()