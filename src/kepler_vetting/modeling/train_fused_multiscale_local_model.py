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
    PhaseViewCNN,
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
from kepler_vetting.modeling.train_tabular_baseline import (
    load_unstandardized_tabular_features,
)
from kepler_vetting.processing.common import (
    MODEL_READY_NPZ_PATH,
    RUN_METRICS_DIR,
    RUN_MODEL_DIR,
)


MODEL_NAME = "fused_tabular_multiscale_local_cnn"

LOCAL_VIEW_KEYS = [
    "local_view_narrow",
    "local_view",
    "local_view_wide",
]

LOCAL_VIEW_NAMES = [
    "narrow",
    "current",
    "wide",
]

METRICS_DIR = RUN_METRICS_DIR
MODEL_DIR = RUN_MODEL_DIR

PER_SEED_METRICS_PATH = METRICS_DIR / "fused_multiscale_local_model_metrics_by_seed.csv"
SUMMARY_METRICS_PATH = METRICS_DIR / "fused_multiscale_local_model_metrics_summary.csv"
PREDICTIONS_PATH = METRICS_DIR / "fused_multiscale_local_model_predictions.csv"
TRAINING_HISTORY_PATH = METRICS_DIR / "fused_multiscale_local_model_training_history.csv"
MODEL_PATH = MODEL_DIR / "fused_tabular_multiscale_local_cnn.pt"


class MultiScaleLocalEncoder(nn.Module):
    def __init__(self, input_length: int) -> None:
        super().__init__()

        self.encoders = nn.ModuleDict(
            {
                name: PhaseViewCNN(input_length=input_length).features
                for name in LOCAL_VIEW_NAMES
            }
        )

        self.output_dim = 64 * len(LOCAL_VIEW_NAMES)

    def forward(
        self,
        narrow: torch.Tensor,
        current: torch.Tensor,
        wide: torch.Tensor,
    ) -> torch.Tensor:
        inputs = {
            "narrow": narrow,
            "current": current,
            "wide": wide,
        }

        embeddings = []

        for name in LOCAL_VIEW_NAMES:
            embedding = self.encoders[name](inputs[name]).flatten(start_dim=1)
            embeddings.append(embedding)

        return torch.cat(embeddings, dim=1)


class FusedTabularMultiScaleLocalCNN(nn.Module):
    def __init__(
        self,
        local_input_length: int,
        n_tabular_features: int,
    ) -> None:
        super().__init__()

        self.local_encoder = MultiScaleLocalEncoder(
            input_length=local_input_length,
        )

        self.tabular_encoder = nn.Sequential(
            nn.Linear(n_tabular_features, 32),
            nn.ReLU(),
            nn.Dropout(p=0.10),
            nn.Linear(32, 16),
            nn.ReLU(),
        )

        self.classifier = nn.Sequential(
            nn.Dropout(p=0.30),
            nn.Linear(self.local_encoder.output_dim + 16, 96),
            nn.ReLU(),
            nn.Dropout(p=0.20),
            nn.Linear(96, 32),
            nn.ReLU(),
            nn.Dropout(p=0.10),
            nn.Linear(32, 1),
        )

        self.local_input_length = local_input_length
        self.n_tabular_features = n_tabular_features

    def forward(
        self,
        local_view_narrow: torch.Tensor,
        local_view_current: torch.Tensor,
        local_view_wide: torch.Tensor,
        tabular_features: torch.Tensor,
    ) -> torch.Tensor:
        local_embedding = self.local_encoder(
            narrow=local_view_narrow,
            current=local_view_current,
            wide=local_view_wide,
        )
        tabular_embedding = self.tabular_encoder(tabular_features)

        fused = torch.cat(
            [
                local_embedding,
                tabular_embedding,
            ],
            dim=1,
        )

        logits = self.classifier(fused)

        return logits.squeeze(1)


def make_multiscale_loader(
    local_views: dict[str, np.ndarray],
    tabular_features: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    dataset = TensorDataset(
        torch.tensor(local_views["local_view_narrow"], dtype=torch.float32),
        torch.tensor(local_views["local_view"], dtype=torch.float32),
        torch.tensor(local_views["local_view_wide"], dtype=torch.float32),
        torch.tensor(tabular_features, dtype=torch.float32),
        torch.tensor(y, dtype=torch.float32),
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=False,
    )


def train_one_epoch_multiscale(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.train()

    total_loss = 0.0
    total_count = 0

    for batch_narrow, batch_current, batch_wide, batch_tabular, batch_y in loader:
        batch_narrow = batch_narrow.to(device)
        batch_current = batch_current.to(device)
        batch_wide = batch_wide.to(device)
        batch_tabular = batch_tabular.to(device)
        batch_y = batch_y.to(device)

        optimizer.zero_grad(set_to_none=True)

        logits = model(
            local_view_narrow=batch_narrow,
            local_view_current=batch_current,
            local_view_wide=batch_wide,
            tabular_features=batch_tabular,
        )
        loss = criterion(logits, batch_y)

        loss.backward()
        optimizer.step()

        batch_count = batch_narrow.shape[0]
        total_loss += float(loss.detach().cpu()) * batch_count
        total_count += batch_count

    return total_loss / max(total_count, 1)


@torch.no_grad()
def predict_multiscale_model(
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

    for batch_narrow, batch_current, batch_wide, batch_tabular, batch_y in loader:
        batch_narrow = batch_narrow.to(device)
        batch_current = batch_current.to(device)
        batch_wide = batch_wide.to(device)
        batch_tabular = batch_tabular.to(device)
        batch_y = batch_y.to(device)

        logits = model(
            local_view_narrow=batch_narrow,
            local_view_current=batch_current,
            local_view_wide=batch_wide,
            tabular_features=batch_tabular,
        )
        loss = criterion(logits, batch_y)

        batch_count = batch_narrow.shape[0]
        total_loss += float(loss.detach().cpu()) * batch_count
        total_count += batch_count

        all_logits.append(logits.detach().cpu().numpy())
        all_targets.append(batch_y.detach().cpu().numpy())

    logits_np = np.concatenate(all_logits)
    targets_np = np.concatenate(all_targets)

    return total_loss / max(total_count, 1), logits_np, targets_np


def main() -> None:
    if not MODEL_READY_NPZ_PATH.exists():
        raise FileNotFoundError(
            f"missing model-ready dataset: {MODEL_READY_NPZ_PATH}. "
            "Run kepler_vetting.processing.filter_model_ready_dataset first."
        )

    data = np.load(MODEL_READY_NPZ_PATH)

    missing_views = [
        key
        for key in LOCAL_VIEW_KEYS
        if key not in data.files
    ]

    if missing_views:
        raise ValueError(
            "model-ready dataset is missing multi-scale local views: "
            f"{missing_views}. Rebuild the processed dataset and rerun the "
            "model-ready filter."
        )

    x_tabular_unscaled = load_unstandardized_tabular_features(data)
    x_local_raw_by_key = {
        key: data[key].astype(np.float32)
        for key in LOCAL_VIEW_KEYS
    }
    y = data["labels"].astype(np.int64)

    kepid = data["kepid"]
    kepoi_name = data["kepoi_name"]
    disposition = data["disposition"]
    feature_names = data["feature_names"].astype(str)

    for key, values in x_local_raw_by_key.items():
        if values.ndim != 2:
            raise ValueError(f"{key} must be 2D; got shape {values.shape}")

        if values.shape[0] != y.shape[0]:
            raise ValueError(
                f"{key} and labels row counts differ: {values.shape[0]} vs {y.shape[0]}"
            )

    local_input_lengths = {
        key: values.shape[1]
        for key, values in x_local_raw_by_key.items()
    }

    if len(set(local_input_lengths.values())) != 1:
        raise ValueError(f"local input lengths differ: {local_input_lengths}")

    local_input_length = next(iter(local_input_lengths.values()))

    if x_tabular_unscaled.ndim != 2:
        raise ValueError(
            f"tabular features must be 2D; got shape {x_tabular_unscaled.shape}"
        )

    if x_tabular_unscaled.shape[0] != y.shape[0]:
        raise ValueError(
            "tabular features and labels row counts differ: "
            f"{x_tabular_unscaled.shape[0]} vs {y.shape[0]}"
        )

    if set(y.tolist()) - {0, 1}:
        raise ValueError(
            f"labels must only contain 0/1 values; got {sorted(set(y.tolist()))}"
        )

    device = get_device()

    print("dataset:", MODEL_READY_NPZ_PATH)
    print("model:", MODEL_NAME)
    print("local_views:", LOCAL_VIEW_KEYS)
    print("n_rows:", y.shape[0])
    print("local_input_lengths:", local_input_lengths)
    print("n_tabular_features:", x_tabular_unscaled.shape[1])
    print("feature_names:", feature_names.tolist())
    print("eval_seeds:", list(EVAL_SEEDS))
    print("split_mode:", SPLIT_MODE)
    print("device:", device)
    print()

    metrics_rows = []
    prediction_frames = []
    training_history_rows = []

    final_model_payload = None

    for seed in tqdm(EVAL_SEEDS, desc="fused multiscale local model seeds"):
        set_seed(seed)

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

        x_local_for_torch_by_key = {}
        local_train_stats = {}

        for key, values in x_local_raw_by_key.items():
            normalized, train_mean, train_std = normalize_from_train(
                x=values,
                train_idx=splits["train"],
            )

            x_local_for_torch_by_key[key] = normalized[:, None, :]
            local_train_stats[key] = {
                "mean": train_mean,
                "std": train_std,
            }

        tabular_scaler = StandardScaler()
        x_tabular_train = tabular_scaler.fit_transform(
            x_tabular_unscaled[splits["train"]]
        )

        split_features = {}

        for split_name, indices in splits.items():
            if split_name == "train":
                tabular_values = x_tabular_train.astype(np.float32)
            else:
                tabular_values = tabular_scaler.transform(
                    x_tabular_unscaled[indices]
                ).astype(np.float32)

            split_features[split_name] = {
                "local": {
                    key: values[indices]
                    for key, values in x_local_for_torch_by_key.items()
                },
                "tabular": tabular_values,
            }

        train_loader = make_multiscale_loader(
            local_views=split_features["train"]["local"],
            tabular_features=split_features["train"]["tabular"],
            y=y[splits["train"]],
            batch_size=BATCH_SIZE,
            shuffle=True,
        )

        eval_loaders = {
            split_name: make_multiscale_loader(
                local_views=features["local"],
                tabular_features=features["tabular"],
                y=y[splits[split_name]],
                batch_size=BATCH_SIZE,
                shuffle=False,
            )
            for split_name, features in split_features.items()
        }

        y_train = y[splits["train"]]
        n_pos = int((y_train == 1).sum())
        n_neg = int((y_train == 0).sum())

        if n_pos == 0 or n_neg == 0:
            raise ValueError(f"seed={seed} has a single-class train split")

        pos_weight = torch.tensor([n_neg / n_pos], dtype=torch.float32, device=device)

        model = FusedTabularMultiScaleLocalCNN(
            local_input_length=local_input_length,
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
            train_loss = train_one_epoch_multiscale(
                model=model,
                loader=train_loader,
                optimizer=optimizer,
                criterion=criterion,
                device=device,
            )

            val_loss, val_logits, val_targets = predict_multiscale_model(
                model=model,
                loader=eval_loaders["val"],
                criterion=criterion,
                device=device,
            )

            val_scores = sigmoid_np(val_logits)
            val_auc = evaluate_predictions(
                model_name=MODEL_NAME,
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

        for split_name, indices in splits.items():
            split_loss, logits, targets = predict_multiscale_model(
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

            for key, stats in local_train_stats.items():
                record[f"{key}_train_mean"] = stats["mean"]
                record[f"{key}_train_std"] = stats["std"]

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
                "local_input_length": local_input_length,
                "local_view_keys": LOCAL_VIEW_KEYS,
                "local_view_names": LOCAL_VIEW_NAMES,
                "n_tabular_features": x_tabular_unscaled.shape[1],
                "feature_names": feature_names,
                "local_train_stats": local_train_stats,
                "tabular_scaler": tabular_scaler,
                "seed": seed,
                "best_epoch": best_epoch,
                "best_val_loss": best_val_loss,
                "source_dataset": str(MODEL_READY_NPZ_PATH),
                "split_mode": SPLIT_MODE,
                "architecture": "FusedTabularMultiScaleLocalCNN",
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
            ]
        ].to_string(index=False)
    )
    print()
    print("wrote:", PER_SEED_METRICS_PATH)
    print("wrote:", SUMMARY_METRICS_PATH)
    print("wrote:", PREDICTIONS_PATH)
    print("wrote:", TRAINING_HISTORY_PATH)
    print("wrote:", MODEL_PATH)
    print("TRAIN_FUSED_MULTISCALE_LOCAL_MODEL_OK")


if __name__ == "__main__":
    main()