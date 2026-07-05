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


MODEL_NAME = "fused_tabular_residual_local_cnn"
VIEW_NAME = "local_view"

METRICS_DIR = RUN_METRICS_DIR
MODEL_DIR = RUN_MODEL_DIR

PER_SEED_METRICS_PATH = METRICS_DIR / "fused_residual_local_model_metrics_by_seed.csv"
SUMMARY_METRICS_PATH = METRICS_DIR / "fused_residual_local_model_metrics_summary.csv"
PREDICTIONS_PATH = METRICS_DIR / "fused_residual_local_model_predictions.csv"
TRAINING_HISTORY_PATH = METRICS_DIR / "fused_residual_local_model_training_history.csv"
MODEL_PATH = MODEL_DIR / "fused_tabular_residual_local_cnn.pt"


class ResidualConvBlock1d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float,
    ) -> None:
        super().__init__()

        padding = dilation * (kernel_size - 1) // 2

        self.main = nn.Sequential(
            nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=padding,
                dilation=dilation,
            ),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Conv1d(
                out_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=padding,
                dilation=dilation,
            ),
            nn.BatchNorm1d(out_channels),
        )

        if in_channels == out_channels:
            self.skip = nn.Identity()
        else:
            self.skip = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm1d(out_channels),
            )

        self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(self.main(x) + self.skip(x))


class ResidualLocalEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
        )

        self.blocks = nn.Sequential(
            ResidualConvBlock1d(
                in_channels=32,
                out_channels=32,
                kernel_size=5,
                dilation=1,
                dropout=0.05,
            ),
            ResidualConvBlock1d(
                in_channels=32,
                out_channels=32,
                kernel_size=5,
                dilation=2,
                dropout=0.05,
            ),
            ResidualConvBlock1d(
                in_channels=32,
                out_channels=64,
                kernel_size=5,
                dilation=4,
                dropout=0.10,
            ),
            ResidualConvBlock1d(
                in_channels=64,
                out_channels=64,
                kernel_size=5,
                dilation=8,
                dropout=0.10,
            ),
            ResidualConvBlock1d(
                in_channels=64,
                out_channels=128,
                kernel_size=3,
                dilation=1,
                dropout=0.10,
            ),
        )

        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.output_dim = 256

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.blocks(x)

        avg_embedding = self.avg_pool(x).flatten(start_dim=1)
        max_embedding = self.max_pool(x).flatten(start_dim=1)

        return torch.cat(
            [
                avg_embedding,
                max_embedding,
            ],
            dim=1,
        )


class FusedTabularResidualLocalCNN(nn.Module):
    def __init__(
        self,
        local_input_length: int,
        n_tabular_features: int,
    ) -> None:
        super().__init__()

        self.local_encoder = ResidualLocalEncoder()

        self.tabular_encoder = nn.Sequential(
            nn.Linear(n_tabular_features, 32),
            nn.ReLU(),
            nn.Dropout(p=0.10),
            nn.Linear(32, 16),
            nn.ReLU(),
        )

        self.classifier = nn.Sequential(
            nn.Dropout(p=0.30),
            nn.Linear(self.local_encoder.output_dim + 16, 128),
            nn.ReLU(),
            nn.Dropout(p=0.20),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Dropout(p=0.10),
            nn.Linear(32, 1),
        )

        self.local_input_length = local_input_length
        self.n_tabular_features = n_tabular_features

    def forward(
        self,
        local_view: torch.Tensor,
        tabular_features: torch.Tensor,
    ) -> torch.Tensor:
        local_embedding = self.local_encoder(local_view)
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


def main() -> None:
    if not MODEL_READY_NPZ_PATH.exists():
        raise FileNotFoundError(
            f"missing model-ready dataset: {MODEL_READY_NPZ_PATH}. "
            "Run kepler_vetting.processing.filter_model_ready_dataset first."
        )

    data = np.load(MODEL_READY_NPZ_PATH)

    x_tabular_unscaled = load_unstandardized_tabular_features(data)
    x_local_raw = data[VIEW_NAME].astype(np.float32)
    y = data["labels"].astype(np.int64)

    kepid = data["kepid"]
    kepoi_name = data["kepoi_name"]
    disposition = data["disposition"]
    feature_names = data["feature_names"].astype(str)

    if x_local_raw.ndim != 2:
        raise ValueError(f"{VIEW_NAME} must be 2D; got shape {x_local_raw.shape}")

    if x_tabular_unscaled.ndim != 2:
        raise ValueError(
            f"tabular features must be 2D; got shape {x_tabular_unscaled.shape}"
        )

    if x_local_raw.shape[0] != y.shape[0]:
        raise ValueError(
            f"{VIEW_NAME} and labels row counts differ: "
            f"{x_local_raw.shape[0]} vs {y.shape[0]}"
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
    print("view:", VIEW_NAME)
    print("n_rows:", y.shape[0])
    print("local_input_length:", x_local_raw.shape[1])
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

    for seed in tqdm(EVAL_SEEDS, desc="fused residual local model seeds"):
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
            y=y[splits["train"]],
            batch_size=BATCH_SIZE,
            shuffle=True,
        )

        eval_loaders = {
            split_name: make_fused_loader(
                local_view=features["local"],
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

        model = FusedTabularResidualLocalCNN(
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

            val_loss, val_logits, val_targets = predict_fused_model(
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
            split_loss, logits, targets = predict_fused_model(
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
            record["local_train_mean"] = local_train_mean
            record["local_train_std"] = local_train_std

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
                "architecture": "FusedTabularResidualLocalCNN",
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
    print("TRAIN_FUSED_RESIDUAL_LOCAL_MODEL_OK")


if __name__ == "__main__":
    main()