"""PyTorch autoencoder used to flag low-confidence zero-day traffic."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class AutoencoderNet(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: list[int]):
        super().__init__()
        if len(hidden_dims) != 3:
            raise ValueError("hidden_dims must have length 3")
        h1, h2, h3 = map(int, hidden_dims)
        self.encoder = nn.Sequential(nn.Linear(input_dim, h1), nn.ReLU(), nn.Linear(h1, h2), nn.ReLU(), nn.Linear(h2, h3))
        self.decoder = nn.Sequential(nn.Linear(h3, h2), nn.ReLU(), nn.Linear(h2, h1), nn.ReLU(), nn.Linear(h1, input_dim))

    def forward(self, x):
        return self.decoder(self.encoder(x))


@dataclass
class AutoencoderArtifacts:
    threshold: float
    model: AutoencoderNet
    val_reconstruction_mean: float
    val_reconstruction_std: float


class ZeroDayAutoencoder:
    def __init__(self, config: dict[str, Any]):
        self.config = config
        requested_gpu = bool(config.get("anomaly", {}).get("use_gpu", False))
        self.device = torch.device("cuda" if requested_gpu and torch.cuda.is_available() else "cpu")
        self.model: AutoencoderNet | None = None
        self.threshold: float = 0.0
        self.training_device = str(self.device)

    def _ensure_model(self, input_dim: int):
        if self.model is None:
            dims = self.config.get("anomaly", {}).get("hidden_dims", [128, 64, 32])
            self.model = AutoencoderNet(input_dim, dims).to(self.device)

    def fit(self, X_train_normal: np.ndarray, X_val_normal: np.ndarray) -> AutoencoderArtifacts:
        X_train_normal = np.asarray(X_train_normal, dtype=np.float32)
        X_val_normal = np.asarray(X_val_normal, dtype=np.float32)
        if len(X_train_normal) == 0 or len(X_val_normal) == 0:
            raise ValueError("Autoencoder requires non-empty normal training and validation data")
        self._ensure_model(X_train_normal.shape[1])
        assert self.model is not None
        ds = TensorDataset(torch.from_numpy(X_train_normal))
        loader = DataLoader(ds, batch_size=int(self.config.get("anomaly", {}).get("batch_size", 256)), shuffle=True)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=float(self.config.get("anomaly", {}).get("learning_rate", 1e-3)))
        criterion = nn.MSELoss()
        self.model.train()
        epochs = int(self.config.get("anomaly", {}).get("epochs", 25))
        for _ in range(max(1, epochs)):
            for (xb,) in loader:
                xb = xb.to(self.device)
                optimizer.zero_grad(set_to_none=True)
                loss = criterion(self.model(xb), xb)
                loss.backward()
                optimizer.step()
        errors = self.reconstruction_error(X_val_normal)
        percentile = float(self.config.get("anomaly", {}).get("threshold_percentile", 95))
        self.threshold = float(np.percentile(errors, percentile))
        return AutoencoderArtifacts(self.threshold, self.model, float(errors.mean()), float(errors.std()))

    def reconstruction_error(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Autoencoder is not trained")
        arr = np.asarray(X, dtype=np.float32)
        self.model.eval()
        with torch.no_grad():
            xb = torch.from_numpy(arr).to(self.device)
            recon = self.model(xb)
            errors = torch.mean((recon - xb) ** 2, dim=1).detach().cpu().numpy()
        return errors.astype(float)

    def is_anomaly(self, X: np.ndarray):
        errors = self.reconstruction_error(X)
        return errors > float(self.threshold), errors

    def update_threshold(self, X_normal: np.ndarray):
        errors = self.reconstruction_error(X_normal)
        percentile = float(self.config.get("anomaly", {}).get("threshold_percentile", 95))
        self.threshold = float(np.percentile(errors, percentile))
        return self.threshold
