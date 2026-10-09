"""Known-attack classifier with safe label encoding and rollback-aware adaptation."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder


@dataclass
class ClassifierArtifacts:
    labels: list[str]
    val_f1_macro: float
    report: dict[str, Any]


class KnownAttackClassifier:
    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.model = None
        self.label_encoder = LabelEncoder()
        self.labels: list[str] = []
        self.training_device = "cpu"
        self._is_fitted = False
        self._last_adapt_accepted = False

    def _make_model(self):
        cfg = self.config.get("classifier", {})
        kind = cfg.get("model_type", "xgboost").lower()
        seed = self.config.get("project", {}).get("random_state", 42)
        if kind == "random_forest":
            p = cfg.get("random_forest", {})
            return RandomForestClassifier(
                n_estimators=int(p.get("n_estimators", 300)),
                max_depth=p.get("max_depth"),
                random_state=seed,
                n_jobs=-1,
                class_weight="balanced",
            )
        if kind == "lightgbm":
            try:
                from lightgbm import LGBMClassifier
            except ImportError as exc:
                raise ImportError("lightgbm is required for classifier.model_type=lightgbm") from exc
            p = cfg.get("lightgbm", {})
            return LGBMClassifier(
                n_estimators=int(p.get("n_estimators", 300)),
                learning_rate=float(p.get("learning_rate", 0.08)),
                max_depth=int(p.get("max_depth", -1)),
                random_state=seed,
                verbosity=-1,
            )
        try:
            from xgboost import XGBClassifier
        except ImportError as exc:
            raise ImportError("xgboost is required for classifier.model_type=xgboost") from exc
        p = cfg.get("xgboost", {})
        params = dict(
            n_estimators=int(p.get("n_estimators", 300)),
            learning_rate=float(p.get("learning_rate", 0.08)),
            max_depth=int(p.get("max_depth", 6)),
            subsample=float(p.get("subsample", 0.9)),
            colsample_bytree=float(p.get("colsample_bytree", 0.9)),
            reg_lambda=float(p.get("reg_lambda", 1.0)),
            random_state=seed,
            eval_metric="mlogloss",
            tree_method="hist",
        )
        use_gpu = bool(cfg.get("use_gpu", False))
        if use_gpu:
            try:
                import torch
                if torch.cuda.is_available():
                    params["device"] = "cuda"
                    self.training_device = "cuda"
            except Exception:
                pass
        return XGBClassifier(**params)

    @staticmethod
    def _balanced_sample_weights(y: np.ndarray) -> np.ndarray:
        y = np.asarray(y)
        labels, counts = np.unique(y, return_counts=True)
        total = len(y)
        n_classes = len(labels)
        weights = {label: total / (n_classes * count) for label, count in zip(labels, counts)}
        return np.asarray([weights[v] for v in y], dtype=float)

    def fit(self, X_train, y_train, X_val, y_val) -> ClassifierArtifacts:
        y_train = np.asarray(y_train).astype(str)
        y_val = np.asarray(y_val).astype(str)
        self.label_encoder.fit(y_train)
        self.labels = self.label_encoder.classes_.tolist()
        yt = self.label_encoder.transform(y_train)
        model = self._make_model()
        kwargs = {}
        if hasattr(model, "sample_weight") or "random_forest" == self.config.get("classifier", {}).get("model_type"):
            kwargs["sample_weight"] = self._balanced_sample_weights(y_train)
        try:
            model.fit(X_train, yt, **kwargs)
        except TypeError:
            model.fit(X_train, yt)
        self.model = model
        self._is_fitted = True
        preds, _ = self.predict(X_val)
        score = float(f1_score(y_val, preds, average="macro", labels=self.labels, zero_division=0))
        return ClassifierArtifacts(labels=self.labels, val_f1_macro=score, report={"f1_macro": score})

    def predict(self, X):
        if not self._is_fitted or self.model is None:
            raise RuntimeError("Classifier is not trained")
        X = np.asarray(X)
        encoded = self.model.predict(X)
        encoded = np.asarray(encoded, dtype=int)
        preds = self.label_encoder.inverse_transform(encoded)
        if hasattr(self.model, "predict_proba"):
            probs = np.asarray(self.model.predict_proba(X))
            confidence = probs.max(axis=1).astype(float)
        else:
            confidence = np.ones(len(preds), dtype=float)
        return preds, confidence

    def partial_adapt(self, X_recent, y_recent, X_past=None, y_past=None):
        """Retrain a candidate on recent data plus optional replay, accepting it only if validation F1 does not degrade."""
        X_recent = np.asarray(X_recent)
        y_recent = np.asarray(y_recent).astype(str)
        if len(X_recent) == 0:
            self._last_adapt_accepted = False
            return False
        X_parts = [X_recent]
        y_parts = [y_recent]
        if X_past is not None and y_past is not None and len(X_past):
            X_parts.append(np.asarray(X_past))
            y_parts.append(np.asarray(y_past).astype(str))
        X_all = np.vstack(X_parts)
        y_all = np.concatenate(y_parts)
        # Keep the original class vocabulary. Unknown labels are allowed to be added only after explicit evidence.
        candidate_encoder = LabelEncoder().fit(np.concatenate([self.labels, y_all]))
        candidate = self._make_model()
        yt = candidate_encoder.transform(y_all)
        try:
            candidate.fit(X_all, yt, sample_weight=self._balanced_sample_weights(y_all))
        except TypeError:
            candidate.fit(X_all, yt)

        old_score = None
        val_X = getattr(self, "_adapt_val_X", None)
        val_y = getattr(self, "_adapt_val_y", None)
        if val_X is not None and val_y is not None and len(val_X):
            old_pred, _ = self.predict(val_X)
            old_score = f1_score(np.asarray(val_y).astype(str), old_pred, average="macro", zero_division=0)
            cand_encoded = candidate.predict(val_X).astype(int)
            cand_pred = candidate_encoder.inverse_transform(cand_encoded)
            new_score = f1_score(np.asarray(val_y).astype(str), cand_pred, average="macro", zero_division=0)
            tolerance = float(self.config.get("drift", {}).get("adapt_accept_tolerance", 0.0))
            accepted = new_score + tolerance >= old_score
        else:
            accepted = True

        if accepted:
            self.model = candidate
            self.label_encoder = candidate_encoder
            self.labels = candidate_encoder.classes_.tolist()
            self._is_fitted = True
        self._last_adapt_accepted = bool(accepted)
        return bool(accepted)
