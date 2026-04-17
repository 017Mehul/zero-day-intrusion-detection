"""Batch streaming engine with hybrid prediction and drift-triggered adaptation."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd
import psutil

from sa_zd_nids.drift.detector import DriftDetector
from sa_zd_nids.evaluation.metrics import compute_classification_metrics, compute_zero_day_metrics
from sa_zd_nids.utils.io import append_jsonl, write_csv
from sa_zd_nids.utils.io import save_model_atomic, save_metadata_atomic
import json
from pathlib import Path
from datetime import datetime


@dataclass
class StreamArtifacts:
    records: pd.DataFrame
    metrics: dict[str, Any]


class StreamingEngine:
    def __init__(self, config: dict, classifier, autoencoder):
        self.config = config
        self.classifier = classifier
        self.autoencoder = autoencoder
        self.detector = DriftDetector(delta=config["drift"].get("adwin_delta", 0.002))

    def run(
        self,
        X: np.ndarray,
        y: np.ndarray,
        timestamps: pd.Series,
        benign_label: str,
        known_labels: list[str] | None = None,
        baseline_feature_mean: np.ndarray | None = None,
        baseline_feature_std: np.ndarray | None = None,
        recent_train_cache: tuple[np.ndarray, np.ndarray] | None = None,
    ) -> StreamArtifacts:
        batch_size = self.config["streaming"].get("batch_size", 500)
        max_batches = self.config["streaming"].get("max_batches")
        conf_thr = self.config["classifier"].get("confidence_threshold", 0.7)
        conf_low_thr = self.config["drift"].get("confidence_low_threshold", conf_thr)
        acc_weight = float(self.config["drift"].get("accuracy_weight", 0.6))
        conf_weight = float(self.config["drift"].get("confidence_weight", 0.3))
        feature_weight = float(self.config["drift"].get("feature_shift_weight", 0.1))
        window_size = int(self.config["drift"].get("window_size", 1000))
        replay_fraction = float(self.config["drift"].get("past_replay_fraction", 0.3))
        csv_path = self.config["logging"]["output_csv"]
        events_path = self.config["logging"]["events_jsonl"]

        rows = []
        proc = psutil.Process()
        rng = np.random.default_rng(self.config["project"].get("random_state", 42))

        y_pred_total = []
        total_adapt_time = 0.0
        drift_count = 0
        drift_indices: list[int] = []

        if recent_train_cache is None:
            recent_X = np.empty((0, X.shape[1]), dtype=X.dtype)
            recent_y = np.empty((0,), dtype=y.dtype)
        else:
            recent_X, recent_y = recent_train_cache

        for batch_i, batch_start in enumerate(range(0, len(X), batch_size)):
            if max_batches is not None and batch_i >= int(max_batches):
                break
            batch_end = min(batch_start + batch_size, len(X))
            Xb = X[batch_start:batch_end]
            yb = y[batch_start:batch_end]
            tsb = timestamps.iloc[batch_start:batch_end] if hasattr(timestamps, "iloc") else timestamps[batch_start:batch_end]

            t0 = perf_counter()
            pred_cls, conf = self.classifier.predict(Xb)
            low_conf_mask = conf < conf_thr

            final_pred = pred_cls.astype(object)
            recon_err = np.full(len(Xb), np.nan, dtype=float)

            if np.any(low_conf_mask):
                flags, errs = self.autoencoder.is_anomaly(Xb[low_conf_mask])
                recon_err[low_conf_mask] = errs
                low_conf_idx = np.where(low_conf_mask)[0]
                for idx, is_anom in zip(low_conf_idx, flags):
                    final_pred[idx] = "ZERO_DAY" if is_anom else benign_label

            batch_latency_ms = ((perf_counter() - t0) * 1000.0) / max(1, len(Xb))
            cpu = psutil.cpu_percent(interval=None)
            mem_mb = proc.memory_info().rss / (1024 * 1024)

            batch_acc = np.mean(final_pred == yb)
            low_conf_rate = float(np.mean(conf < conf_low_thr))
            mean_conf = float(np.mean(conf))
            feature_shift = 0.0
            if baseline_feature_mean is not None and baseline_feature_std is not None:
                denom = np.maximum(np.asarray(baseline_feature_std, dtype=np.float32), 1e-6)
                feature_shift = float(np.mean(np.abs(np.mean(Xb, axis=0) - baseline_feature_mean) / denom))

            drift_signal = (acc_weight * (1.0 - float(batch_acc))) + (conf_weight * low_conf_rate) + (feature_weight * feature_shift)
            drift_event = self.detector.update(metric_value=float(drift_signal), index=batch_end)
            adapted = False
            adapt_time = 0.0

            if drift_event.detected:
                drift_count += 1
                drift_indices.append(int(batch_end))
                min_samples = self.config["drift"].get("min_adapt_samples", 500)
                if len(recent_X) >= min_samples:
                    t_adapt = perf_counter()
                    split_idx = int(len(recent_X) * (1.0 - replay_fraction))
                    X_past = recent_X[:split_idx] if split_idx > 0 else None
                    y_past = recent_y[:split_idx] if split_idx > 0 else None
                    X_recent_win = recent_X[split_idx:]
                    y_recent_win = recent_y[split_idx:]

                    if X_past is not None and len(X_past) > 0 and len(X_past) > len(X_recent_win):
                        pick = rng.choice(len(X_past), size=len(X_recent_win), replace=False)
                        X_past = X_past[pick]
                        y_past = y_past[pick]

                    # split recent window into adapt-train and adapt-val for validation/rollback
                    val_frac = float(self.config.get("drift", {}).get("adapt_validation_fraction", 0.2))
                    if val_frac > 0 and len(X_recent_win) > 1:
                        vsize = max(1, int(len(X_recent_win) * val_frac))
                        train_end = max(1, len(X_recent_win) - vsize)
                        X_recent_train = X_recent_win[:train_end]
                        y_recent_train = y_recent_win[:train_end]
                        X_recent_val = X_recent_win[train_end:]
                        y_recent_val = y_recent_win[train_end:]
                    else:
                        X_recent_train = X_recent_win
                        y_recent_train = y_recent_win
                        X_recent_val = None
                        y_recent_val = None

                    # attach validation data to classifier for partial_adapt to use
                    if X_recent_val is not None and y_recent_val is not None:
                        setattr(self.classifier, "_adapt_val_X", X_recent_val)
                        setattr(self.classifier, "_adapt_val_y", y_recent_val)

                    self.classifier.partial_adapt(X_recent_train, y_recent_train, X_past=X_past, y_past=y_past)

                    # cleanup any attached validation attributes
                    if hasattr(self.classifier, "_adapt_val_X"):
                        try:
                            delattr(self.classifier, "_adapt_val_X")
                        except Exception:
                            pass
                    if hasattr(self.classifier, "_adapt_val_y"):
                        try:
                            delattr(self.classifier, "_adapt_val_y")
                        except Exception:
                            pass

                    # If adaptation accepted by classifier, persist models and bump metadata
                    accepted = bool(getattr(self.classifier, "_last_adapt_accepted", False))
                    if accepted:
                        # atomic save classifier
                        try:
                            save_model_atomic("models/classifier.joblib", self.classifier, method="joblib")
                        except Exception:
                            pass

                        # update autoencoder threshold from recent normals if present
                        metadata_path = Path("models/metadata.json")
                        try:
                            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                        except Exception:
                            metadata = {}

                        # update threshold if we have normal samples
                        normal_mask = recent_y == benign_label
                        if np.any(normal_mask):
                            try:
                                self.autoencoder.update_threshold(recent_X[normal_mask])
                                metadata["autoencoder_threshold"] = float(self.autoencoder.threshold)
                            except Exception:
                                pass

                        # bump version and timestamps
                        metadata["version"] = int(metadata.get("version", 0)) + 1
                        metadata["last_updated"] = datetime.utcnow().isoformat()
                        metadata["last_adapted_at"] = datetime.utcnow().isoformat()

                        try:
                            save_metadata_atomic("models/metadata.json", metadata)
                        except Exception:
                            pass
                    normal_mask = recent_y == benign_label
                    if np.any(normal_mask):
                        self.autoencoder.update_threshold(recent_X[normal_mask])
                    adapt_time = perf_counter() - t_adapt
                    total_adapt_time += adapt_time
                    adapted = True

            append_jsonl(
                events_path,
                {
                    "timestamp": datetime.utcnow().isoformat(),
                    "batch_start": int(batch_start),
                    "batch_end": int(batch_end),
                    "drift_detected": drift_event.detected,
                    "adapted": adapted,
                    "adapt_time_sec": adapt_time,
                    "batch_accuracy": float(batch_acc),
                    "batch_mean_confidence": mean_conf,
                    "batch_low_confidence_rate": low_conf_rate,
                    "batch_feature_shift": feature_shift,
                    "drift_signal": drift_signal,
                },
            )

            for i in range(len(Xb)):
                rows.append(
                    {
                        "timestamp": str(tsb.iloc[i]) if len(tsb) > i else str(batch_start + i),
                        "true_label": str(yb[i]),
                        "prediction": str(final_pred[i]),
                        "confidence": float(conf[i]),
                        "reconstruction_error": None if np.isnan(recon_err[i]) else float(recon_err[i]),
                        "drift_flag": bool(drift_event.detected),
                        "adapted": bool(adapted),
                        "cpu_percent": float(cpu),
                        "memory_mb": float(mem_mb),
                        "latency_ms_per_sample": float(batch_latency_ms),
                    }
                )

            y_pred_total.extend(final_pred)
            recent_X = np.vstack([recent_X, Xb])
            recent_y = np.concatenate([recent_y, yb])
            if len(recent_X) > window_size:
                recent_X = recent_X[-window_size:]
                recent_y = recent_y[-window_size:]

        out_df = pd.DataFrame(rows)
        write_csv(csv_path, out_df)
        y_eval = np.asarray(y[: len(y_pred_total)], dtype=object)
        y_pred_eval = np.asarray(y_pred_total, dtype=object)

        metrics = compute_classification_metrics(y_true=y_eval, y_pred=y_pred_eval)
        if known_labels:
            zero_day = compute_zero_day_metrics(
                y_true=y_eval,
                y_pred=y_pred_eval,
                known_labels=set(str(v) for v in known_labels),
                benign_label=benign_label,
            )
            metrics.update(zero_day)

        if drift_indices:
            first_drift = drift_indices[0]
            pre_mask = np.arange(len(y_eval)) < first_drift
            post_mask = ~pre_mask
            if np.any(pre_mask):
                metrics["pre_drift_accuracy"] = float(np.mean(y_pred_eval[pre_mask] == y_eval[pre_mask]))
            if np.any(post_mask):
                metrics["post_drift_accuracy"] = float(np.mean(y_pred_eval[post_mask] == y_eval[post_mask]))

            if known_labels:
                unseen_idx = np.where(~np.isin(np.asarray(y_eval, dtype=object).astype(str), list(set(str(v) for v in known_labels))))[0]
                if len(unseen_idx) > 0 and first_drift >= int(unseen_idx[0]):
                    metrics["drift_detection_latency_samples"] = int(first_drift - int(unseen_idx[0]))

        metrics.update(
            {
                "num_drifts": drift_count,
                "adaptation_time_total_sec": float(total_adapt_time),
                "avg_latency_ms_per_sample": float(out_df["latency_ms_per_sample"].mean()) if len(out_df) else 0.0,
                "avg_cpu_percent": float(out_df["cpu_percent"].mean()) if len(out_df) else 0.0,
                "avg_memory_mb": float(out_df["memory_mb"].mean()) if len(out_df) else 0.0,
                "samples": int(len(out_df)),
            }
        )

        return StreamArtifacts(records=out_df, metrics=metrics)
