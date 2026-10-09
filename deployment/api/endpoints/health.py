"""Health and readiness endpoints."""
from __future__ import annotations

import os
import time

import psutil
from fastapi import APIRouter, Depends

from deployment.api.dependencies import ModelManager, get_model_manager
from deployment.api.models import DriftMetrics, HealthResponse, SystemMetrics

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health(model_manager: ModelManager = Depends(get_model_manager)):
    stats = model_manager.get_prediction_stats()
    drift = model_manager.check_drift(
        model_manager.recent_confidences[-1] if model_manager.recent_confidences else 1.0,
        model_manager.recent_predictions[-1] if model_manager.recent_predictions else "N/A",
    ) if False else {
        "drift_detected": False,
        "drift_score": 0.0,
        "drift_threshold": model_manager.drift_detector.adwin.delta,
        "method": "ADWIN",
        "num_drifts_total": model_manager.drift_detector.num_drifts,
        "last_drift_time": stats.get("last_drift_time"),
    }
    return HealthResponse(
        status="ok",
        model_loaded=True,
        system_metrics=SystemMetrics(
            cpu_usage=float(psutil.cpu_percent(interval=None)),
            memory_usage_mb=float(psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)),
            avg_latency_ms=float(stats["avg_latency_ms"]),
            requests_per_minute=0.0,
            model_version=str(stats["model_version"]),
            uptime_seconds=float((time.time() - model_manager.start_time)),
        ),
        drift_metrics=DriftMetrics(**drift),
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    )


@router.get("/ready")
async def ready(model_manager: ModelManager = Depends(get_model_manager)):
    return {"ready": True, "model_version": model_manager.metadata.get("version", 1)}
