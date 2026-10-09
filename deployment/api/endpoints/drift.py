"""Drift monitoring endpoints."""
from __future__ import annotations

from fastapi import APIRouter, Depends

from deployment.api.dependencies import ModelManager, get_model_manager

router = APIRouter()


@router.get("/drift")
async def drift_status(model_manager: ModelManager = Depends(get_model_manager)):
    stats = model_manager.get_prediction_stats()
    return {
        "method": "ADWIN",
        "drift_detected": False,
        "drift_threshold": model_manager.drift_detector.adwin.delta,
        "num_drifts_total": model_manager.drift_detector.num_drifts,
        "last_drift_time": stats.get("last_drift_time"),
        "last_adaptation": stats.get("last_adaptation"),
        "model_version": stats.get("model_version"),
    }
