"""Prediction endpoints for SA-ZD-NIDS."""
from __future__ import annotations

import time
from datetime import datetime

import numpy as np
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException

from deployment.api.dependencies import ModelManager, get_model_manager
from deployment.api.models import BatchPredictionRequest, BatchPredictionResponse, PredictionRequest, PredictionResponse

router = APIRouter()


@router.post("/predict", response_model=PredictionResponse)
async def predict_single(request: PredictionRequest, model_manager: ModelManager = Depends(get_model_manager)):
    start_time = time.perf_counter()
    try:
        features = np.asarray(request.features, dtype=np.float32)
        prediction, confidence, is_zero_day, reconstruction_error = model_manager.predict(features)
        return PredictionResponse(
            prediction=prediction,
            confidence=confidence,
            is_zero_day=is_zero_day,
            reconstruction_error=reconstruction_error,
            processing_time_ms=(time.perf_counter() - start_time) * 1000,
            timestamp=datetime.utcnow().isoformat(),
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}") from exc


@router.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(
    request: BatchPredictionRequest,
    background_tasks: BackgroundTasks,
    model_manager: ModelManager = Depends(get_model_manager),
):
    start_time = time.perf_counter()
    try:
        batch_features = np.asarray(request.batch_features, dtype=np.float32)
        if batch_features.ndim != 2 or batch_features.shape[0] == 0:
            raise HTTPException(status_code=400, detail="batch_features must be a non-empty 2D array")
        if batch_features.shape[0] > 1000:
            raise HTTPException(status_code=400, detail="Batch size cannot exceed 1000")
        results = model_manager.predict_batch(batch_features)
        processing_time = (time.perf_counter() - start_time) * 1000
        background_tasks.add_task(log_batch_processing, len(batch_features), processing_time)
        return BatchPredictionResponse(
            predictions=results["predictions"],
            confidences=results["confidences"],
            zero_day_flags=results["zero_day_flags"],
            reconstruction_errors=results["reconstruction_errors"],
            processing_time_ms=processing_time,
            batch_size=len(batch_features),
            timestamp=datetime.utcnow().isoformat(),
        )
    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {exc}") from exc


@router.post("/predict/stream")
async def predict_stream(
    request: PredictionRequest,
    background_tasks: BackgroundTasks,
    model_manager: ModelManager = Depends(get_model_manager),
):
    start_time = time.perf_counter()
    try:
        features = np.asarray(request.features, dtype=np.float32)
        prediction, confidence, is_zero_day, reconstruction_error = model_manager.predict(features)
        drift_info = model_manager.check_drift(confidence, prediction)
        adaptation_result = None
        if drift_info["drift_detected"]:
            background_tasks.add_task(model_manager.trigger_adaptation)
            adaptation_result = {"status": "scheduled"}

        return {
            "prediction": prediction,
            "confidence": confidence,
            "is_zero_day": is_zero_day,
            "reconstruction_error": reconstruction_error,
            "processing_time_ms": (time.perf_counter() - start_time) * 1000,
            "timestamp": datetime.utcnow().isoformat(),
            "drift_detected": drift_info["drift_detected"],
            "drift_score": drift_info["drift_score"],
            "drift_threshold": drift_info["drift_threshold"],
            "method": drift_info["method"],
            "num_drifts_total": drift_info["num_drifts_total"],
            "adaptation": adaptation_result,
        }
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Stream prediction failed: {exc}") from exc


async def log_batch_processing(batch_size: int, processing_time: float):
    print(f"Batch processed: {batch_size} samples in {processing_time:.2f}ms")


@router.get("/predict/stats")
async def get_prediction_stats(model_manager: ModelManager = Depends(get_model_manager)):
    try:
        return {**model_manager.get_prediction_stats(), "timestamp": datetime.utcnow().isoformat()}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to get prediction stats: {exc}") from exc
