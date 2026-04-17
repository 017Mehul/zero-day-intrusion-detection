"""Prediction endpoints for SA-ZD-NIDS API."""
from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from fastapi.responses import JSONResponse
import time
import numpy as np
import psutil
from datetime import datetime
from typing import List, Dict, Any

from deployment.api.models import (
    PredictionRequest, 
    PredictionResponse, 
    BatchPredictionRequest, 
    BatchPredictionResponse,
    ErrorResponse
)
from deployment.api.dependencies import get_model_manager, ModelManager

router = APIRouter()

@router.post("/predict", response_model=PredictionResponse)
async def predict_single(
    request: PredictionRequest,
    model_manager: ModelManager = Depends(get_model_manager)
):
    """Single prediction endpoint."""
    try:
        start_time = time.time()
        
        # Convert features to numpy array
        features = np.array(request.features, dtype=np.float32)
        
        # Validate feature dimensions
        if features.shape[0] != model_manager.expected_features:
            raise HTTPException(
                status_code=400,
                detail=f"Expected {model_manager.expected_features} features, got {features.shape[0]}"
            )
        
        # Make prediction
        prediction, confidence, is_zero_day, reconstruction_error = model_manager.predict(features)
        
        processing_time = (time.time() - start_time) * 1000  # Convert to milliseconds
        
        return PredictionResponse(
            prediction=prediction,
            confidence=confidence,
            is_zero_day=is_zero_day,
            reconstruction_error=reconstruction_error,
            processing_time_ms=processing_time,
            timestamp=datetime.utcnow().isoformat()
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )

@router.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(
    request: BatchPredictionRequest,
    model_manager: ModelManager = Depends(get_model_manager),
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    """Batch prediction endpoint."""
    try:
        start_time = time.time()
        
        # Convert batch to numpy array
        batch_features = np.array(request.batch_features, dtype=np.float32)
        
        # Validate feature dimensions
        if batch_features.shape[1] != model_manager.expected_features:
            raise HTTPException(
                status_code=400,
                detail=f"Expected {model_manager.expected_features} features per sample, got {batch_features.shape[1]}"
            )
        
        # Make batch predictions
        results = model_manager.predict_batch(batch_features)
        
        processing_time = (time.time() - start_time) * 1000
        
        # Log batch processing in background
        background_tasks.add_task(
            log_batch_processing,
            batch_size=len(request.batch_features),
            processing_time=processing_time
        )
        
        return BatchPredictionResponse(
            predictions=results["predictions"],
            confidences=results["confidences"],
            zero_day_flags=results["zero_day_flags"],
            reconstruction_errors=results["reconstruction_errors"],
            processing_time_ms=processing_time,
            batch_size=len(request.batch_features),
            timestamp=datetime.utcnow().isoformat()
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Batch prediction failed: {str(e)}"
        )

@router.post("/predict/stream")
async def predict_stream(
    request: PredictionRequest,
    model_manager: ModelManager = Depends(get_model_manager)
):
    """Streaming prediction endpoint (for real-time processing)."""
    try:
        start_time = time.time()
        
        features = np.array(request.features, dtype=np.float32)
        
        # Validate features
        if features.shape[0] != model_manager.expected_features:
            raise HTTPException(
                status_code=400,
                detail=f"Expected {model_manager.expected_features} features, got {features.shape[0]}"
            )
        
        # Make prediction
        prediction, confidence, is_zero_day, reconstruction_error = model_manager.predict(features)
        
        # Check for drift
        drift_info = model_manager.check_drift(confidence, prediction)
        
        processing_time = (time.time() - start_time) * 1000
        
        response = {
            "prediction": prediction,
            "confidence": confidence,
            "is_zero_day": is_zero_day,
            "reconstruction_error": reconstruction_error,
            "processing_time_ms": processing_time,
            "timestamp": datetime.utcnow().isoformat(),
            "drift_detected": drift_info["drift_detected"],
            "drift_score": drift_info["drift_score"]
        }
        
        # If drift detected, trigger adaptation in background
        if drift_info["drift_detected"]:
            background_tasks = BackgroundTasks()
            background_tasks.add_task(model_manager.trigger_adaptation)
            response["adaptation_triggered"] = True
        
        return JSONResponse(content=response)
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Stream prediction failed: {str(e)}"
        )

async def log_batch_processing(batch_size: int, processing_time: float):
    """Log batch processing metrics (background task)."""
    try:
        # This could log to a database, file, or monitoring system
        print(f"Batch processed: {batch_size} samples in {processing_time:.2f}ms")
    except Exception as e:
        print(f"Failed to log batch processing: {e}")

@router.get("/predict/stats")
async def get_prediction_stats(model_manager: ModelManager = Depends(get_model_manager)):
    """Get prediction statistics."""
    try:
        stats = model_manager.get_prediction_stats()
        return {
            "total_predictions": stats["total_predictions"],
            "zero_day_detections": stats["zero_day_detections"],
            "avg_confidence": stats["avg_confidence"],
            "avg_latency_ms": stats["avg_latency_ms"],
            "model_version": stats["model_version"],
            "uptime_hours": stats["uptime_hours"],
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get prediction stats: {str(e)}"
        )
