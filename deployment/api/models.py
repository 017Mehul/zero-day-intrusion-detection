"""Pydantic models for API request/response schemas."""
from pydantic import BaseModel, Field, validator
from typing import List, Optional, Union, Dict, Any
from datetime import datetime
import numpy as np

class NetworkFeature(BaseModel):
    """Single network traffic feature."""
    feature_name: str
    value: float

class PredictionRequest(BaseModel):
    """Request for network traffic prediction."""
    features: List[float] = Field(..., description="Feature values for prediction")
    feature_names: Optional[List[str]] = Field(None, description="Optional feature names")
    timestamp: Optional[str] = Field(None, description="Timestamp of the traffic")
    
    @validator('features')
    def validate_features(cls, v):
        if len(v) == 0:
            raise ValueError("Features list cannot be empty")
        return v

class PredictionResponse(BaseModel):
    """Response from prediction endpoint."""
    prediction: str = Field(..., description="Predicted class")
    confidence: float = Field(..., description="Prediction confidence")
    is_zero_day: bool = Field(..., description="Whether this is a zero-day attack")
    reconstruction_error: Optional[float] = Field(None, description="Reconstruction error for zero-day detection")
    processing_time_ms: float = Field(..., description="Time taken for prediction in milliseconds")
    timestamp: str = Field(..., description="Response timestamp")

class BatchPredictionRequest(BaseModel):
    """Request for batch prediction."""
    batch_features: List[List[float]] = Field(..., description="Batch of feature vectors")
    feature_names: Optional[List[str]] = Field(None, description="Optional feature names")
    
    @validator('batch_features')
    def validate_batch_features(cls, v):
        if len(v) == 0:
            raise ValueError("Batch features cannot be empty")
        if len(v) > 1000:
            raise ValueError("Batch size cannot exceed 1000")
        return v

class BatchPredictionResponse(BaseModel):
    """Response from batch prediction endpoint."""
    predictions: List[str] = Field(..., description="List of predictions")
    confidences: List[float] = Field(..., description="List of confidence scores")
    zero_day_flags: List[bool] = Field(..., description="List of zero-day flags")
    reconstruction_errors: List[Optional[float]] = Field(..., description="List of reconstruction errors")
    processing_time_ms: float = Field(..., description="Total processing time in milliseconds")
    batch_size: int = Field(..., description="Size of the processed batch")
    timestamp: str = Field(..., description="Response timestamp")

class DriftMetrics(BaseModel):
    """Drift detection metrics."""
    drift_detected: bool = Field(..., description="Whether drift was detected")
    drift_score: float = Field(..., description="Current drift score")
    drift_threshold: float = Field(..., description="Drift detection threshold")
    method: str = Field(..., description="Drift detection method used")
    num_drifts_total: int = Field(..., description="Total number of drifts detected")
    last_drift_time: Optional[str] = Field(None, description="Time of last drift detection")

class SystemMetrics(BaseModel):
    """System performance metrics."""
    cpu_usage: float = Field(..., description="CPU usage percentage")
    memory_usage_mb: float = Field(..., description="Memory usage in MB")
    avg_latency_ms: float = Field(..., description="Average prediction latency in milliseconds")
    requests_per_minute: float = Field(..., description="Requests per minute")
    model_version: str = Field(..., description="Current model version")
    uptime_seconds: float = Field(..., description="System uptime in seconds")

class HealthResponse(BaseModel):
    """Health check response."""
    status: str = Field(..., description="System status")
    model_loaded: bool = Field(..., description="Whether models are loaded")
    system_metrics: SystemMetrics = Field(..., description="System performance metrics")
    drift_metrics: DriftMetrics = Field(..., description="Drift detection metrics")
    timestamp: str = Field(..., description="Health check timestamp")

class ErrorResponse(BaseModel):
    """Error response model."""
    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Additional error details")
    timestamp: str = Field(..., description="Error timestamp")
