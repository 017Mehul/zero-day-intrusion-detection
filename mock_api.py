"""Mock API server for SA-ZD-NIDS GUI demonstration."""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import random
import time
from datetime import datetime
import uvicorn

app = FastAPI(title="SA-ZD-NIDS Mock API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class PredictionRequest(BaseModel):
    features: list[float]

class BatchPredictionRequest(BaseModel):
    batch_features: list[list[float]]

@app.get("/")
async def root():
    return {"message": "SA-ZD-NIDS Mock API", "version": "1.0.0"}

@app.get("/api/v1/health")
async def health():
    return {
        "status": "healthy",
        "system_metrics": {
            "cpu_usage": random.uniform(20, 80),
            "memory_usage_mb": random.uniform(100, 500),
            "avg_latency_ms": random.uniform(0.5, 2.0),
            "requests_per_minute": random.uniform(10, 100),
            "model_version": "1.0.0",
            "uptime_seconds": 3600
        },
        "drift_metrics": {
            "drift_detected": random.choice([True, False]),
            "drift_score": random.uniform(0.1, 0.8),
            "num_drifts_total": random.randint(0, 10)
        }
    }

@app.post("/api/v1/predict")
async def predict(request: PredictionRequest):
    start_time = time.time()
    
    # Mock prediction logic
    features = request.features
    if len(features) < 10:
        raise HTTPException(status_code=400, detail="Insufficient features")
    
    # Generate random predictions
    predictions = ["BENIGN", "DOS", "PROBE", "R2L", "U2R", "ZERO_DAY"]
    weights = [0.7, 0.1, 0.05, 0.05, 0.05, 0.05]  # Mostly benign
    
    prediction = random.choices(predictions, weights=weights)[0]
    confidence = random.uniform(0.8, 0.99)
    is_zero_day = prediction == "ZERO_DAY"
    reconstruction_error = random.uniform(0.05, 0.9) if is_zero_day else random.uniform(0.01, 0.3)
    
    processing_time = (time.time() - start_time) * 1000  # Convert to ms
    
    return {
        "prediction": prediction,
        "confidence": confidence,
        "is_zero_day": is_zero_day,
        "reconstruction_error": reconstruction_error,
        "processing_time_ms": processing_time
    }

@app.post("/api/v1/predict/batch")
async def predict_batch(request: BatchPredictionRequest):
    start_time = time.time()
    
    batch_features = request.batch_features
    results = []
    
    for features in batch_features:
        if len(features) < 10:
            continue
            
        # Generate random predictions
        predictions = ["BENIGN", "DOS", "PROBE", "R2L", "U2R", "ZERO_DAY"]
        weights = [0.7, 0.1, 0.05, 0.05, 0.05, 0.05]
        
        prediction = random.choices(predictions, weights=weights)[0]
        confidence = random.uniform(0.8, 0.99)
        is_zero_day = prediction == "ZERO_DAY"
        reconstruction_error = random.uniform(0.05, 0.9) if is_zero_day else random.uniform(0.01, 0.3)
        
        results.append({
            "prediction": prediction,
            "confidence": confidence,
            "is_zero_day": is_zero_day,
            "reconstruction_error": reconstruction_error
        })
    
    processing_time = (time.time() - start_time) * 1000
    
    return {
        "predictions": [r["prediction"] for r in results],
        "confidences": [r["confidence"] for r in results],
        "zero_day_flags": [r["is_zero_day"] for r in results],
        "reconstruction_errors": [r["reconstruction_error"] for r in results],
        "processing_time_ms": processing_time
    }

@app.get("/api/v1/predict/stats")
async def get_stats():
    return {
        "total_predictions": random.randint(1000, 10000),
        "zero_day_detections": random.randint(50, 200),
        "avg_confidence": random.uniform(0.85, 0.95),
        "avg_processing_time_ms": random.uniform(0.5, 2.0)
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
