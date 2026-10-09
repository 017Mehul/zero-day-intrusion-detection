"""FastAPI backend for SA-ZD-NIDS."""
from __future__ import annotations

import logging
import os
from datetime import datetime

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from deployment.api.security import SecurityMiddleware

from deployment.api.dependencies import get_model_manager
from deployment.api.endpoints import drift, health, predict

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger(__name__)

app = FastAPI(
    title="SA-ZD-NIDS API",
    description="Self-Adaptive Zero-Day Aware Network Intrusion Detection System",
    version="1.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

allowed_origins = [
    origin.strip()
    for origin in os.getenv(
        "SA_ZD_NIDS_ALLOWED_ORIGINS",
        "http://localhost:8501,http://127.0.0.1:8501",
    ).split(",")
    if origin.strip()
]
app.add_middleware(SecurityMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

app.include_router(predict.router, prefix="/api/v1", tags=["prediction"])
app.include_router(drift.router, prefix="/api/v1", tags=["drift"])
app.include_router(health.router, prefix="/api/v1", tags=["health"])


@app.on_event("startup")
async def startup_event():
    """Load model artifacts and fail fast when serving cannot start."""
    get_model_manager()
    logger.info("SA-ZD-NIDS models loaded successfully")


@app.get("/")
async def root():
    return {
        "message": "SA-ZD-NIDS API",
        "version": "1.1.0",
        "timestamp": datetime.utcnow().isoformat(),
        "health": "/api/v1/health",
        "docs": "/docs",
    }


if __name__ == "__main__":
    uvicorn.run(
        "deployment.api.main:app",
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", "8000")),
        reload=False,
        log_level="info",
    )
