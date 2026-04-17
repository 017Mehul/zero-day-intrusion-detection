"""FastAPI backend for SA-ZD-NIDS production deployment."""
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import logging
from datetime import datetime

from deployment.api.endpoints import predict, drift, health
from deployment.api.dependencies import get_model_manager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="SA-ZD-NIDS API",
    description="Self-Adaptive Zero-Day Aware Network Intrusion Detection System",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify actual origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(predict.router, prefix="/api/v1", tags=["prediction"])
app.include_router(drift.router, prefix="/api/v1", tags=["drift"])
app.include_router(health.router, prefix="/api/v1", tags=["health"])

# Global model manager (will be initialized on startup)
model_manager = None

@app.on_event("startup")
async def startup_event():
    """Initialize models and dependencies on startup."""
    global model_manager
    try:
        model_manager = get_model_manager()
        logger.info("Models loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load models: {e}")
        raise

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("Shutting down SA-ZD-NIDS API")

@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "SA-ZD-NIDS API",
        "version": "1.0.0",
        "timestamp": datetime.utcnow().isoformat(),
        "docs": "/docs"
    }

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
