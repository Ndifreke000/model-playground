"""
FastAPI application for Misinformation Detection
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from typing import List
import logging

from model import MisinformationDetector
import config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Misinformation Detection API",
    description="API for detecting fake news using ML",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize model (singleton pattern)
detector = None


def get_detector() -> MisinformationDetector:
    """Get or create detector instance"""
    global detector
    if detector is None:
        if not config.MODEL_PATH.exists() or not config.VECTORIZER_PATH.exists():
            logger.error("Model files not found. Please train the model first.")
            raise RuntimeError(
                "Model not trained. Run 'python train.py' to train the model."
            )
        
        detector = MisinformationDetector(
            model_path=str(config.MODEL_PATH),
            vectorizer_path=str(config.VECTORIZER_PATH)
        )
        logger.info("Model loaded successfully")
    
    return detector


# Request/Response Models
class PredictRequest(BaseModel):
    text: str = Field(..., description="Text to analyze for misinformation")
    
    @validator('text')
    def validate_text_length(cls, v):
        if len(v) < config.MIN_TEXT_LENGTH:
            raise ValueError(f"Text too short. Minimum {config.MIN_TEXT_LENGTH} characters.")
        if len(v) > config.MAX_TEXT_LENGTH:
            raise ValueError(f"Text too long. Maximum {config.MAX_TEXT_LENGTH} characters.")
        return v


class PredictResponse(BaseModel):
    prediction: str = Field(..., description="Prediction: 'fake' or 'real'")
    confidence: float = Field(..., description="Confidence score (0-1)")
    raw_score: float = Field(..., description="Raw model output")


class BatchPredictRequest(BaseModel):
    texts: List[str] = Field(..., description="List of texts to analyze")
    
    @validator('texts')
    def validate_batch_size(cls, v):
        if len(v) > 100:
            raise ValueError("Maximum 100 texts per batch")
        return v


class BatchPredictResponse(BaseModel):
    results: List[PredictResponse]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    device: str


# API Endpoints
@app.get("/", tags=["General"])
def root():
    """Root endpoint"""
    return {
        "message": "Misinformation Detection API",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse, tags=["General"])
def health_check():
    """Health check endpoint"""
    try:
        det = get_detector()
        return {
            "status": "healthy",
            "model_loaded": det.model is not None,
            "device": str(det.device)
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "unhealthy",
            "model_loaded": False,
            "device": "unknown"
        }


@app.post("/predict", response_model=PredictResponse, tags=["Prediction"])
def predict(request: PredictRequest):
    """Predict if a single text is fake news"""
    try:
        det = get_detector()
        result = det.predict(request.text)
        logger.info(f"Prediction: {result['prediction']} (confidence: {result['confidence']:.2f})")
        return result
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/batch_predict", response_model=BatchPredictResponse, tags=["Prediction"])
def batch_predict(request: BatchPredictRequest):
    """Predict multiple texts at once"""
    try:
        det = get_detector()
        results = det.batch_predict(request.texts)
        logger.info(f"Batch prediction completed for {len(results)} texts")
        return {"results": results}
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app:app",
        host=config.API_HOST,
        port=config.API_PORT,
        reload=True,
        log_level="info"
    )
