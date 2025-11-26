"""Configuration for ML Service"""
import os
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).parent
MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "fake_news_model.pth"
VECTORIZER_PATH = MODEL_DIR / "tfidf_vectorizer.pkl"

# Ensure model directory exists
MODEL_DIR.mkdir(exist_ok=True)

# API Configuration
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))

# CORS
CORS_ORIGINS = os.getenv(
    "CORS_ORIGINS",
    "http://localhost:5173,http://localhost:3000,http://127.0.0.1:5173,http://localhost:8080"
).split(",")

# Model Hyperparameters (from notebook)
MAX_FEATURES = 1000
BATCH_SIZE = 32
LEARNING_RATE = 0.001
NUM_EPOCHS = 10

# Validation
MAX_TEXT_LENGTH = 10000  # Maximum characters for input text
MIN_TEXT_LENGTH = 10     # Minimum characters for input text
