# ML Integration Setup Guide

## Overview

The fake news detection model from `Misinformation for fake.ipynb` has been integrated into the application with a Python FastAPI backend and React frontend.

## Architecture

```
┌─────────────────┐         ┌──────────────────┐         ┌───────────────┐
│  React Frontend │ ──HTTP──▶│  FastAPI Backend │ ──────▶│  PyTorch Model│
│  (TypeScript)   │  POST   │    (Python)      │  Loads │   + TF-IDF    │
└─────────────────┘         └──────────────────┘         └───────────────┘
        │                            │
        ▼                            ▼
  shadcn/ui Components        Model Weights (.pth)
  API Client                  Vectorizer (.pkl)
```

## Quick Start

### 1. Set Up Python Backend

```bash
cd api/ml_service

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download dataset (required for training)
# Visit: https://www.kaggle.com/stevenpeutz/misinformation-fake-news-text-dataset-79k
# Place files in api/ml_service/data/:
#   - DataSet_Misinfo_TRUE.csv
#   - DataSet_Misinfo_FAKE.csv

# Train the model
python train.py

# Start the API server
python app.py
```

### 2. Configure Frontend

The `.env` file has been updated with:
```bash
VITE_ML_API_URL="http://localhost:8000"
```

### 3. Access the Feature

Visit `/misinformation` route in your application to use the detector.

## File Structure

```
model-playground/
├── api/ml_service/              # Python ML backend
│   ├── app.py                   # FastAPI application
│   ├── model.py                 # PyTorch model classes
│   ├── train.py                 # Training script
│   ├── config.py                # Configuration
│   ├── requirements.txt         # Python dependencies
│   ├── models/                  # Trained model files (generated)
│   │   ├── fake_news_model.pth
│   │   └── tfidf_vectorizer.pkl
│   └── data/                    # Training data (you provide)
│       ├── DataSet_Misinfo_TRUE.csv
│       └── DataSet_Misinfo_FAKE.csv
│
└── src/
    ├── lib/api/
    │   └── misinformation.ts    # TypeScript API client
    └── components/misinformation/
        ├── MisinformationDetector.tsx  # Main UI component
        └── index.ts
```

## API Endpoints

### Health Check
```bash
GET http://localhost:8000/health
```

### Single Prediction
```bash
POST http://localhost:8000/predict
Content-Type: application/json

{
  "text": "Article text here..."
}

# Response:
{
  "prediction": "fake",
  "confidence": 0.87,
  "raw_score": 0.13
}
```

### Batch Prediction
```bash
POST http://localhost:8000/batch_predict
Content-Type: application/json

{
  "texts": ["Article 1...", "Article 2..."]
}
```

## Model Details

- **Accuracy**: ~67.5% on test set
- **Architecture**: 2-layer neural network (1000 → 64 → 1)
- **Input**: TF-IDF vectors (1000 features)
- **Training**: 10 epochs, Adam optimizer
- **Dataset**: 79K articles (fake + real news)

## Development Workflow

### Testing Backend
```bash
cd api/ml_service

# Test health endpoint
curl http://localhost:8000/health

# Test prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Scientists announce breakthrough..."}'
```

### Testing Frontend
1. Ensure backend is running on port 8000
2. Start frontend dev server: `npm run dev`
3. Navigate to `/misinformation`
4. Enter text and click "Analyze Text"

## Deployment

### Option 1: Docker

Create `api/ml_service/Dockerfile`:
```dockerfile
FROM python:3.12-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
```

Build and run:
```bash
docker build -t ml-service api/ml_service
docker run -p 8000:8000 ml-service
```

### Option 2: Serverless (AWS Lambda)

Use `mangum` adapter for FastAPI:
```bash
pip install mangum
```

Add to `app.py`:
```python
from mangum import Mangum
handler = Mangum(app)
```

### Option 3: Cloud Run (Google Cloud)

```bash
gcloud run deploy ml-service \
  --source api/ml_service \
  --region us-central1 \
  --allow-unauthenticated
```

## Troubleshooting

### Model Not Found
- Ensure you've run `python train.py`
- Check `api/ml_service/models/` directory exists with .pth and .pkl files

### CORS Errors
- Update `CORS_ORIGINS` in `config.py`
- Or set environment variable: `export CORS_ORIGINS="http://localhost:5173"`

### Connection Refused
- Backend not running: Start with `python app.py`
- Wrong port: Check `.env` has `VITE_ML_API_URL="http://localhost:8000"`

### Low Accuracy
The model achieves ~67.5% accuracy. To improve:
- Use more training data
- Try transfer learning (BERT, RoBERTa)
- Tune hyperparameters
- Add more features (metadata, source credibility)

## Next Steps

1. **Improve Model**: Train with more data or use pre-trained transformers
2. **Add Features**: 
   - Source credibility checking
   - Claim verification
   - Fact-checking links
3. **Monitoring**: Add logging and metrics (Sentry, CloudWatch)
4. **Caching**: Cache predictions for identical texts
5. **Rate Limiting**: Prevent abuse with rate limits

## Resources

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [PyTorch Tutorials](https://pytorch.org/tutorials/)
- [Fact-Checking Best Practices](https://www.poynter.org/fact-checking/)
