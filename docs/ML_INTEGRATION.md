# ML Integration Setup Guide

## Overview

The fake news detection model has been integrated into the application with a Python FastAPI backend deployed to **Hugging Face Spaces** and a React frontend.

## Architecture

```
┌─────────────────┐         ┌──────────────────────┐         ┌───────────────┐
│  React Frontend │ ──HTTP──▶│  FastAPI Backend     │ ──────▶│  PyTorch Model│
│  (TypeScript)   │  POST   │  (Hugging Face)      │  Loads │   + TF-IDF    │
│   on Vercel     │         │   Docker Container   │        └───────────────┘
└─────────────────┘         └──────────────────────┘
        │                            │
        ▼                            ▼
  shadcn/ui Components         Model Weights (.pth)
  API Client                   Vectorizer (.pkl)

Production URL: https://yosemite000-misinformation-detector.hf.space
```

## Quick Start

### 1. Frontend Setup

The `.env` file should contain:
```bash
VITE_ML_API_URL="https://yosemite000-misinformation-detector.hf.space"
VITE_SUPABASE_URL="your_supabase_url"
VITE_SUPABASE_PUBLISHABLE_KEY="your_supabase_key"
```

### 2. Access the Feature

Visit `/misinformation` route in your application to use the detector. The API calls will automatically route to the Hugging Face Spaces deployment.

### 3. Local ML Backend Setup (Optional - For Development Only)

> **Note:** The ML API is already deployed to Hugging Face Spaces. You only need this if you're developing new ML features.

```bash
cd api/ml_service

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\\Scripts\\activate

# Install dependencies
pip install -r requirements.txt

# Download dataset (required for training)
# Visit: https://www.kaggle.com/stevenpeutz/misinformation-fake-news-text-dataset-79k
# Place files in api/ml_service/data/:
#   - DataSet_Misinfo_TRUE.csv
#   - DataSet_Misinfo_FAKE.csv

# Train the model
python train.py

# Start the API server (for local testing)
python app.py
```

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

### Production Endpoints

Base URL: `https://yosemite000-misinformation-detector.hf.space`

### Health Check
```bash
GET https://yosemite000-misinformation-detector.hf.space/health
```

### Single Prediction
```bash
POST https://yosemite000-misinformation-detector.hf.space/predict
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
POST https://yosemite000-misinformation-detector.hf.space/batch_predict
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

> [!NOTE]
> **The ML service is already deployed to Hugging Face Spaces!**
> 
> URL: https://yosemite000-misinformation-detector.hf.space

For detailed deployment instructions and redeployment steps, see [DEPLOYMENT.md](../DEPLOYMENT.md).

### Quick Reference

The ML service is containerized using Docker and deployed to Hugging Face Spaces. The deployment includes:
- FastAPI backend
- PyTorch model weights
- TF-IDF vectorizer
- CORS configuration for frontend access

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
