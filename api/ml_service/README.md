# Misinformation Detection ML Service

This service provides an API for detecting fake news using a PyTorch-based machine learning model extracted from the Jupyter notebook.

## Quick Start

### 1. Install Dependencies

```bash
cd api/ml_service
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Prepare Data

The dataset will be **automatically downloaded** using kagglehub when you run training:

```bash
python train.py  # Downloads dataset automatically!
```

**Alternative: Manual download** (optional)
```bash
python download_data.py  # Download separately first
```

**Kaggle Credentials Required:**
- Create account at [kaggle.com](https://www.kaggle.com)
- Go to Account → API → Create New Token
- Save `kaggle.json` to `~/.kaggle/kaggle.json`
- Set permissions: `chmod 600 ~/.kaggle/kaggle.json`

More info: [Kaggle API Docs](https://www.kaggle.com/docs/api)

### 3. Train the Model

```bash
python train.py
```

This will:
- Load and preprocess the dataset
- Train the model for 10 epochs
- Save model weights to `models/fake_news_model.pth`
- Save vectorizer to `models/tfidf_vectorizer.pkl`
- Generate training plots

### 4. Start the API Server

```bash
python app.py
# Or using uvicorn directly:
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`

## API Endpoints

### Health Check
```bash
GET /health
```

### Single Prediction
```bash
POST /predict
Content-Type: application/json

{
  "text": "Your news article text here..."
}
```

Response:
```json
{
  "prediction": "fake",
  "confidence": 0.87,
  "raw_score": 0.13
}
```

### Batch Prediction
```bash
POST /batch_predict
Content-Type: application/json

{
  "texts": [
    "First article...",
    "Second article..."
  ]
}
```

## Testing

Test the API with curl:

```bash
# Health check
curl http://localhost:8000/health

# Single prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Breaking news: Scientists discover..."}'
```

## Model Details

- **Architecture**: 2-layer neural network (1000 → 64 → 1)
- **Input**: TF-IDF vectors (max 1000 features)
- **Output**: Binary classification (fake/real)
- **Accuracy**: ~67.5% on test set
- **Framework**: PyTorch 2.9.1

## Environment Variables

- `API_HOST`: Server host (default: 0.0.0.0)
- `API_PORT`: Server port (default: 8000)
- `CORS_ORIGINS`: Allowed origins (default: localhost:5173,3000)

## Troubleshooting

**Model not found error:**
- Ensure you've run `python train.py` first
- Check that `models/` directory contains .pth and .pkl files

**CUDA errors:**
- The model automatically falls back to CPU if CUDA is unavailable
- Check with: `python -c "import torch; print(torch.cuda.is_available())"`

**Import errors:**
- Activate virtual environment: `source venv/bin/activate`
- Reinstall dependencies: `pip install -r requirements.txt`
