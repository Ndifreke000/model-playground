# Misinformation Detection System

A machine learning-powered platform for detecting fake news using a PyTorch neural network with TF-IDF text vectorization.

## 🚀 Live API

**ML API:** `https://yosemite000-misinformation-detector.hf.space`

## 🧠 Machine Learning Model

### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Neural Network                            │
│                                                              │
│   Input Layer        Hidden Layer        Output Layer        │
│   (1000 neurons)     (64 neurons)        (1 neuron)         │
│                                                              │
│   ┌─────────┐        ┌─────────┐        ┌─────────┐         │
│   │ TF-IDF  │───────▶│  ReLU   │───────▶│ Sigmoid │         │
│   │ Vector  │        │ + Dropout│        │         │         │
│   └─────────┘        └─────────┘        └─────────┘         │
│                                                              │
│   1000 features      64 hidden units     Binary output       │
│                      0.3 dropout         (fake/real)         │
└─────────────────────────────────────────────────────────────┘
```

### Model Details

| Specification | Value |
|---------------|-------|
| **Framework** | PyTorch 2.9.1 |
| **Architecture** | Feedforward Neural Network |
| **Input Features** | 1000 (TF-IDF) |
| **Hidden Layer** | 64 neurons, ReLU activation |
| **Dropout** | 0.3 |
| **Output** | Sigmoid (binary classification) |
| **Optimizer** | Adam (lr=0.001) |
| **Loss Function** | Binary Cross-Entropy |
| **Training Epochs** | 10 |

### Performance Metrics

| Metric | Value |
|--------|-------|
| **Test Accuracy** | 69.5% |
| **Precision (Real)** | 99.2% |
| **Recall (Fake)** | 66.6% |
| **Inference Speed** | <50ms |

### Text Preprocessing Pipeline

```
Raw Text → Lowercase → Remove Punctuation → TF-IDF Vectorization → 1000-dim Vector
```

1. **Text Cleaning**: Convert to lowercase, remove special characters
2. **TF-IDF Vectorization**: Transform text to numerical features (max 1000 features)
3. **Model Inference**: Feed vector through neural network
4. **Threshold**: Sigmoid output > 0.5 = Real, otherwise = Fake

## 🛠️ Tech Stack

### ML Backend (Python)

| Technology | Version | Purpose |
|------------|---------|---------|
| **Python** | 3.11 | Runtime |
| **PyTorch** | 2.9.1 | Deep learning framework |
| **FastAPI** | 0.115+ | REST API framework |
| **scikit-learn** | 1.6+ | TF-IDF vectorization |
| **Uvicorn** | 0.34+ | ASGI server |
| **NumPy** | 2.2+ | Numerical computing |
| **Pandas** | 2.2+ | Data manipulation |
| **Matplotlib** | 3.10+ | Training visualization |

### Frontend (TypeScript/React)

| Technology | Version | Purpose |
|------------|---------|---------|
| **React** | 18.3 | UI framework |
| **TypeScript** | 5.x | Type-safe JavaScript |
| **Vite** | 5.x | Build tool |
| **Tailwind CSS** | 3.x | Styling |
| **shadcn/ui** | Latest | Component library |
| **TanStack Query** | 5.x | Server state management |
| **React Router** | 6.x | Client-side routing |

### Backend Services

| Technology | Purpose |
|------------|---------|
| **Supabase** | Database & Authentication |
| **PostgreSQL** | Data persistence |
| **Deno** | Edge functions runtime |

### Deployment

| Service | Purpose |
|---------|---------|
| **Hugging Face Spaces** | ML API hosting |
| **Docker** | Containerization |

## 📁 Project Structure

```
├── api/ml_service/              # ML Backend
│   ├── app.py                   # FastAPI server
│   ├── model.py                 # Neural network definition
│   ├── train.py                 # Training script
│   ├── config.py                # Configuration
│   ├── download_data.py         # Dataset downloader
│   ├── requirements.txt         # Python dependencies
│   ├── Dockerfile               # Container config
│   └── README.md                # ML service docs
│
├── models/                      # Trained artifacts
│   ├── fake_news_model.pth      # PyTorch weights
│   ├── tfidf_vectorizer.pkl     # Fitted TF-IDF
│   ├── confusion_matrix.png     # Evaluation plot
│   └── training_loss.png        # Training curve
│
├── src/                         # Frontend
│   ├── components/
│   │   ├── analysis/            # Analysis UI components
│   │   └── ui/                  # shadcn components
│   ├── pages/                   # Route pages
│   ├── lib/api/
│   │   └── misinformation.ts    # ML API client
│   ├── types/                   # TypeScript interfaces
│   └── integrations/supabase/   # Database client
│
├── supabase/
│   ├── functions/               # Edge functions
│   └── config.toml              # Supabase config
│
└── docs/                        # Documentation
    ├── ARCHITECTURE.md
    └── ML_INTEGRATION.md
```

## 🚀 Getting Started

### Prerequisites

- Node.js 18+ and npm
- For ML development only: Python 3.11+

### Frontend Development

```bash
# Install dependencies
npm install

# Configure environment variables
# Create .env file with:
# VITE_ML_API_URL=https://yosemite000-misinformation-detector.hf.space
# VITE_SUPABASE_URL=your_supabase_url
# VITE_SUPABASE_PUBLISHABLE_KEY=your_supabase_key

# Start dev server
npm run dev
```

App available at `http://localhost:8080`

### ML Backend Development (Optional)

> **Note:** The ML API is already deployed to Hugging Face Spaces. You only need to run this locally if you're developing new ML features.

```bash
cd api/ml_service

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\\Scripts\\activate

# Install dependencies
pip install -r requirements.txt

# Train the model (downloads dataset automatically)
python train.py

# Start API server (local testing only)
python app.py
```

Local API available at `http://localhost:8000`

## 🌐 Deployment

### Current Architecture

| Component | Platform | URL |
|-----------|----------|-----|
| **Frontend** | Vercel (Ready to deploy) | TBD |
| **ML API** | Hugging Face Spaces | https://yosemite000-misinformation-detector.hf.space |
| **Database** | Supabase | Configured |

### Deploy Frontend to Vercel

```bash
# Install Vercel CLI
npm i -g vercel

# Deploy to production
vercel --prod

# Set environment variable in Vercel dashboard:
# VITE_ML_API_URL=https://yosemite000-misinformation-detector.hf.space
```

The ML backend is already live on Hugging Face Spaces. See [DEPLOYMENT.md](./DEPLOYMENT.md) for details.

## 📡 API Endpoints

### Health Check
```bash
GET /health

Response:
{
  "status": "healthy",
  "model_loaded": true,
  "device": "cpu"
}
```

### Single Prediction
```bash
POST /predict
Content-Type: application/json

{
  "text": "News article text to analyze..."
}

Response:
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
  "texts": ["Article 1...", "Article 2..."]
}

Response:
{
  "results": [
    {"prediction": "fake", "confidence": 0.87, "raw_score": 0.13},
    {"prediction": "real", "confidence": 0.92, "raw_score": 0.92}
  ]
}
```

## 📊 Dataset

The model is trained on the **Fake and Real News Dataset** from Kaggle:
- ~44,000 articles total
- Binary labels: Fake (0) / Real (1)
- Features: Title + Text content

Dataset download requires Kaggle API credentials:
1. Create account at [kaggle.com](https://www.kaggle.com)
2. Go to Account → API → Create New Token
3. Save `kaggle.json` to `~/.kaggle/kaggle.json`

## 🐳 Docker Deployment

> **Note:** The ML service is already deployed to Hugging Face Spaces using Docker. This section is for reference only.

```bash
cd api/ml_service

# Build image
docker build -t misinformation-detector .

# Run container locally
docker run -p 8000:8000 misinformation-detector
```

## 🔧 Environment Variables

### ML Backend
| Variable | Default | Description |
|----------|---------|-------------|
| `API_HOST` | 0.0.0.0 | Server host |
| `API_PORT` | 8000 | Server port |
| `CORS_ORIGINS` | localhost:5173,3000 | Allowed origins |

### Frontend
| Variable | Description |
|----------|-------------|
| `VITE_ML_API_URL` | ML backend URL |
| `VITE_SUPABASE_URL` | Supabase project URL |
| `VITE_SUPABASE_PUBLISHABLE_KEY` | Supabase anon key |

## 📈 Training Visualization

The training script generates:
- `models/training_loss.png` - Loss curve over epochs
- `models/confusion_matrix.png` - Test set evaluation

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/improvement`)
3. Commit changes (`git commit -m 'Add improvement'`)
4. Push to branch (`git push origin feature/improvement`)
5. Open Pull Request

## 📄 License

Academic research project for misinformation detection.
