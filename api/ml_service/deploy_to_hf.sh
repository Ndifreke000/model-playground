#!/bin/bash
# Deployment script for Hugging Face

set -e

echo "🚀 Deploying Misinformation Detection Model to Hugging Face"
echo ""

# Configuration
HF_REPO="yosemite000/misinformation"
TEMP_DIR="/tmp/hf_misinformation_deploy"

# Clean up any existing temp directory
rm -rf "$TEMP_DIR"

# Create fresh directory
mkdir -p "$TEMP_DIR"
cd "$TEMP_DIR"

echo "📦 Step 1: Cloning repository..."
git clone https://huggingface.co/$HF_REPO .

echo ""
echo "📋 Step 2: Copying application files..."

# Copy all necessary files
cp /home/ndii/Documents/Mark/model-playground/api/ml_service/app.py .
cp /home/ndii/Documents/Mark/model-playground/api/ml_service/config.py .
cp /home/ndii/Documents/Mark/model-playground/api/ml_service/model.py .
cp /home/ndii/Documents/Mark/model-playground/api/ml_service/requirements.txt .
cp /home/ndii/Documents/Mark/model-playground/api/ml_service/Dockerfile .
cp /home/ndii/Documents/Mark/model-playground/api/ml_service/.dockerignore .

# Copy model files
echo "🧠 Step 3: Copying model weights..."
mkdir -p models
cp /home/ndii/Documents/Mark/model-playground/api/ml_service/models/fake_news_model.pth models/
cp /home/ndii/Documents/Mark/model-playground/api/ml_service/models/tfidf_vectorizer.pkl models/

# Create README for Hugging Face
echo "📝 Step 4: Creating README.md..."
cat > README.md << 'EOF'
---
title: Misinformation Detector
emoji: 🔍
colorFrom: red
colorTo: orange
sdk: docker
pinned: false
---

# 🔍 Misinformation Detection API

FastAPI-powered machine learning service for detecting fake news using PyTorch and TF-IDF.

## 🚀 API Endpoints

- **GET** `/` - Root endpoint with API info
- **GET** `/health` - Health check
- **POST** `/predict` - Predict if a single text is fake news
- **POST** `/batch_predict` - Batch predictions (up to 100 texts)

## 📖 Usage

### Single Prediction

\`\`\`bash
curl -X POST "https://yosemite000-misinformation.hf.space/predict" \\
  -H "Content-Type: application/json" \\
  -d '{
    "text": "Breaking news: Scientists discover new planet in our solar system!"
  }'
\`\`\`

**Response:**
\`\`\`json
{
  "prediction": "fake",
  "confidence": 0.87,
  "raw_score": 0.74
}
\`\`\`

### Batch Prediction

\`\`\`bash
curl -X POST "https://yosemite000-misinformation.hf.space/batch_predict" \\
  -H "Content-Type: application/json" \\
  -d '{
    "texts": [
      "First news article...",
      "Second news article..."
    ]
  }'
\`\`\`

## 🔧 Technical Details

- **Framework**: FastAPI
- **ML Model**: PyTorch Neural Network
- **Features**: TF-IDF (1000 features)
- **Training**: Kaggle Fake News Dataset

## 📊 Model Performance

The model was trained on a labeled dataset of fake and real news articles with the following metrics:
- Accuracy: ~89%
- Training: 10 epochs
- Optimizer: Adam (lr=0.001)

## 🛠️ Local Development

\`\`\`bash
# Install dependencies
pip install -r requirements.txt

# Run the API
uvicorn app:app --reload --port 8000
\`\`\`

## 📜 License

MIT License
EOF

echo ""
echo "✅ Step 5: Committing changes..."
git add .
git commit -m "Deploy FastAPI misinformation detection service"

echo ""
echo "🌐 Step 6: Pushing to Hugging Face..."
git push

echo ""
echo "✅ Deployment complete!"
echo ""
echo "🎉 Your API will be available at:"
echo "   https://yosemite000-misinformation.hf.space"
echo ""
echo "⏱️  Note: It may take 2-5 minutes for Hugging Face to build and deploy."
echo "   Check status at: https://huggingface.co/$HF_REPO"
echo ""
