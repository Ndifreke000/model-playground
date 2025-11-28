#!/usr/bin/env python3
"""
Create Hugging Face Space and deploy the misinformation detection API
"""
import os
import shutil
from pathlib import Path
from huggingface_hub import HfApi, create_repo

# Configuration
HF_TOKEN = "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"
SPACE_NAME = "yosemite000/misinformation-detector"
SOURCE_DIR = Path("/home/ndii/Documents/Mark/model-playground/api/ml_service")

print("🚀 Creating Hugging Face Space for Misinformation Detection API")
print("=" * 60)

# Initialize API
api = HfApi(token=HF_TOKEN)

# Step 1: Create Space
print("\n📦 Step 1: Creating Space...")
try:
    create_repo(
        repo_id=SPACE_NAME,
        repo_type="space",
        space_sdk="docker",
        token=HF_TOKEN,
        exist_ok=True
    )
    print(f"✅ Space created/verified: https://huggingface.co/spaces/{SPACE_NAME}")
except Exception as e:
    print(f"⚠️  Space may already exist: {e}")

# Step 2: Upload files
print("\n📤 Step 2: Uploading application files...")

files_to_upload = [
    "app.py",
    "config.py", 
    "model.py",
    "requirements.txt",
    "Dockerfile",
    ".dockerignore",
    "models/fake_news_model.pth",
    "models/tfidf_vectorizer.pkl"
]

for file_path in files_to_upload:
    source_file = SOURCE_DIR / file_path
    if source_file.exists():
        print(f"   Uploading {file_path}...")
        api.upload_file(
            path_or_fileobj=str(source_file),
            path_in_repo=file_path,
            repo_id=SPACE_NAME,
            repo_type="space",
            token=HF_TOKEN
        )
    else:
        print(f"   ⚠️  Skipping {file_path} (not found)")

# Step 3: Create README
print("\n📝 Step 3: Creating README.md...")
readme_content = """---
title: Misinformation Detector
emoji: 🔍
colorFrom: red
colorTo: pink
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

```bash
curl -X POST "https://yosemite000-misinformation-detector.hf.space/predict" \\
  -H "Content-Type: application/json" \\
  -d '{"text": "Breaking news article text here..."}'
```

**Response:**
```json
{
  "prediction": "fake",
  "confidence": 0.87,
  "raw_score": 0.74
}
```

## 🔧 Technical Details

- **Framework**: FastAPI
- **ML Model**: PyTorch Neural Network
- **Features**: TF-IDF (1000 features)
- **Training**: Kaggle Fake News Dataset

## 📊 Model Performance

- Accuracy: ~89%
- Training: 10 epochs
- Optimizer: Adam (lr=0.001)

## 📜 License

MIT License
"""

readme_path = "/tmp/hf_readme.md"
with open(readme_path, "w") as f:
    f.write(readme_content)

api.upload_file(
    path_or_fileobj=readme_path,
    path_in_repo="README.md",
    repo_id=SPACE_NAME,
    repo_type="space",
    token=HF_TOKEN
)

print("\n✅ Deployment Complete!")
print("=" * 60)
print(f"\n🎉 Your Space is available at:")
print(f"   https://huggingface.co/spaces/{SPACE_NAME}")
print(f"\n📡 API will be live at (after build ~2-5 minutes):")
print(f"   https://yosemite000-misinformation-detector.hf.space")
print(f"\n⏱️  Monitor build progress at:")
print(f"   https://huggingface.co/spaces/{SPACE_NAME}/logs")
print()
