#!/bin/bash
# Quick start script for ML service

echo "==================================="
echo "ML Service Quick Start"
echo "==================================="

cd "$(dirname "$0")"

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install -q -r requirements.txt

# Check if model files exist
if [ ! -f "models/fake_news_model.pth" ] || [ ! -f "models/tfidf_vectorizer.pkl" ]; then
    echo ""
    echo "⚠️  Model files not found!"
    echo "Running training will automatically download the dataset via kagglehub"
    echo "Make sure you have Kaggle credentials configured (~/.kaggle/kaggle.json)"
    echo ""
    read -p "Run training now? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        python train.py
    else
        echo "Skipping training. Start server will fail without model files."
    fi
else
    echo "✓ Model files found!"
fi

# Start the server
echo "Starting FastAPI server..."
python app.py
