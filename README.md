# Misinformation Detection System 🕵️‍♀️

> A hybrid intelligence platform combining local machine learning and AI reasoning to detect and analyze fake news.

![Project Status](https://img.shields.io/badge/status-active-success.svg)
![Python](https://img.shields.io/badge/python-3.12-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.5-ee4c2c.svg)
![React](https://img.shields.io/badge/react-18.0-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## 📖 Introduction

In an era of information overload, distinguishing fact from fiction is critical. This project implements a **Misinformation Detection System** that analyzes news articles using Natural Language Processing (NLP).

It features a **Hybrid Architecture**:
1.  **Local ML Model:** A fast, privacy-first PyTorch model that screens text for linguistic patterns of fake news.
2.  **Frontend Interface:** A modern React application for real-time analysis.
3.  **Extensible Design:** Built to integrate with LLMs (like Gemini) for deep reasoning.

## 🚀 Key Features

-   **Real-time Detection:** Instant classification of text as "Real" or "Fake".
-   **Confidence Scoring:** Returns a probability score to indicate certainty.
-   **Privacy First:** Inference runs locally; data doesn't need to leave your server.
-   **Auto-Training:** Automated pipeline to download datasets and retrain the model.
-   **Modern UI:** Clean, responsive interface built with shadcn/ui.

## 📊 Model Performance

The model was trained on the **Misinformation Fake News Text Dataset (79k articles)**.

| Metric | Score | Description |
|--------|-------|-------------|
| **Accuracy** | **69.5%** | Overall correctness on unseen test data. |
| **Precision (Real)** | **99.2%** | When it says "Real", it is almost always right. |
| **Recall (Fake)** | **66.6%** | Identifies ~2/3rds of all fake news samples. |
| **Inference Time** | **<50ms** | Ultra-low latency on standard CPU. |

*Note: The model is optimized for high precision on real news to avoid false alarms, making it a conservative screening tool.*

## 🛠️ Tech Stack

### Backend (ML Service)
-   **Python 3.12**
-   **FastAPI**: High-performance API framework.
-   **PyTorch**: Deep learning framework (CPU-optimized).
-   **Scikit-learn**: TF-IDF vectorization.
-   **KaggleHub**: Automated dataset management.

### Frontend (App)
-   **React + TypeScript**: Type-safe UI development.
-   **Vite**: Next-generation frontend tooling.
-   **Tailwind CSS**: Utility-first styling.
-   **shadcn/ui**: Reusable component library.

## 🏁 Quick Start

### Prerequisites
-   Node.js & npm
-   Python 3.10+

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/model-playground.git
cd model-playground
```

### 2. Set Up ML Backend
```bash
cd api/ml_service

# Run the automated setup script
# This creates venv, installs dependencies, downloads data, trains model, and starts server
./start.sh
```
*The API will be available at `http://localhost:8000`*

### 3. Start Frontend
Open a new terminal:
```bash
# Install dependencies
npm install

# Start development server
npm run dev
```
*The App will be available at `http://localhost:5173`*

## 🧠 Methodology

### Data Pipeline
1.  **Ingestion:** 79,000+ news articles sourced from Kaggle.
2.  **Preprocessing:** Text cleaning (lowercase, regex removal of non-alpha chars).
3.  **Vectorization:** TF-IDF (Term Frequency-Inverse Document Frequency) with top 1000 features.

### Model Architecture
A lightweight Feed-Forward Neural Network:
```
Input Layer (1000 nodes) ➔ ReLU ➔ Hidden Layer (64 nodes) ➔ Sigmoid ➔ Output (Probability)
```

## 📂 Project Structure

```
├── api/ml_service/        # Python Backend
│   ├── app.py             # FastAPI Server
│   ├── model.py           # PyTorch Model Definition
│   ├── train.py           # Training Pipeline
│   └── models/            # Saved Artifacts (.pth, .pkl)
├── src/                   # React Frontend
│   ├── components/        # UI Components
│   └── lib/api/           # API Client
└── docs/                  # Documentation
    ├── ARCHITECTURE.md    # System Design
    └── ML_INTEGRATION.md  # Integration Guide
```

## ⚠️ Limitations

-   **Context Window:** The model analyzes text patterns but does not verify facts against external databases.
-   **Language:** Trained primarily on English news articles.
-   **Drift:** News topics change rapidly; the model requires periodic retraining with fresh data.

## 🤝 Contributing

1.  Fork the Project
2.  Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3.  Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4.  Push to the Branch (`git push origin feature/AmazingFeature`)
5.  Open a Pull Request

## 📜 License

Distributed under the MIT License. See `LICENSE` for more information.