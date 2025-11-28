"""
Fake News Detection Model
Extracted from Misinformation for fake.ipynb
"""
import torch
import torch.nn as nn
import re
import pickle
from pathlib import Path
from typing import Dict, List
import numpy as np


class FakeNewsClassifier(nn.Module):
    """Simple PyTorch neural network for fake news detection"""
    
    def __init__(self, input_dim: int):
        super(FakeNewsClassifier, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.fc(x)


class MisinformationDetector:
    """Wrapper class for the fake news detection model"""
    
    def __init__(self, model_path: str = None, vectorizer_path: str = None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.vectorizer = None
        
        if model_path and vectorizer_path:
            self.load_model(model_path, vectorizer_path)
    
    def preprocess_text(self, text: str) -> str:
        """Preprocess text as done in the notebook"""
        if not isinstance(text, str):
            return ""
        
        text = text.lower()
        text = re.sub(r'[^a-z\s]', '', text)
        return text
    
    def load_model(self, model_path: str, vectorizer_path: str):
        """Load trained model and vectorizer from disk"""
        # Load vectorizer
        with open(vectorizer_path, 'rb') as f:
            self.vectorizer = pickle.load(f)
        
        # Load model
        input_dim = len(self.vectorizer.get_feature_names_out())
        self.model = FakeNewsClassifier(input_dim).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
    
    def predict(self, text: str) -> Dict[str, float]:
        """Predict if text is fake news"""
        if self.model is None or self.vectorizer is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        # Preprocess
        processed_text = self.preprocess_text(text)
        
        # Vectorize
        X = self.vectorizer.transform([processed_text]).toarray()
        
        # Predict
        with torch.no_grad():
            X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
            output = self.model(X_tensor)
            confidence = output.item()
        
        # Label: 1 = True (real news), 0 = Fake
        is_real = confidence >= 0.5
        
        return {
            "prediction": "real" if is_real else "fake",
            "confidence": confidence if is_real else (1 - confidence),
            "raw_score": confidence
        }
    
    def batch_predict(self, texts: List[str]) -> List[Dict[str, float]]:
        """Predict multiple texts at once"""
        if self.model is None or self.vectorizer is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        # Preprocess all texts
        processed_texts = [self.preprocess_text(text) for text in texts]
        
        # Vectorize
        X = self.vectorizer.transform(processed_texts).toarray()
        
        # Predict
        with torch.no_grad():
            X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
            outputs = self.model(X_tensor)
        
        # Format results
        results = []
        for confidence in outputs.cpu().numpy():
            confidence = float(confidence[0])
            is_real = confidence >= 0.5
            results.append({
                "prediction": "real" if is_real else "fake",
                "confidence": confidence if is_real else (1 - confidence),
                "raw_score": confidence
            })
        
        return results
