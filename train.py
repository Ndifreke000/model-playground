"""
Training script for fake news detection model
Extracted from Misinformation for fake.ipynb
"""
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix
import pickle
import time
import matplotlib.pyplot as plt
import seaborn as sns

from model import FakeNewsClassifier, MisinformationDetector
import config


def load_data():
    """Load data from Kaggle dataset using kagglehub"""
    print("Loading data from Kaggle...")
    
    try:
        import kagglehub
        
        # Download dataset using kagglehub (same as notebook)
        print("Downloading dataset via kagglehub...")
        dataset_path = kagglehub.dataset_download('stevenpeutz/misinformation-fake-news-text-dataset-79k')
        print(f'Data source download complete: {dataset_path}')
        
        # List files in dataset
        import os
        print("\nDataset files:")
        for dirname, _, filenames in os.walk(dataset_path):
            for filename in filenames:
                filepath = os.path.join(dirname, filename)
                print(f"  {filepath}")
        
        # Load the CSV files
        true_df = pd.read_csv(os.path.join(dataset_path, 'DataSet_Misinfo_TRUE.csv'))
        fake_df = pd.read_csv(os.path.join(dataset_path, 'DataSet_Misinfo_FAKE.csv'))
        
    except ImportError:
        print("ERROR: kagglehub not installed!")
        print("Install with: pip install kagglehub")
        raise
    except FileNotFoundError as e:
        print("ERROR: Dataset files not found!")
        print(f"Expected path: {dataset_path}")
        print("\nTrying alternative data/ directory...")
        
        # Fallback to manual data directory
        try:
            true_df = pd.read_csv('data/DataSet_Misinfo_TRUE.csv')
            fake_df = pd.read_csv('data/DataSet_Misinfo_FAKE.csv')
            print("Loaded from local data/ directory")
        except FileNotFoundError:
            print("\nPlease ensure dataset is available via kagglehub or manually placed in data/")
            raise
    
    # Add labels
    true_df['label'] = 1  # Real news
    fake_df['label'] = 0  # Fake news
    
    # Combine
    df = pd.concat([true_df, fake_df], ignore_index=True)
    df = df[['text', 'label']].dropna()
    
    print(f"Loaded {len(df)} articles ({len(true_df)} real, {len(fake_df)} fake)")
    return df


def preprocess_data(df):
    """Preprocess text data"""
    print("Preprocessing text...")
    detector = MisinformationDetector()
    df['text'] = df['text'].apply(detector.preprocess_text)
    return df


def train_model():
    """Main training function"""
    print("=" * 60)
    print("FAKE NEWS DETECTION MODEL TRAINING")
    print("=" * 60)
    
    # Load and preprocess data
    df = load_data()
    df = preprocess_data(df)
    
    # Create TF-IDF vectorizer
    print(f"\nCreating TF-IDF vectorizer (max_features={config.MAX_FEATURES})...")
    vectorizer = TfidfVectorizer(max_features=config.MAX_FEATURES)
    X = vectorizer.fit_transform(df['text']).toarray()
    y = df['label'].values
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Convert to tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1).to(device)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1).to(device)
    
    # Initialize model
    model = FakeNewsClassifier(input_dim=X_train.shape[1]).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    
    # Training loop
    print(f"\nTraining for {config.NUM_EPOCHS} epochs...")
    losses = []
    start_time = time.time()
    
    for epoch in range(config.NUM_EPOCHS):
        model.train()
        optimizer.zero_grad()
        
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        print(f"Epoch {epoch + 1}/{config.NUM_EPOCHS}, Loss: {loss.item():.4f}")
    
    training_time = time.time() - start_time
    print(f"\nTraining completed in {training_time:.2f} seconds")
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test_tensor)
        predictions = (test_outputs >= 0.5).float()
        accuracy = accuracy_score(y_test_tensor.cpu(), predictions.cpu())
    
    print(f"Test Accuracy: {accuracy:.4f}")
    
    # Confusion matrix
    cm = confusion_matrix(y_test_tensor.cpu(), predictions.cpu())
    print("\nConfusion Matrix:")
    print(cm)
    
    # Save model and vectorizer
    print(f"\nSaving model to {config.MODEL_PATH}...")
    torch.save(model.state_dict(), config.MODEL_PATH)
    
    print(f"Saving vectorizer to {config.VECTORIZER_PATH}...")
    with open(config.VECTORIZER_PATH, 'wb') as f:
        pickle.dump(vectorizer, f)
    
    # Plot training loss
    plt.figure(figsize=(10, 6))
    plt.plot(losses, marker='o')
    plt.title('Training Loss per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.savefig(config.MODEL_DIR / 'training_loss.png')
    print(f"Training plot saved to {config.MODEL_DIR / 'training_loss.png'}")
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(config.MODEL_DIR / 'confusion_matrix.png')
    print(f"Confusion matrix saved to {config.MODEL_DIR / 'confusion_matrix.png'}")
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE!")
    print("=" * 60)
    print(f"Model files saved in: {config.MODEL_DIR}")
    print("You can now start the API server with: python app.py")


if __name__ == "__main__":
    train_model()
