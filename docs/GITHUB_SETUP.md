# GitHub Setup & Deployment Guide

This guide explains how to push the project to GitHub and deploy the ML service, ensuring reproducibility and best practices.

## 1. Project Structure

Ensure your repository follows this clean structure:

```
model-playground/
├── .gitignore               # Global gitignore
├── README.md                # Main project documentation
├── package.json             # Frontend dependencies
├── src/                     # React Frontend code
├── docs/                    # Documentation
│   ├── ARCHITECTURE.md
│   └── ML_INTEGRATION.md
└── api/
    └── ml_service/          # Standalone Python Service
        ├── .gitignore       # Service-specific ignore
        ├── requirements.txt # Python dependencies
        ├── app.py           # API Entrypoint
        ├── model.py         # Model Architecture
        ├── train.py         # Training Script
        ├── config.py        # Configuration
        └── models/          # Trained Artifacts
            ├── fake_news_model.pth
            └── tfidf_vectorizer.pkl
```

## 2. Git Large File Storage (LFS)

While our current model is small (~300KB), production models can be huge. It is best practice to use Git LFS for binary artifacts.

### Setup LFS
```bash
# 1. Install Git LFS (if not installed)
sudo apt install git-lfs
git lfs install

# 2. Track model files
git lfs track "*.pth"
git lfs track "*.pkl"
git lfs track "*.bin"  # If using Hugging Face models later

# 3. Commit the .gitattributes file
git add .gitattributes
git commit -m "Setup Git LFS for model artifacts"
```

## 3. Pushing to GitHub

```bash
# 1. Initialize repo (if not done)
git init

# 2. Add files
git add .

# 3. Commit
git commit -m "Initial commit: Misinformation Detection System"

# 4. Add remote and push
git remote add origin https://github.com/YOUR_USERNAME/model-playground.git
git push -u origin main
```

## 4. Hugging Face Integration (Optional)

For better model hosting, you can host the *model artifacts* on Hugging Face Hub and load them dynamically, keeping your GitHub repo light.

### Uploading to Hugging Face
1.  Create a model repo on [huggingface.co](https://huggingface.co/new).
2.  Clone it locally.
3.  Copy `fake_news_model.pth` and `tfidf_vectorizer.pkl` into it.
4.  Push to Hugging Face.

### Loading from Hugging Face
Update `model.py` to download from HF Hub:

```python
from huggingface_hub import hf_hub_download

model_path = hf_hub_download(repo_id="your-username/fake-news-detector", filename="fake_news_model.pth")
```

## 5. Reproducibility Best Practices

To ensure this runs on another machine:

1.  **Pin Versions:** We used `requirements.txt` with specific versions (e.g., `torch==2.5.1+cpu`).
2.  **Environment Variables:** Never commit `.env` files. Use `.env.example` template.
3.  **Data Handling:** Do NOT commit the dataset (`.csv` files). We handled this by:
    *   Adding `data/` to `.gitignore`.
    *   Using `kagglehub` in `train.py` to auto-download data on the new machine.

## 6. Deployment Checklist

- [ ] **Docker:** Use the provided Dockerfile in `docs/ML_INTEGRATION.md` for containerization.
- [ ] **CI/CD:** Set up GitHub Actions to run `pytest` on push.
- [ ] **Security:** Ensure `CORS_ORIGINS` in `config.py` is set to your production frontend domain.
