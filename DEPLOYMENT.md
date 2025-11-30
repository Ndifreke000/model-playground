# Deployment Guide

This guide covers how to deploy the Misinformation Detection ML Service to various platforms.

---

## 🤗 Hugging Face Spaces (Recommended for ML)

> [!NOTE]
> **Status:** ✅ **DEPLOYED AND LIVE**
> 
> The ML API is currently running on Hugging Face Spaces at:
> **https://yosemite000-misinformation-detector.hf.space**

Hugging Face Spaces is optimized for deploying machine learning models with excellent support for Docker-based deployments.

### Testing the Live Deployment

```bash
# Health check
curl https://yosemite000-misinformation-detector.hf.space/health

# Make a prediction
curl -X POST "https://yosemite000-misinformation-detector.hf.space/predict" \
  -H "Content-Type: application/json" \
  -d '{"text": "This is a sample news article to test the API..."}'
```

### Redeploying (If Needed)


### Prerequisites
- A Hugging Face account ([sign up here](https://huggingface.co/join))
- Git installed locally
- Docker installed (for local testing)

### Step 1: Create a New Space

1. Go to [Hugging Face Spaces](https://huggingface.co/spaces)
2. Click **"Create new Space"**
3. Choose:
   - **Space name**: `misinformation-detector` (or your preferred name)
   - **License**: MIT or Apache 2.0
   - **Space SDK**: **Docker** ⚠️ Important!
   - **Visibility**: Public or Private

### Step 2: Clone the Space Repository

```bash
# Clone your new Space
git clone https://huggingface.co/spaces/YOUR_USERNAME/misinformation-detector
cd misinformation-detector
```

### Step 3: Copy Your ML Service Files

```bash
# Copy all files from api/ml_service to the Space directory
cp -r /path/to/model-playground/api/ml_service/* .

# Ensure model files are included
ls models/
# Should show: fake_news_model.pth, tfidf_vectorizer.pkl
```

### Step 4: Create a README.md (Optional but Recommended)

Create a `README.md` in the root with Space metadata:

```markdown
---
title: Misinformation Detector
emoji: 🔍
colorFrom: red
colorTo: orange
sdk: docker
pinned: false
---

# Misinformation Detection API

FastAPI-powered machine learning service for detecting fake news.

## API Endpoints

- `GET /health` - Health check
- `POST /predict` - Predict single text
- `POST /batch_predict` - Batch predictions

## Example Usage

\`\`\`bash
curl -X POST "https://YOUR_USERNAME-misinformation-detector.hf.space/predict" \\
  -H "Content-Type: application/json" \\
  -d '{"text": "Breaking news article text here..."}'
\`\`\`
```

### Step 5: Push to Hugging Face

```bash
git add .
git commit -m "Initial deployment"
git push
```

### Step 6: Monitor the Space

1. Visit your Space page: `https://huggingface.co/spaces/yosemite000/misinformation-detector`
2. Check the logs in the "Logs" tab for any errors
3. Status should show "Running" with a green indicator

### Testing Your Deployment

```bash
# Health check
curl https://YOUR_USERNAME-misinformation-detector.hf.space/health

# Make a prediction
curl -X POST "https://YOUR_USERNAME-misinformation-detector.hf.space/predict" \
  -H "Content-Type: application/json" \
  -d '{"text": "This is a sample news article to test the API..."}'
```

### Important Notes

- **Cold Starts**: Free Hugging Face Spaces sleep after inactivity. First request may take 30-60s.
- **Upgrade**: For always-on service, upgrade to **Spaces Hardware** (starts at $0.60/day).
- **Logs**: View logs in the Space's "Logs" tab for debugging.

---

## 🎨 Render (Alternative)

Render is great for web services with straightforward Git-based deployments.

### Prerequisites
- A Render account ([sign up here](https://render.com))
- GitHub/GitLab repository with your code

### Deployment Steps

1. **Push to GitHub**:
   ```bash
   cd api/ml_service
   git init
   git add .
   git commit -m "Initial commit"
   git remote add origin https://github.com/YOUR_USERNAME/ml-service.git
   git push -u origin main
   ```

2. **Create New Web Service** on Render:
   - Go to [Render Dashboard](https://dashboard.render.com)
   - Click **"New +"** → **"Web Service"**
   - Connect your GitHub repository
   - Configure:
     - **Name**: `misinformation-detector`
     - **Environment**: Docker
     - **Branch**: `main`
     - **Plan**: Free (or paid for production)

3. **Environment Variables** (if needed):
   - `CORS_ORIGINS`: Your frontend URL

4. **Deploy**: Render will auto-build and deploy from your Dockerfile

### Testing
```bash
curl https://misinformation-detector.onrender.com/health
```

---

## ✈️ Fly.io (Production-Grade)

Fly.io offers global edge deployment with excellent performance.

### Prerequisites
- A Fly.io account ([sign up here](https://fly.io/app/sign-up))
- Flyctl CLI installed:
  ```bash
  curl -L https://fly.io/install.sh | sh
  ```

### Deployment Steps

1. **Login to Fly.io**:
   ```bash
   flyctl auth login
   ```

2. **Initialize Fly App**:
   ```bash
   cd api/ml_service
   flyctl launch
   ```

3. **Configure** (when prompted):
   - **App name**: `misinformation-detector` (or auto-generate)
   - **Region**: Choose closest to your users
   - **Deploy now**: Yes

4. **Set Environment Variables** (if needed):
   ```bash
   flyctl secrets set CORS_ORIGINS="https://yourfrontend.com"
   ```

5. **Deploy**:
   ```bash
   flyctl deploy
   ```

### Accessing Your App
```bash
# Your app will be at:
https://misinformation-detector.fly.dev

# Check status
flyctl status

# View logs
flyctl logs
```

---

## 📊 Comparison

| Platform | Best For | Free Tier | Cold Starts | Ease of Use |
|----------|----------|-----------|-------------|-------------|
| **Hugging Face** | ML demos, community sharing | ✅ Yes (with sleep) | Yes | ⭐⭐⭐⭐⭐ |
| **Render** | Simple web services | ✅ Yes (with sleep) | Yes | ⭐⭐⭐⭐ |
| **Fly.io** | Production apps | ✅ Limited | No | ⭐⭐⭐ |

---

## 🔧 Local Testing with Docker

Before deploying, test locally:

```bash
# Build the image
docker build -t ml-service .

# Run the container
docker run -p 7860:7860 ml-service

# Test in another terminal
curl http://localhost:7860/health
```

---

## 🚀 Next Steps

1. **Update Frontend**: Point your frontend's `VITE_ML_API_URL` to the deployed URL
2. **Monitor Performance**: Use platform logs to track API usage
3. **Scale**: Upgrade to paid tiers for production workloads
4. **CI/CD**: Set up auto-deployments on Git push

---

## 📝 Troubleshooting

### Model Files Not Found
- Ensure `models/fake_news_model.pth` and `models/tfidf_vectorizer.pkl` are committed
- Check `.gitignore` and `.dockerignore` aren't excluding them

### Port Issues
- Hugging Face uses port `7860` (configured via `PORT` env var)
- Railway/Render auto-assign ports (handled by `$PORT`)
- Local default is `8000`

### Memory Issues
- ML models can be RAM-intensive
- Upgrade to larger instances if needed
- Consider model quantization for smaller deployments
