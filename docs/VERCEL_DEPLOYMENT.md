# Vercel Deployment Guide

## ⚠️ Important: Recommended Deployment Strategy

**For ML-Heavy Applications, we recommend a Hybrid Approach:**

1. **Frontend:** Deploy to Vercel (fast, free, auto-deploys from GitHub)
2. **ML Backend:** Deploy to [Railway](https://railway.app) or [Render](https://render.com)

**Why?**
- Vercel serverless functions have limitations for ML workloads:
  - 250MB deployment size limit (PyTorch with CUDA is ~800MB)
  - 10-second timeout on free tier
  - Cold starts can be slow for model loading
- Railway/Render support persistent containers perfect for ML services

**Quick Railway Deployment (Recommended):**
```bash
# Install Railway CLI
npm i -g @railway/cli

# Deploy backend
cd api/ml_service
railway login
railway init
railway up

# Get your backend URL (e.g., https://your-service.railway.app)
# Copy it for the next step
```

Then deploy frontend to Vercel with:
```bash
vercel --prod
# Set environment variable: VITE_ML_API_URL=https://your-service.railway.app
```

---

## Vercel-Only Deployment (Experimental)

If you still want to deploy everything to Vercel, continue below. Note: This may hit size limits or timeout issues.

## Prerequisites

1. A [Vercel account](https://vercel.com/signup) (free tier works)
2. Vercel CLI installed: `npm i -g vercel`
3. The project pushed to GitHub

## Architecture

```
Vercel Deployment
├── Frontend (Static + SPA) ➔ https://your-app.vercel.app
└── Backend (Serverless Functions) ➔ https://your-app.vercel.app/api/*
```

## Deployment Steps

### 1. Install Vercel CLI

```bash
npm install -g vercel
```

### 2. Login to Vercel

```bash
vercel login
```

### 3. Deploy from Local

```bash
# From project root
vercel

# Follow prompts:
# - Set up and deploy? Y
# - Which scope? (your account)
# - Link to existing project? N
# - Project name? model-playground
# - Directory? ./
# - Override settings? N
```

### 4. Set Environment Variables

In your Vercel dashboard:
1. Go to your project → Settings → Environment Variables
2. Add these variables:

```
VITE_SUPABASE_URL=https://your-project.supabase.co
VITE_SUPABASE_PUBLISHABLE_KEY=your_key_here
VITE_ML_API_URL=https://your-app.vercel.app/api
```

### 5. Deploy to Production

```bash
vercel --prod
```

## Configuration Files

### `vercel.json`

The project includes a `vercel.json` that configures:
- **Frontend:** Built with Vite, served as static files
- **Backend:** Python functions at `/api/*` route

### `api/index.py`

This is the **serverless handler** that wraps the FastAPI app using Mangum adapter.

## Important Notes

### Model Files

**Issue:** Vercel has a 250MB deployment limit, but our model is small (~300KB), so it's fine.

**For larger models:**
1. Host model on Hugging Face Hub
2. Update `model.py` to download from HF on cold start:

```python
from huggingface_hub import hf_hub_download

model_path = hf_hub_download(
    repo_id="your-username/fake-news-model",
    filename="fake_news_model.pth"
)
```

### Cold Starts

Serverless functions "wake up" on first request, causing ~2-5 second delays. Subsequent requests are fast.

**Mitigation:**
- Use Vercel's Edge Functions (if your model is small enough)
- Implement caching for predictions
- Add a "warming" endpoint that gets pinged periodically

### CORS

The backend's CORS settings in `config.py` should include your Vercel domain:

```python
CORS_ORIGINS = [
    "https://your-app.vercel.app",
    "http://localhost:8080"
]
```

## Vercel Dashboard

After deployment, visit:
```
https://vercel.com/your-username/model-playground
```

You can:
- View deployments
- Check logs
- Set environment variables
- Configure custom domains

## Testing the Deployed App

1. **Frontend:** Visit `https://your-app.vercel.app`
2. **Backend API:** Visit `https://your-app.vercel.app/api/health`

Expected response:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "device": "cpu"
}
```

## Troubleshooting

### Build Fails

**Error:** "PyTorch too large"
**Solution:** Use the CPU-only version (already configured in `requirements.txt`)

### API Returns 500

**Error:** Model not found
**Solution:** 
1. Check that `models/` directory is included in deployment
2. Or switch to Hugging Face Hub hosting

### Cold Start Timeout

**Error:** Function timeout after 10s
**Solution:**
1. Upgrade to Vercel Pro (60s timeout)
2. Or host the ML backend on Railway/Render (see below)

## Alternative: Hybrid Deployment

If Vercel serverless is too limiting for the ML model:

1. **Frontend:** Deploy to Vercel (as above)
2. **Backend:** Deploy to [Railway](https://railway.app) or [Render](https://render.com)
   - These support long-running containers
   - Update `VITE_ML_API_URL` to point to Railway/Render URL

### Railway Deployment (Backend only)

```bash
# Install Railway CLI
npm i -g @railway/cli

# Login and deploy
railway login
railway init
railway up
```

Railway will auto-detect the Python backend and deploy it as a persistent service.

## CI/CD with GitHub

Vercel automatically deploys on every push to `main`. Check the "Deployments" tab in your dashboard to see build status.

---

**Ready to deploy!** 🚀

Run `vercel --prod` and share your live link!
