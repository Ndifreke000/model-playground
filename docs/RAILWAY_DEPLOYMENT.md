# Railway Deployment Guide (Recommended for ML Backend)

## Why Railway for ML Services?

Railway is perfect for hosting the Python ML backend because:
- ✅ Supports full PyTorch with CPU optimizations
- ✅ No 250MB deployment limit
- ✅ Persistent containers (no cold starts)
- ✅ Automatic HTTPS
- ✅ Free tier available ($5/month credit)

## Deployment Steps

### 1. Install Railway CLI

```bash
npm install -g @railway/cli
```

### 2. Login to Railway

```bash
railway login
```

This will open your browser for authentication.

### 3. Deploy the ML Backend

```bash
cd api/ml_service
railway init
```

Railway will prompt:
- **Project name:** `model-playground-ml`
- **Environment:** `production`

### 4. Deploy

```bash
railway up
```

Railway will:
1. Detect it's a Python project
2. Install dependencies from `requirements.txt`
3. Start the app using `app.py`

### 5. Get Your Backend URL

```bash
railway domain
```

This returns something like: `https://model-playground-ml.railway.app`

### 6. Configure Environment Variables (Optional)

```bash
railway variables set CORS_ORIGINS="https://your-frontend.vercel.app"
```

## Deploy Frontend to Vercel

Now that your backend is on Railway, deploy the frontend:

```bash
cd ../..  # Back to project root
vercel --prod
```

When prompted for environment variables:
```
VITE_ML_API_URL=https://model-playground-ml.railway.app
```

## Verify Deployment

Test your Railway backend:
```bash
curl https://model-playground-ml.railway.app/api/health

# Expected response:
# {"status":"healthy","model_loaded":true,"device":"cpu"}
```

Test the full flow:
```bash
curl -X POST https://model-playground-ml.railway.app/api/predict \
  -H "Content-Type: application/json" \
  -d '{"text":"Scientists announce breakthrough..."}'
```

## Railway Configuration

Create `railway.toml` in `api/ml_service/`:

```toml
[build]
builder = "NIXPACKS"

[deploy]
startCommand = "python app.py"
healthcheckPath = "/health"
healthcheckTimeout = 30
restartPolicyType = "ON_FAILURE"
restartPolicyMaxRetries = 3

[env]
PORT = "8000"
```

## Monitoring

Railway provides:
- **Logs:** Real-time logs in the dashboard
- **Metrics:** CPU, memory, and network usage
- **Deployments:** Version history and rollback

Access the dashboard:
```bash
railway open
```

## Cost Estimation

**Free Tier:**
- $5 of usage credits per month
- ~500 hours of execution time
- Perfect for development/testing

**Pro Tier ($5/month):**
- $5 base + usage
- Unlimited execution time
- Custom domains included

For a ML API with moderate traffic, expect ~$5-10/month total.

## Updating the Backend

To deploy updates:

```bash
cd api/ml_service
railway up
```

Railway will automatically redeploy with zero downtime.

## Troubleshooting

### Port Binding Error

Ensure your `app.py` uses Railway's PORT variable:

```python
import os
port = int(os.getenv("PORT", 8000))
uvicorn.run(app, host="0.0.0.0", port=port)
```

### Model Not Loading

Check logs:
```bash
railway logs
```

If model files are missing, ensure they're not in `.gitignore` or download them on startup from Hugging Face.

---

**Result:** Your ML backend runs 24/7 on Railway with automatic scaling, while your frontend deploys instantly to Vercel's CDN. Best of both worlds! 🚀
