# Vercel Deployment Guide

## ✅ Hybrid Deployment Architecture (Implemented)

**Current Setup:**

| Component | Platform | Status |
|-----------|----------|--------|
| **Frontend** | Vercel | Ready to deploy |
| **ML Backend** | Hugging Face Spaces | ✅ Deployed and Live |
| **Database** | Supabase | ✅ Configured |

**ML API URL:** `https://yosemite000-misinformation-detector.hf.space`

## Why This Architecture?

This hybrid approach provides the best of both platforms:

**Vercel (Frontend):**
- ✅ Instant deployments from GitHub
- ✅ Global CDN for fast page loads
- ✅ Automatic HTTPS and custom domains
- ✅ Free tier for unlimited static sites

**Hugging Face Spaces (ML Backend):**
- ✅ Optimized for ML model hosting
- ✅ No 250MB deployment size limits
- ✅ Persistent Docker containers
- ✅ Long-running inference workloads
- ✅ Already deployed and tested

---

## Deploying Frontend to Vercel

### Prerequisites

1. A [Vercel account](https://vercel.com/signup) (free tier works)
2. Vercel CLI installed: `npm i -g vercel`
3. The project pushed to GitHub (recommended) or deploy from local

### Option 1: Deploy from GitHub (Recommended)

1. **Push to GitHub** (if not already done):
   ```bash
   git add .
   git commit -m "Ready for Vercel deployment"
   git push origin main
   ```

2. **Connect to Vercel**:
   - Go to [vercel.com/new](https://vercel.com/new)
   - Click "Import Git Repository"
   - Select your `model-playground` repository
   - Framework Preset: **Vite** (should auto-detect)
   - Root Directory: `./` (keep default)
   - Click **Deploy**

3. **Set Environment Variables**:
   
   In Vercel Dashboard → Settings → Environment Variables, add:
   
   ```
   VITE_ML_API_URL=https://yosemite000-misinformation-detector.hf.space
   VITE_SUPABASE_URL=https://your-project.supabase.co
   VITE_SUPABASE_PUBLISHABLE_KEY=your_key_here
   ```

4. **Redeploy** after adding environment variables:
   - Go to Deployments tab
   - Click "Redeploy" on the latest deployment

### Option 2: Deploy from CLI

```bash
# From project root
vercel login

# Deploy to production
vercel --prod

# You'll be prompted to enter project details
# Set environment variables in Vercel dashboard afterwards
```

---

## Verification

### 1. Test Frontend
Visit your Vercel deployment URL (e.g., `https://model-playground.vercel.app`)

### 2. Test ML Integration
1. Navigate to `/misinformation` route
2. Enter sample text
3. Click "Analyze Text"
4. Verify prediction results appear

### 3. Check API Connection
Open browser DevTools → Network tab and verify requests are going to:
```
https://yosemite000-misinformation-detector.hf.space/predict
```

---

## Configuration Files

### `vercel.json`

The project includes a `vercel.json` that configures:
- **Frontend:** Built with Vite, served as static files
- **API Folder:** Ignored (since ML backend is on Hugging Face)
- **Rewrites:** SPA routing to `index.html`

```json
{
  "buildCommand": "npm run build",
  "framework": "vite",
  "outputDirectory": "dist",
  "rewrites": [
    {
      "source": "/(.*)",
      "destination": "/index.html"
    }
  ],
  "ignores": [
    "api/**",
    "*.md",
    "*.ipynb"
  ]
}
```

---

## Troubleshooting

### Build Fails on Vercel

**Error:** "Build failed" or TypeScript errors

**Solution:**
1. Verify build works locally: `npm run build`
2. Check that all dependencies are in `package.json` (not just `devDependencies`)
3. Review build logs in Vercel dashboard

### ML API Not Responding

**Error:** Network error or timeout when testing misinformation detection

**Solution:**
1. Verify `VITE_ML_API_URL` environment variable is set correctly in Vercel
2. Test HF Spaces directly:
   ```bash
   curl https://yosemite000-misinformation-detector.hf.space/health
   ```
3. Check browser console for CORS errors
4. Redeploy after updating environment variables

### Environment Variables Not Working

**Issue:** App can't connect to Supabase or ML API

**Solution:**
1. Environment variables must be prefixed with `VITE_` for Vite projects
2. After adding/changing env vars in Vercel, you must **redeploy**
3. Check the deployment logs to see which env vars were detected

### Custom Domain Issues

**Issue:** Custom domain not working

**Solution:**
1. Add domain in Vercel Dashboard → Settings → Domains
2. Update DNS records as instructed by Vercel
3. Wait for DNS propagation (can take up to 48 hours)

---

## CI/CD with GitHub

Once connected to GitHub, Vercel automatically:
- ✅ Deploys every push to `main` branch (production)
- ✅ Creates preview deployments for pull requests
- ✅ Runs build checks before deploying

Check the "Git" section in Vercel dashboard to configure branch deployments.

---

**Ready to deploy!** 🚀

The ML backend is already live on Hugging Face. Deploy your frontend with `vercel --prod` or connect your GitHub repository!
