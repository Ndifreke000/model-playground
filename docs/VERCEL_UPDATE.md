# Vercel Environment Variable Update Guide

Follow these steps to update your Vercel deployment with the new Hugging Face ML API URL.

## Step 1: Access Vercel Dashboard

1. Go to [Vercel Dashboard](https://vercel.com/dashboard)
2. Find your project: `model-playground` (or similar name)
3. Click on the project to open it

## Step 2: Update Environment Variable

1. Click on **"Settings"** tab
2. Click on **"Environment Variables"** in the left sidebar
3. Find the variable: `ml_api_url` (referenced as `@ml_api_url` in vercel.json)
4. Click **"Edit"** or add it if it doesn't exist

**Variable Configuration:**
- **Name**: `ml_api_url`
- **Value**: `https://yosemite000-misinformation-detector.hf.space`
- **Environments**: Select all (Production, Preview, Development)

5. Click **"Save"**

## Step 3: Redeploy

After updating the environment variable:

1. Go to the **"Deployments"** tab
2. Click on the latest deployment
3. Click **"Redeploy"** button (or wait for automatic deployment from GitHub)
4. Confirm the redeployment

**OR** simply push a new commit to GitHub and Vercel will auto-deploy.

## Step 4: Verify

Once deployed:

1. Open your live site URL (e.g., `https://your-site.vercel.app`)
2. Navigate to the Misinformation Detection page
3. Open browser DevTools (F12) → Console
4. Test the ML prediction feature
5. Check that API calls go to: `https://yosemite000-misinformation-detector.hf.space`

## Expected Result

The frontend should now call the Hugging Face API instead of Railway or localhost, and fake news detection should work seamlessly!

---

## Alternative: Update via Vercel CLI

If you prefer command line:

```bash
# Login
vercel login

# Link to project  
vercel link

# Add/update environment variable
vercel env add ml_api_url production
# When prompted, enter: https://yosemite000-misinformation-detector.hf.space

vercel env add ml_api_url preview
# When prompted, enter: https://yosemite000-misinformation-detector.hf.space

# Redeploy
vercel --prod
```

---

## Notes

- The `.env` file has been updated locally to use Hugging Face URL
- This is for local development
- Vercel environment variables override `.env` values in production
- Once pushed to GitHub, Vercel will auto-deploy with the new config
