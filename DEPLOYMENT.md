# Quick Start: Deploy SmartQuit to Render

## Prerequisites
- GitHub account
- Render account (free tier works)
- Neon PostgreSQL account (free tier)

## Step 1: Create Neon Database (5 minutes)

1. Go to https://console.neon.tech/sign_up
2. Create new project
3. Copy connection string from "Connection string" tab
   - Format: `postgresql://user:password@host/dbname`

## Step 2: Push Code to GitHub (2 minutes)

```bash
cd SmartpauseApp
git add .
git commit -m "SmartQuit API ready for deployment"
git push origin main
```

**Files automatically included for Render:**
- ✅ `requirements.txt` - Python dependencies
- ✅ `Procfile` - Gunicorn configuration
- ✅ `runtime.txt` - Python 3.11.7
- ✅ `.gitignore` - Keeps `.env` and cache out of repo

## Step 3: Deploy on Render (5 minutes)

1. Go to https://render.com
2. Sign in with GitHub
3. Click **"New +"** → **"Web Service"**
4. Select your SmartpauseApp repository
5. Fill in deployment settings:
   - **Name**: `smartquit-api`
   - **Environment**: `Python 3`
   - **Build Command**: Leave as is (auto-detected from Procfile)
   - **Start Command**: Leave as is (Procfile)
   - **Plan**: Free (or Starter)

6. Click **"Create Web Service"**
7. Go to **"Environment"** tab on the dashboard
8. Add environment variables:
   ```
   ENVIRONMENT=production
   DATABASE_URL=<paste-your-neon-connection-string>
   ```
9. **Save** and Render auto-redeploys

## Step 4: Test Your API (1 minute)

Your API is live at: `https://smartquit-api.onrender.com`

Test with:
```bash
curl https://smartquit-api.onrender.com/
```

You should see:
```json
{
  "service": "SmartQuit Edge ML API",
  "version": "2.0.0 - Database Enabled",
  "status": "running",
  ...
}
```

## Continuous Deployment

Every time you push to GitHub:
```bash
git add .
git commit -m "Your changes"
git push origin main
```

Render automatically redeploys! 🚀

## Troubleshooting

**Problem**: "Deployment failed"
- Check Render logs: Dashboard → "Logs"
- Verify `DATABASE_URL` is set in Environment

**Problem**: "Build succeeded but app won't start"
- Check `Procfile` syntax
- Verify Python version in `runtime.txt`

**Problem**: "ModuleNotFoundError"
- Check all imports are in `requirements.txt`
- Redeploy after updating

## Next Steps

1. **Test API endpoints**: Open `https://smartquit-api.onrender.com/docs`
2. **Integrate with Android app**: Use the live API URL
3. **Monitor**: Check Render logs regularly
4. **Scale**: Upgrade plan if needed for production use

---

**Need help?** Check README.md for full documentation
