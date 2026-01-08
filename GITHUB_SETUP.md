# GitHub Setup Guide

## Initialize Git Repository (if not already done)

```bash
cd SmartpauseApp

# Initialize git
git init

# Add all files
git add .

# Create initial commit
git commit -m "Initial commit: SmartQuit API with database support"

# Add GitHub remote
git remote add origin https://github.com/YOUR_USERNAME/SmartpauseApp.git

# Rename branch to main
git branch -M main

# Push to GitHub
git push -u origin main
```

## Create Repository on GitHub

1. Go to https://github.com/new
2. Repository name: `SmartpauseApp`
3. Description: "SmartQuit Edge ML API - Q-Learning models for app usage reduction"
4. Select **Public** (for Render integration)
5. **Do NOT** initialize with README (we already have one)
6. Click **"Create repository"**

Then run the commands above to push your code.

## Repository Structure on GitHub

Your repo will have:

```
SmartpauseApp/
├── SmartPauseApp.py          # Main app (874 lines)
├── database.py               # Database models & config (226 lines)
├── db_service.py             # Database operations (256 lines)
├── SmartQuit.ipynb          # Reference notebook
├── requirements.txt          # ← Render uses this
├── Procfile                  # ← Render uses this
├── runtime.txt               # ← Render uses this
├── .gitignore                # Keeps .env files safe
├── README.md                 # Full documentation
└── DEPLOYMENT.md             # Quick deployment guide
```

## Files Safe from GitHub

These files are in `.gitignore` so they won't be committed:
- ✅ `.env` - Your local secrets
- ✅ `smartquit.db` - Local SQLite database
- ✅ `__pycache__/` - Python cache
- ✅ `venv/` - Virtual environment

## Render's Auto-Detection

When you connect Render to this GitHub repo, it will:
1. ✅ Read `requirements.txt` for dependencies
2. ✅ Read `Procfile` for start command
3. ✅ Read `runtime.txt` for Python version
4. ✅ Install everything automatically
5. ✅ Start your app with Gunicorn

No additional configuration needed in Render settings!

## Keep Your Repo Clean

### Before Pushing

```bash
# Check what will be committed
git status

# Should NOT include:
# - .env (already in .gitignore)
# - __pycache__ (already in .gitignore)
# - smartquit.db (already in .gitignore)
```

### Making Updates

```bash
# Make changes to any .py file
nano SmartPauseApp.py

# Commit
git add SmartPauseApp.py
git commit -m "Fix: improved error handling for model training"

# Push (triggers auto-deploy on Render!)
git push
```

## GitHub Best Practices

1. **Always use descriptive commit messages**
   ```bash
   ✅ Good:   git commit -m "Add analytics endpoint for user dashboard"
   ❌ Bad:    git commit -m "update"
   ```

2. **Don't commit sensitive data**
   - Never push `.env` files with real credentials
   - Never commit API keys or passwords
   - These are already in `.gitignore` ✓

3. **Keep main branch deployment-ready**
   - Always test locally before pushing
   - Only push stable code to main
   - Consider using branches for development

4. **Monitor Deployments**
   - After each push, check Render logs
   - Go to Render dashboard → Logs tab
   - Verify "Build succeeded" message

## Example Workflow

```bash
# 1. Make changes locally
nano SmartPauseApp.py

# 2. Test locally
uvicorn SmartPauseApp:app --reload

# 3. Commit
git add .
git commit -m "Add user compliance analytics"

# 4. Push to GitHub
git push origin main

# 5. Check Render deployment
# → Open Render dashboard
# → View logs
# → Confirm "Build succeeded"

# 6. Verify live API
curl https://smartquit-api.onrender.com/

# Done! ✅
```

## Troubleshooting GitHub Integration

**Problem**: Render doesn't see changes after push
- Wait 30 seconds for webhook
- Go to Render dashboard → "Manual Deploy" → "Deploy latest commit"

**Problem**: "Permission denied" when pushing
- Check remote URL: `git remote -v`
- Re-authenticate: `git config --global credential.helper osxkeychain` (Mac)

**Problem**: Accidentally committed `.env`
```bash
# Remove it from repo history
git rm --cached .env
git commit -m "Remove .env file from repo"
git push
```

---

**Happy deploying!** 🚀
