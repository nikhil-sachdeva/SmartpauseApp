# SmartQuit Edge ML API 🧠

Backend FastAPI service for training and serving Q-Learning models to Android devices. Features daily model updates at 3 AM per user and PostgreSQL persistence via Neon.

**Version:** 2.0.0 - Database Enabled  
**Status:** Production Ready ✅

---

## Features

- 🤖 **Edge ML Models**: Q-Learning agents serialized for on-device inference
- 📱 **Device Integration**: Android app model download & session upload
- 📊 **Baseline Statistics**: Automatic calculation from first 7 days
- 🎯 **Daily Training**: Background tasks train models with real user data
- 🗄️ **PostgreSQL Storage**: Neon DB for production persistence
- 🔄 **CORS Enabled**: Full Android app integration
- 📈 **Analytics**: Real-time compliance and usage analytics

---

## Project Structure

```
SmartpauseApp/
├── SmartPauseApp.py          # Main FastAPI application
├── database.py               # SQLAlchemy models & DB config
├── db_service.py             # Database operations service
├── requirements.txt          # Python dependencies
├── Procfile                  # Render deployment config
├── runtime.txt               # Python version specification
├── .gitignore                # Git ignore rules
├── .env                      # Environment variables (local)
└── SmartQuit.ipynb          # Jupyter notebook (reference)
```

---

## Local Development Setup

### Prerequisites

- Python 3.11+
- pip or poetry
- PostgreSQL 13+ (or SQLite for local dev)

### Installation

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd SmartpauseApp
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Create `.env` file** (local development)
   ```bash
   # Local SQLite database (no DATABASE_URL = automatic SQLite)
   ENVIRONMENT=development
   ```
   
   OR for PostgreSQL:
   ```bash
   ENVIRONMENT=production
   DATABASE_URL=postgresql://user:password@localhost/smartquit_db
   ```

5. **Run the application**
   ```bash
   uvicorn SmartPauseApp:app --reload --host 0.0.0.0 --port 8000
   ```

6. **Access the API**
   - Docs: http://localhost:8000/docs
   - ReDoc: http://localhost:8000/redoc
   - API Tester: http://localhost:8000/static/api_tester.html

---

## Deployment on Render 🚀

### Prerequisites

- GitHub account with this repository
- Render account (free tier available)
- Neon PostgreSQL account (free tier available)

### Step 1: Set Up PostgreSQL Database on Neon

1. Go to [Neon Console](https://console.neon.tech)
2. Create a new project
3. Copy your database connection string: `postgresql://user:password@host/dbname`

### Step 2: Deploy on Render

1. **Push code to GitHub**
   ```bash
   git add .
   git commit -m "Initial commit: SmartQuit API ready for Render"
   git push origin main
   ```

2. **Go to [Render.com](https://render.com)** and sign in

3. **Create a new Web Service**
   - Click "New +" → "Web Service"
   - Connect your GitHub account
   - Select this repository
   - Configure:
     - **Name**: `smartquit-api` (or your choice)
     - **Environment**: `Python`
     - **Build Command**: `pip install -r requirements.txt`
     - **Start Command**: `gunicorn -w 4 -k uvicorn.workers.UvicornWorker SmartPauseApp:app`

4. **Add Environment Variables**
   - Click "Environment" in the dashboard
   - Add:
     ```
     ENVIRONMENT=production
     DATABASE_URL=<your-neon-connection-string>
     ```

5. **Deploy**
   - Click "Create Web Service"
   - Render will automatically deploy when you push to GitHub

### Step 3: Verify Deployment

- Your API will be available at: `https://<service-name>.onrender.com`
- Test the endpoints:
  ```bash
  curl https://<service-name>.onrender.com/
  ```

---

## API Endpoints

### User Management

**Register User**
```bash
POST /api/v1/users/register
Content-Type: application/json

{
  "user_id": "user123",
  "device_info": {
    "brand": "Samsung",
    "model": "Galaxy S21",
    "os_version": "13"
  }
}
```

### Session Management

**Upload Daily Sessions**
```bash
POST /api/v1/sessions/upload
Content-Type: application/json

{
  "user_id": "user123",
  "date": "2024-01-09",
  "sessions": [
    {
      "app_name": "com.instagram.android",
      "start_time": "2024-01-09T10:30:00",
      "end_time": "2024-01-09T10:45:00",
      "duration_seconds": 900,
      "vibration_occurred": true,
      "user_complied": true
    }
  ]
}
```

### Model Management

**Download Model**
```bash
GET /api/v1/model/download/{user_id}?format=binary
```

**Get Model Status**
```bash
GET /api/v1/model/status/{user_id}
```

### Analytics

**Get User Analytics**
```bash
GET /api/v1/analytics/{user_id}
```

---

## Architecture

### Data Flow

1. **Days 0-6 (Baseline Week)**
   - Device uploads sessions daily
   - API calculates baseline statistics
   - No model training

2. **Days 7+ (Intervention Week)**
   - Device uploads sessions daily
   - Background task trains Q-Learning model
   - Updated model available for download at 3 AM

### Database Schema

- **users**: User accounts and metadata
- **sessions**: Individual app usage sessions
- **baseline_stats**: Calculated statistics from first 7 days
- **model_checkpoints**: Saved Q-Learning models
- **training_logs**: Training history and learning updates
- **grouped_sessions**: Sessions grouped by time gaps
- **feedback_logs**: Optional on-device feedback

### Q-Learning Agent

- **State**: (is_target_app, is_short_session, day_quarter, is_weekday)
- **Action**: 0 (no vibration) or 1 (send vibration)
- **Reward**: Based on vibration compliance, session breaks, and long session penalties
- **Learning**: Daily batch training on uploaded sessions

---

## Environment Variables

### Production (Render)

```env
ENVIRONMENT=production
DATABASE_URL=postgresql://user:password@neon-host/dbname
```

### Local Development

```env
ENVIRONMENT=development
# DATABASE_URL optional - uses SQLite if not set
```

---

## Troubleshooting

### "Database connection failed"

**Solution**: Check that `DATABASE_URL` environment variable is set correctly on Render

### "Module not found" errors

**Solution**: Ensure all dependencies are in `requirements.txt` and reinstall:
```bash
pip install --no-cache-dir -r requirements.txt
```

### Port already in use

**Solution**: Change the port:
```bash
uvicorn SmartPauseApp:app --port 8001
```

### Static files not found

**Solution**: Ensure `/static` directory exists with `api_tester.html`

---

## Development Workflow

### Making Changes

1. Create a new branch
   ```bash
   git checkout -b feature/your-feature
   ```

2. Make changes to `.py` files

3. Test locally
   ```bash
   uvicorn SmartPauseApp:app --reload
   ```

4. Commit and push
   ```bash
   git add .
   git commit -m "Description of changes"
   git push origin feature/your-feature
   ```

5. Render will auto-deploy when you merge to main

### Adding Dependencies

1. Install the package locally
   ```bash
   pip install new-package
   ```

2. Update `requirements.txt`
   ```bash
   pip freeze | grep new-package >> requirements.txt
   ```

3. Commit and push

---

## Production Checklist

- ✅ `.env` file added to `.gitignore`
- ✅ `DATABASE_URL` configured in Render
- ✅ `ENVIRONMENT=production` set
- ✅ `requirements.txt` complete
- ✅ `Procfile` configured
- ✅ `runtime.txt` specifies Python version
- ✅ Code pushed to GitHub main branch
- ✅ Render service created and deployed

---

## Support & Documentation

- FastAPI Docs: https://fastapi.tiangolo.com/
- SQLAlchemy Docs: https://docs.sqlalchemy.org/
- Render Docs: https://render.com/docs
- Neon Docs: https://neon.tech/docs

---

## License

Proprietary - SmartQuit Project

---

**Last Updated**: January 2026  
**Maintained By**: SmartQuit Development Team
