# SmartPause Backend API

Backend service for the SmartPause screen time management system.

## What It Does

- **User Management** — Register users and track their app monitoring preferences
- **Session Tracking** — Store daily app usage sessions uploaded from Android devices
- **Q-Learning Model Training** — Train personalized ML models based on user behavior
- **Model Distribution** — Serve trained models back to devices for on-device inference
- **Baseline Calculation** — Compute usage baselines from initial data to personalize interventions
- **Rate Limiting** — Prevent duplicate uploads (1-hour minimum gap between uploads)

## Tech Stack

- **Framework**: FastAPI (Python)
- **Database**: PostgreSQL (Neon)
- **ORM**: SQLAlchemy
- **ML**: Custom Q-Learning implementation
- **Hosting**: Vercel / Render

## API Endpoints

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/api/v1/users/register` | POST | Register new user |
| `/api/v1/sessions/upload` | POST | Upload daily session & query data |
| `/api/v1/model/download/{user_id}` | GET | Download trained Q-table model |
| `/api/v1/model/status/{user_id}` | GET | Get model training status |
| `/api/v1/analytics/{user_id}` | GET | Get user analytics |
| `/api/v1/baseline/{user_id}` | GET | Get baseline statistics |

## Environment Variables

| Variable | Description |
|----------|-------------|
| `DATABASE_URL` | PostgreSQL connection string |
| `ENVIRONMENT` | `production` or `development` |
