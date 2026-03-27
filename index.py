# Vercel entrypoint - imports FastAPI app from main module
from SmartPauseApp import app as application

# Expose as 'app' for Vercel detection
app = application


