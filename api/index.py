# Vercel Serverless Function entrypoint
import sys
import os

# Add parent directory to path so we can import SmartPauseApp
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from SmartPauseApp import app
