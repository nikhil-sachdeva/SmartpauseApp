"""
SmartQuit Edge ML API
Backend API for training and serving Q-Learning models to Android devices
Daily model updates at 3 AM per user - trains from day 1 regardless of baseline period
"""

from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel
from typing import List, Dict, Tuple, Optional
from datetime import datetime, timedelta
from collections import defaultdict
import json
import math
import random
import pickle
import base64
import os
import ast

from database import get_db, initialize_database
from db_service import DatabaseService
from sqlalchemy.orm import Session as SQLSession
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
import io

app = FastAPI(title="SmartQuit API", version="2.0.0 - Database Enabled")

# Initialize database on startup
@app.on_event("startup")
async def startup_event():
    """Initialize database with tables and migrations on server startup"""
    print("🚀 Initializing database on server startup...")
    try:
        if initialize_database():
            print("✅ Database initialization completed")
        else:
            print("❌ Database initialization failed")
    except Exception as e:
        print(f"❌ Database initialization error: {e}")

# CORS middleware for Android app
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# API LOGGING MIDDLEWARE
# ============================================================================

class APILoggingMiddleware(BaseHTTPMiddleware):
    """Middleware to log all API calls with status and timing"""
    
    async def dispatch(self, request: Request, call_next):
        from database import SessionLocal
        from database import APILog
        
        # Get user_id from request (if available)
        user_id = None
        
        # Try to extract user_id from URL path first
        # For paths like /api/v1/model/download/{user_id}
        path_parts = request.url.path.split('/')
        for part in path_parts:
            # Check if this looks like a user_id (not empty and not a number)
            if part and not part.isdigit() and part not in ['api', 'v1', 'users', 'register', 'sessions', 'upload', 'model', 'download', 'status', 'logs', 'feedback', 'analytics', 'home', 'weekly-usage', 'baseline', 'generate-sample', 'upload-custom', 'queries']:
                user_id = part
                break
        
        # If not found in path, try to get from request body (for POST requests)
        if not user_id and request.method in ["POST", "PUT"]:
            try:
                body = await request.body()
                if body:
                    try:
                        body_data = json.loads(body)
                        user_id = body_data.get("user_id")
                    except:
                        pass
                    # Reset the body so it can be read again
                    async def receive():
                        return {"type": "http.request", "body": body}
                    request._receive = receive
            except:
                pass
        
        # Call the next middleware/handler
        start_time = datetime.now()
        response = None
        status_code = 500
        error_message = None
        response_body = None
        
        try:
            response = await call_next(request)
            status_code = response.status_code
            
            # Capture response body for logging
            if user_id and not request.url.path.startswith("/api/v1/logs"):
                try:
                    response_body_bytes = b""
                    async for chunk in response.body_iterator:
                        response_body_bytes += chunk
                    
                    # Try to parse as JSON
                    try:
                        response_body = json.loads(response_body_bytes.decode())
                    except:
                        response_body = {"raw": response_body_bytes.decode()[:1000]}  # Limit to 1000 chars
                    
                    # Create new response with same body
                    from starlette.responses import Response
                    response = Response(
                        content=response_body_bytes,
                        status_code=response.status_code,
                        headers=dict(response.headers),
                        media_type=response.media_type
                    )
                except Exception as e:
                    print(f"⚠️  Failed to capture response body: {e}")
                    
        except Exception as e:
            error_message = str(e)
            raise
        finally:
            # Log the API call to database (but skip logging the logs endpoint itself)
            if user_id and not request.url.path.startswith("/api/v1/logs"):
                try:
                    db = SessionLocal()
                    api_log = APILog(
                        user_id=user_id,
                        endpoint=request.url.path,
                        method=request.method,
                        status_code=status_code,
                        response_body=response_body,
                        error_message=error_message,
                        created_at=start_time
                    )
                    db.add(api_log)
                    db.commit()
                    db.close()
                    print(f"✅ Logged API call: {request.method} {request.url.path} - Status: {status_code} - User: {user_id}")
                except Exception as e:
                    print(f"⚠️  Failed to log API call: {e}")
        
        return response

# Add the logging middleware
app.add_middleware(APILoggingMiddleware)

# Mount static files (API tester HTML)
static_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "static")
if os.path.exists(static_path):
    app.mount("/static", StaticFiles(directory=static_path), name="static")

# ============================================================================
# DATABASE INITIALIZATION
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Initialize database on app startup"""
    try:
        init_db()
        print("✅ Database initialized successfully")
    except Exception as e:
        print(f"⚠️ Database initialization: {e}")
        print("Note: Using PostgreSQL database for persistence")

# ============================================================================
# DATA MODELS
# ============================================================================

class Session(BaseModel):
    app_name: str
    start_time: str  # ISO format
    end_time: str
    duration_seconds: float
    num_vibrations: int = 0  # Number of vibrations in this session
    user_complied: bool = False  # Did user stop using app after vibration?
    group_id: int = 1  # Group ID for session grouping (defaults to 1)

class Query(BaseModel):
    group_id: int
    timestamp: str  # ISO format
    current_app: str
    state: List[int]  # JSON array of state values
    action: int  # 0 or 1
    compliance: int  # 0 or 1
    is_exploit: int = 0  # 0 = random/explore, 1 = Q-table exploit

class DailyUpload(BaseModel):
    user_id: str
    sessions: List[Session]
    queries: Optional[List[Query]] = []  # List of queries from device
    date: str  # The date these sessions are from (YYYY-MM-DD format)

class UserRegistration(BaseModel):
    user_id: str
    device_info: Optional[Dict] = None
    apps_to_monitor: Optional[List[str]] = None  # Apps user wants to monitor (empty list = all apps)
    is_test_mode: bool = False  # Random allocation for A/B testing

class ModelDownload(BaseModel):
    user_id: str
    day_number: int

class CustomBaselineStats(BaseModel):
    user_id: str
    total_sessions: int
    total_usage_time_minutes: int
    unique_apps: int
    most_used_app: str
    avg_session_duration: float
    peak_usage_hour: int

class SampleDataUpload(BaseModel):
    user_id: str
    baseline_stats: CustomBaselineStats
    sample_sessions: List[Session]
    agent_parameters: Dict
    q_table: Optional[Dict] = None

# ============================================================================
# Q-LEARNING AGENT (SERIALIZABLE FOR EDGE DEPLOYMENT)
# ============================================================================

def _default_q_values():
    """Separate function for defaultdict to allow pickling"""
    return [0.0, 0.0]

class EdgeQLearningAgent:
    """Lightweight Q-Learning agent that can be serialized to device"""
    
    def __init__(self):
        self.q_table = defaultdict(_default_q_values)
        self.alpha = 0.1
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_decay = 0.99
        self.epsilon_min = 0.1
        self.training_steps = 0
        
    def act(self, state: Tuple) -> int:
        """Device-side action selection"""
        if random.random() < self.epsilon:
            return random.randint(0, 1)
        return int(self.q_table[state][1] > self.q_table[state][0])
    
    def act_greedy(self, state: Tuple) -> int:
        """Greedy action for production (no exploration)"""
        return int(self.q_table[state][1] > self.q_table[state][0])
    
    def learn(self, state: Tuple, next_state: Tuple, action: int, reward: float):
        """Server-side learning"""
        best_next = max(self.q_table[next_state])
        td_target = reward + self.gamma * best_next
        td_error = td_target - self.q_table[state][action]
        self.q_table[state][action] += self.alpha * td_error
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)
        self.training_steps += 1
    
    def to_dict(self) -> Dict:
        """Convert to JSON-serializable format for device"""
        return {
            "q_table": {json.dumps(list(k)): v for k, v in dict(self.q_table).items()},
            "epsilon": self.epsilon,
            "training_steps": self.training_steps,
            "alpha": self.alpha,
            "gamma": self.gamma
        }
    
    def to_compact_binary(self) -> str:
        """Compact binary format for bandwidth optimization"""
        return base64.b64encode(pickle.dumps(self)).decode('utf-8')
    
    @staticmethod
    def from_dict(data: Dict):
        """Reconstruct from JSON"""
        agent = EdgeQLearningAgent()
        agent.q_table = defaultdict(_default_q_values)
        for k, v in data["q_table"].items():
            # Parse the key back to tuple - handle JSON array format
            try:
                # Try parsing as JSON array first (safer)
                key_list = json.loads(k) if k.startswith('[') else ast.literal_eval(k)
                if not isinstance(key_list, (list, tuple)):
                    continue
                agent.q_table[tuple(key_list)] = v
            except (ValueError, SyntaxError, TypeError) as e:
                print(f"Warning: Could not parse q_table key: {k}, error: {e}")
                continue
        agent.epsilon = data["epsilon"]
        agent.training_steps = data["training_steps"]
        agent.alpha = data.get("alpha", 0.1)
        agent.gamma = data.get("gamma", 0.95)
        return agent
    
    @staticmethod
    def from_compact_binary(data: str):
        """Reconstruct from binary"""
        return pickle.loads(base64.b64decode(data))
    
    @staticmethod
    def load_from_checkpoint(checkpoint):
        """
        Load agent from database checkpoint.
        This is the ONLY method that should be used to reconstruct an agent from the database.
        Ensures Q-table is properly loaded with all learned values.
        """
        agent = EdgeQLearningAgent()
        
        if not checkpoint:
            print("🆕 No checkpoint provided - starting with empty Q-table")
            return agent
        
        # Load Q-table from checkpoint
        agent.q_table = defaultdict(_default_q_values)
        loaded_keys = 0
        failed_keys = 0
        
        print(f"📥 Loading checkpoint with {len(checkpoint.q_table_json) if checkpoint.q_table_json else 0} states")
        
        if checkpoint.q_table_json:
            for k, v in checkpoint.q_table_json.items():
                try:
                    # Parse key as JSON array: "[0, 1, 1, 1]" -> [0, 1, 1, 1]
                    key_list = json.loads(k) if k.startswith('[') else ast.literal_eval(k)
                    
                    # Validate key is iterable
                    if not isinstance(key_list, (list, tuple)):
                        print(f"⚠️  Skipping invalid key type {type(key_list)}: {key_list}")
                        failed_keys += 1
                        continue
                    
                    # Store in Q-table
                    state_tuple = tuple(key_list)
                    agent.q_table[state_tuple] = v
                    loaded_keys += 1
                    
                except (ValueError, SyntaxError, TypeError, json.JSONDecodeError) as e:
                    print(f"⚠️  Failed to parse Q-table key '{k}': {e}")
                    failed_keys += 1
                    continue
        
        # Load hyperparameters
        agent.epsilon = checkpoint.epsilon
        agent.alpha = checkpoint.alpha
        agent.gamma = checkpoint.gamma
        agent.training_steps = checkpoint.training_step
        
        print(f"✅ Loaded agent: {loaded_keys} states, {failed_keys} failed")
        print(f"   Hyperparameters: ε={agent.epsilon:.4f}, α={agent.alpha}, γ={agent.gamma}, steps={agent.training_steps}")
        
        return agent

# ============================================================================
# DATABASE STORAGE (Using Neon PostgreSQL via SQLAlchemy)
# See database.py and db_service.py for database schema and operations
# ============================================================================

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def calculate_baseline_stats(sessions: List[Session], apps_to_monitor: List[str] = None) -> Dict:
    """
    EXACTLY matches original baseline calculation from SmartQuit.ipynb
    - calculate_median_target_app_usage: median of target app duration per GROUP
    - calculate_short_session_length: median of target app duration per GROUP (50th percentile)
    - calculate_75_percentile_length: 75th percentile of all session durations (for query interval)
    
    Uses apps_to_monitor from user table to determine target apps.
    If not provided, uses empty list (no apps monitored).
    
    Groups sessions by group_id and calculates median based on grouped target app usage.
    """
    if apps_to_monitor is None:
        apps_to_monitor = []
    
    if not sessions:
        # No sessions available — return explicit nulls so API consumers
        # can distinguish "no baseline" from a computed/default value.
        return {
            "median_target_app_usage_seconds": None,
            "median_session_usage_seconds": None,
            "query_interval_seconds": None
        }
    
    # Group sessions by group_id AND date (so sessions from different days aren't grouped together)
    groups = defaultdict(list)
    for session in sessions:
        # Use the date column from session data
        groups[(session.group_id, session.date)].append(session)
    
    # Calculate target app duration per group AND total duration per group
    group_target_durations = []
    group_total_durations = []
    all_session_durations = []
    
    for (group_id, date), group_sessions in groups.items():
        # Sum up target app durations in this group
        group_target_duration = sum(
            s.duration_seconds for s in group_sessions 
            if s.app_name in apps_to_monitor
        )
        if group_target_duration > 0:  # Only include groups with target app usage
            group_target_durations.append(group_target_duration)
        
        # Sum up ALL session durations in this group (for median_session_usage)
        group_total_duration = sum(s.duration_seconds for s in group_sessions)
        group_total_durations.append(group_total_duration)
        
        # Collect all individual session durations for 75th percentile calculation
        for s in group_sessions:
            all_session_durations.append(s.duration_seconds)
    
    # Calculate medians from grouped data
    if not group_target_durations:
        median_target_usage = None
    else:
        group_target_durations.sort()
        median_target_usage = group_target_durations[len(group_target_durations) // 2]
    
    if not group_total_durations:
        median_session_usage = None
    else:
        group_total_durations.sort()
        median_session_usage = group_total_durations[len(group_total_durations) // 2]
    
    # 75th percentile for query interval (from all individual sessions)
    if not all_session_durations:
        percentile_75 = None
    else:
        all_session_durations.sort()
        percentile_75_index = int(len(all_session_durations) * 0.75)
        percentile_75 = all_session_durations[percentile_75_index] if percentile_75_index < len(all_session_durations) else all_session_durations[-1]
    
    return {
        "median_target_app_usage_seconds": median_target_usage,
        "median_session_usage_seconds": median_session_usage,
        "query_interval_seconds": percentile_75
    }

def get_first_app_in_group(grouped_session: Dict) -> str:
    """Get the first app in a grouped session"""
    if grouped_session.get('sessions') and len(grouped_session['sessions']) > 0:
        return grouped_session['sessions'][0]['app_name']
    return None

def extract_state_from_first_app(first_app: str, group: Dict, baseline_stats: Dict, apps_to_monitor: List[str] = None) -> Tuple:
    """Extract state from grouped session for learning
    
    State: (num_vibrations, is_target_app, day_quarter, is_weekday)
    
    Uses apps_to_monitor from user table to determine if app is target.
    If not provided, uses empty list (no apps monitored).
    """
    if apps_to_monitor is None:
        apps_to_monitor = []
    
    timestamp = group['start_time']
    
    # Handle timezone-aware timestamps properly
    # Convert to UTC for consistent time-of-day processing across all users
    if hasattr(timestamp, 'tzinfo') and timestamp.tzinfo is not None:
        # Timezone-aware timestamp - convert to UTC
        from datetime import timezone
        timestamp = timestamp.astimezone(timezone.utc)
    # If timezone-naive, assume it's already in the intended timezone
    
    hour = timestamp.hour
    if 0 <= hour < 6:
        day_quarter = 0  # Night/Early morning (0-6)
    elif 6 <= hour < 12:
        day_quarter = 1  # Morning (6-12)
    elif 12 <= hour < 18:
        day_quarter = 2  # Afternoon (12-18)  
    else:
        day_quarter = 3  # Evening (18-24)
    
    is_weekday = int(timestamp.weekday() < 5)
    is_target_app = int(first_app in apps_to_monitor)
    num_vibrations = group.get('total_vibrations', 0)
    
    return (num_vibrations, is_target_app, day_quarter, is_weekday)

def calculate_reward(action: int, compliance: int) -> float:
    """
    Calculate reward based on action and compliance.
    
    Simplified reward logic:
    - action == 0 (no vibration): reward = 0
    - action == 1 (vibration):
        - compliance == 1 (user complied): reward = 1
        - compliance == 0 (user did not comply): reward = -1
    """
    if action == 0:
        return 0.0
    elif action == 1:
        return 1.0 if compliance == 1 else -1.0
    else:
        return 0.0

def extract_state(session: Session, num_vibrations: int, baseline_stats: Dict, apps_to_monitor: List[str] = None) -> Tuple:
    """Extract state features for Q-learning
    
    State: (num_vibrations, is_target_app, day_quarter, is_weekday)
    
    Uses apps_to_monitor from user table to determine if app is target.
    If not provided, uses empty list (no apps monitored).
    """
    if apps_to_monitor is None:
        apps_to_monitor = []
    
    # Handle both string (from API) and datetime (from database) inputs
    if isinstance(session.start_time, str):
        timestamp = datetime.fromisoformat(session.start_time)
    else:
        timestamp = session.start_time  # Already a datetime object
    
    # Time of day quarter (0-3)
    hour = timestamp.hour
    if 0 <= hour < 6:
        day_quarter = 0
    elif 6 <= hour < 12:
        day_quarter = 1
    elif 12 <= hour < 18:
        day_quarter = 2
    else:
        day_quarter = 3
    
    # Is weekday
    is_weekday = int(timestamp.weekday() < 5)
    
    # Is target app
    is_target_app = int(session.app_name in apps_to_monitor)
    
    return (num_vibrations, is_target_app, day_quarter, is_weekday)

# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.post("/api/v1/users/register")
async def register_user(registration: UserRegistration, db: SQLSession = Depends(get_db)):
    """Register a new user and initialize their model"""
    print(f"\n📱 REGISTRATION REQUEST RECEIVED:")
    print(f"  User ID: {registration.user_id}")
    print(f"  Device Info: {registration.device_info}")
    print(f"  Apps to Monitor (raw): {registration.apps_to_monitor}")
    print(f"  Apps to Monitor Type: {type(registration.apps_to_monitor)}")
    print(f"  Apps is None: {registration.apps_to_monitor is None}")
    print(f"  Apps is empty: {registration.apps_to_monitor == [] if registration.apps_to_monitor is not None else 'N/A'}")
    print(f"  Test Mode: {registration.is_test_mode}")
    
    if DatabaseService.user_exists(db, registration.user_id):
        raise HTTPException(status_code=400, detail="User already exists")
    
    # Use provided apps or empty list if not specified
    apps_to_monitor = registration.apps_to_monitor if registration.apps_to_monitor else []
    
    print(f"  Final apps_to_monitor to save: {apps_to_monitor}")
    
    user = DatabaseService.create_user(db, registration.user_id, registration.device_info, apps_to_monitor, registration.is_test_mode)
    
    # Create basic Q-table model for new user (proper RL initialization with zeros)
    try:
        basic_qtable = generate_basic_qtable_for_new_user(registration.user_id)
        DatabaseService.save_model_checkpoint(
            db=db,
            user_id=registration.user_id,
            training_step=0,
            epsilon=0.9,  # High exploration for new user
            alpha=0.1,
            gamma=0.95,
            q_table=basic_qtable
        )
        print(f"  ✅ Basic Q-table model created for new user (32 states, all zeros)")
    except Exception as e:
        print(f"  ⚠️  Warning: Could not create basic model for new user: {e}")
    
    print(f"  ✅ User created successfully")
    print(f"  Saved apps_to_monitor: {user.apps_to_monitor}\n")
    
    return {
        "status": "success",
        "user_id": registration.user_id,
        "apps_to_monitor": apps_to_monitor,
        "is_test_mode": registration.is_test_mode,
        "message": f"User registered in {'TEST' if registration.is_test_mode else 'PRODUCTION'} mode. Upload data to start model training (interventions begin immediately if you have existing baseline stats)."
    }

@app.post("/api/v1/sessions/upload")
async def upload_daily_sessions(
    batch: DailyUpload,
    db: SQLSession = Depends(get_db)
):
    """
    Upload daily session data from device.
    Day number is based on user's current_day field in database (incremented on each upload).
    
    Baseline Logic (1-indexed, first upload = Day 1):
    - If user has existing baseline stats: Start interventions immediately  
    - If no baseline stats: Day 1 = collect data, Day 2 = calculate baseline, Day 3+ = intervention
    
    Training: Model trains daily from day 1 regardless of baseline status
    Device should call this every day at 3 AM with yesterday's sessions
    """
    if not DatabaseService.user_exists(db, batch.user_id):
        raise HTTPException(status_code=404, detail="User not found")
    
    try:
        # Validate date format
        print(f"📥 Upload received: User={batch.user_id}, Date={batch.date}, Sessions={len(batch.sessions)}")
        print(f"   Date format: {batch.date}")
        if len(batch.sessions) > 0:
            print(f"   Sample session start_time: {batch.sessions[0].start_time} (type: {type(batch.sessions[0].start_time).__name__}, length: {len(str(batch.sessions[0].start_time))})")
            print(f"   Sample session end_time: {batch.sessions[0].end_time} (length: {len(str(batch.sessions[0].end_time))})")
            
            # Add timezone parsing verification
            try:
                parsed_start = datetime.fromisoformat(batch.sessions[0].start_time)
                print(f"   ✅ Parsed start_time: {parsed_start} (has timezone: {parsed_start.tzinfo is not None})")
                if parsed_start.tzinfo:
                    from datetime import timezone
                    utc_time = parsed_start.astimezone(timezone.utc)
                    print(f"   ✅ UTC equivalent: {utc_time} (hour: {utc_time.hour})")
            except Exception as e:
                print(f"   ❌ Error parsing sample timestamp: {e}")
        
        # Check if baseline stats already exist for this user
        existing_baseline_stats = DatabaseService.get_baseline_stats(db, batch.user_id)
        baseline_exists = existing_baseline_stats is not None
        
        # Get user info for test mode and current day
        user = DatabaseService.get_user(db, batch.user_id)
        is_test_mode = user.is_test_mode if user else False

        # Use user's current_day as source of truth, increment for this upload
        day_number = user.current_day + 1
        
        print(f"✅ User {batch.user_id} day progression: {user.current_day} → {day_number}")
        print(f"✅ Baseline stats exist: {baseline_exists}")
        print(f"✅ User test mode: {is_test_mode}")

        # Save sessions to database
        DatabaseService.save_sessions(db, batch.user_id, batch.date, batch.sessions)

        # Save queries to database (if provided)
        queries_count = 0
        if batch.queries and len(batch.queries) > 0:
            try:
                DatabaseService.save_queries(db, batch.user_id, batch.date, batch.queries)
                queries_count = len(batch.queries)
                print(f"✅ Saved {queries_count} queries for user {batch.user_id}")
            except Exception as e:
                print(f"⚠️ Failed to save queries: {e}")
                # Don't fail the entire upload if queries fail

        response = {
            "status": "received",
            "sessions_count": len(batch.sessions),
            "queries_count": queries_count,
            "day_number": day_number,
            "date": batch.date,
            "baseline_exists": baseline_exists,
            "is_test_mode": is_test_mode
        }
        # Default baseline_stats to None unless we compute or fetch them
        response["baseline_stats"] = None

        # Handle baseline stats logic
        mode_suffix = f" [{('TEST' if is_test_mode else 'PRODUCTION')} mode]"
        if baseline_exists:
            # Skip baseline period - user already has baseline stats
            response["message"] = f"Baseline stats exist - starting intervention period immediately. Training with day {day_number} data.{mode_suffix}"
            response["baseline_stats"] = {
                "median_target_app_usage_seconds": existing_baseline_stats.median_target_app_usage_seconds,
                "median_session_usage_seconds": existing_baseline_stats.median_session_usage_seconds,
                "query_interval_seconds": existing_baseline_stats.query_interval_seconds
            }
        elif day_number == 2:
            # Calculate baseline stats on second upload (day 2)
            baseline_sessions = DatabaseService.get_baseline_sessions(db, batch.user_id)
            print(f"Calculating baseline stats from {len(baseline_sessions)} sessions (1st and 2nd upload)")

            # Get user's apps_to_monitor for baseline calculation
            user = DatabaseService.get_user(db, batch.user_id)
            user_apps = user.apps_to_monitor if user and user.apps_to_monitor else []

            stats = calculate_baseline_stats(baseline_sessions, user_apps)
            DatabaseService.save_baseline_stats(db, batch.user_id, stats)
            response["baseline_stats"] = stats
            response["message"] = f"Baseline period completed (2nd upload). Model training continues daily.{mode_suffix}"
            print(f"Baseline stats: {stats}")
        else:
            response["message"] = f"Day {day_number} recorded. Baseline stats will be calculated on day 2 (2nd upload).{mode_suffix}"

        # Train model daily regardless of baseline/intervention period
        training_result = None
        if day_number >= 1:  # Start training from day 1 (first upload)
            print(f"Starting synchronous training for user {batch.user_id}, day {day_number}")

            # Run training synchronously to ensure model is updated before response
            try:
                training_result = await train_model_daily(batch.user_id, batch.date, batch.queries if batch.queries else [], db)
                print(f"✅ Training completed for user {batch.user_id}: {training_result}")

                if training_result.get("status") == "skipped":
                    # Training was skipped due to no queries
                    response["model_training"] = {
                        "status": "skipped",
                        "reason": training_result.get("reason", "no_queries_to_learn_from"),
                        "learned_transitions": 0,
                        "q_table_size": 0,
                        "training_steps": 0,
                        "checkpoint_saved": False
                    }
                    response["message"] = f"Day {day_number} recorded. No queries to learn from - training skipped.{mode_suffix}"
                else:
                    # Normal training completed
                    response["model_training"] = {
                        "status": "completed",
                        "learned_transitions": training_result.get("learned_transitions", 0),
                        "q_table_size": training_result.get("q_table_size", 0),
                        "training_steps": training_result.get("training_steps", 0),
                        "checkpoint_saved": training_result.get("checkpoint_saved", False)
                    }

                    # Include updated Q-table and metadata in response only if training actually happened
                    response["updated_model"] = {
                        "q_table": training_result.get("q_table", {}),
                        "metadata": training_result.get("model_metadata", {})
                    }

                    # Determine if we're in intervention period based on baseline existence or day number  
                    mode_suffix = f" [{('TEST' if is_test_mode else 'PRODUCTION')} mode]"
                    if baseline_exists:
                        response["message"] = f"Training completed with day {day_number} data (intervention period - baseline exists). Model updated and ready for download.{mode_suffix}"
                    elif day_number >= 3:
                        response["message"] = f"Training completed with day {day_number} data (intervention period). Model updated and ready for download.{mode_suffix}"
                    elif day_number == 2:
                        response["message"] = f"Training completed with day {day_number} data (baseline period completed). Model updated and ready for download.{mode_suffix}"
                    else:
                        response["message"] = f"Training completed with day {day_number} data (baseline period). Model updated and ready for download.{mode_suffix}"

            except Exception as e:
                print(f"❌ Training failed for user {batch.user_id}: {e}")
                response["model_training"] = {
                    "status": "failed",
                    "error": str(e),
                    "learned_transitions": 0,
                    "q_table_size": 0,
                    "training_steps": 0,
                    "checkpoint_saved": False
                }
                response["message"] = f"Session upload successful but model training failed for day {day_number}. Error: {str(e)}"

        # Update user's current day number
        DatabaseService.update_user_day(db, batch.user_id, day_number)

        return response
    
    except ValueError as e:
        print(f"❌ Validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        print(f"❌ Unexpected error: {type(e).__name__}: {e}")
        raise HTTPException(status_code=500, detail=f"Server error: {type(e).__name__}: {str(e)}")

@app.get("/api/v1/model/download/{user_id}")
async def download_model(user_id: str, format: str = "binary", db: SQLSession = Depends(get_db)):
    """
    Download trained model for on-device inference.
    Device should call this every day at 3 AM after uploading sessions.
    Format: 'json' or 'binary' (binary is smaller)
    """
    if not DatabaseService.user_exists(db, user_id):
        raise HTTPException(status_code=404, detail="User not found")
    
    # Get user's apps_to_monitor
    user = DatabaseService.get_user(db, user_id)
    apps_to_monitor = user.apps_to_monitor if user and user.apps_to_monitor else []
    current_day = user.current_day if user else 0  # Get current day for frontend
    
    baseline_stats = DatabaseService.get_baseline_stats(db, user_id)
    
    # Allow access even without baseline stats for viewing purposes
    baseline_data = {}
    if baseline_stats:
        baseline_data = {
            "median_target_app_usage_seconds": baseline_stats.median_target_app_usage_seconds,
            "median_session_usage_seconds": baseline_stats.median_session_usage_seconds,
            "query_interval_seconds": baseline_stats.query_interval_seconds
        }
    else:
        # No baseline stats — return explicit nulls so client can detect absence
        baseline_data = {
            "median_target_app_usage_seconds": None,
            "median_session_usage_seconds": None,
            "query_interval_seconds": None
        }
    
    # Get latest model checkpoint
    checkpoint = DatabaseService.get_latest_model(db, user_id)

    model_data = {
        "user_id": user_id,
        "current_day": current_day,  # Add current_day for frontend
        "model_version": checkpoint.training_step if checkpoint else 0,
        "updated_at": checkpoint.created_at.isoformat() if checkpoint else datetime.now().isoformat(),
        "baseline_stats": baseline_data,
        "apps_to_monitor": apps_to_monitor,
    }
    
    if format == "binary" and checkpoint and checkpoint.model_binary:
        model_data["agent_data"] = checkpoint.model_binary.hex()
        model_data["format"] = "binary"
    else:
        # Return Q-table as JSON
        q_table = checkpoint.q_table_json if checkpoint else {}
        model_data["agent_data"] = q_table
        model_data["format"] = "json"
        model_data["q_table_info"] = {
            "states_count": len(q_table) if q_table else 0,
            "has_q_table": bool(q_table),
            "training_step": checkpoint.training_step if checkpoint else 0,
            "epsilon": checkpoint.epsilon if checkpoint else 0.9,
            "alpha": checkpoint.alpha if checkpoint else 0.1,
            "gamma": checkpoint.gamma if checkpoint else 0.95
        }
    
    return model_data

@app.get("/api/v1/model/status/{user_id}")
async def get_model_status(user_id: str, db: SQLSession = Depends(get_db)):
    """Check if model is ready for download and get current day with detailed training info"""
    if not DatabaseService.user_exists(db, user_id):
        raise HTTPException(status_code=404, detail="User not found")
    
    user = DatabaseService.get_user(db, user_id)
    baseline_stats = DatabaseService.get_baseline_stats(db, user_id)
    checkpoint = DatabaseService.get_latest_model(db, user_id)
    
    # Get recent training history to show model update activity
    training_logs = DatabaseService.get_training_history(db, user_id, limit=5)
    recent_training = []
    for log in training_logs:
        recent_training.append({
            "date": log.date,
            "reward": log.reward,
            "action": log.action,
            "q_value_change": log.q_value_after - log.q_value_before,
            "updated_at": log.created_at.isoformat() if log.created_at else None
        })
    
    return {
        "user_id": user_id,
        "current_day": user.current_day,
        "baseline_completed": baseline_stats is not None,
        "model_ready": baseline_stats is not None and checkpoint is not None and checkpoint.training_step > 0,
        "model_info": {
            "training_steps": checkpoint.training_step if checkpoint else 0,
            "epsilon": checkpoint.epsilon if checkpoint else 1.0,
            "q_table_size": len(checkpoint.q_table_json) if checkpoint and checkpoint.q_table_json else 0,
            "last_model_update": checkpoint.created_at.isoformat() if checkpoint else None,
            "alpha": checkpoint.alpha if checkpoint else 0.1,
            "gamma": checkpoint.gamma if checkpoint else 0.95
        },
        "recent_training_activity": recent_training,
        "training_active": len(recent_training) > 0,
        "model_update_confirmed": checkpoint is not None and checkpoint.training_step > 0
    }

@app.post("/api/v1/feedback/upload")
async def upload_feedback(
    user_id: str,
    state: List[int],
    next_state: List[int],
    action: int,
    reward: float
):
    """
    Optional: Upload feedback from device for online learning.
    This allows the model to continue improving on-device experiences.
    """
    if user_id not in store.users:
        raise HTTPException(status_code=404, detail="User not found")
    
    agent = store.get_agent(user_id)
    agent.learn(tuple(state), tuple(next_state), action, reward)
    
    return {
        "status": "learned",
        "training_steps": agent.training_steps,
        "epsilon": agent.epsilon
    }

@app.get("/api/v1/logs/{user_id}")
async def get_api_logs(user_id: str, limit: int = 100, offset: int = 0, db: SQLSession = Depends(get_db)):
    """
    Get API call logs for a user
    Shows all API requests made by the user with their status and timestamp
    """
    from database import APILog
    
    if not DatabaseService.user_exists(db, user_id):
        raise HTTPException(status_code=404, detail="User not found")
    
    # Query API logs for this user, sorted by most recent first
    logs = db.query(APILog)\
        .filter(APILog.user_id == user_id)\
        .order_by(APILog.created_at.desc())\
        .offset(offset)\
        .limit(limit)\
        .all()
    
    # Format logs for response
    log_list = []
    for log in logs:
        log_list.append({
            "id": log.id,
            "endpoint": log.endpoint,
            "method": log.method,
            "status_code": log.status_code,
            "response_body": log.response_body,
            "error_message": log.error_message,
            "created_at": log.created_at.isoformat() if log.created_at else None,
            "updated_at": log.updated_at.isoformat() if log.updated_at else None
        })
    
    return {
        "user_id": user_id,
        "total_logs": db.query(APILog).filter(APILog.user_id == user_id).count(),
        "returned": len(log_list),
        "offset": offset,
        "limit": limit,
        "logs": log_list
    }

@app.get("/api/v1/analytics/{user_id}")
async def get_analytics(user_id: str, db: SQLSession = Depends(get_db)):
    """Get usage analytics for the user"""
    if not DatabaseService.user_exists(db, user_id):
        raise HTTPException(status_code=404, detail="User not found")
    
    sessions = DatabaseService.get_all_sessions(db, user_id)
    training_logs = DatabaseService.get_training_history(db, user_id, limit=100)
    baseline_stats = DatabaseService.get_baseline_stats(db, user_id)
    checkpoint = DatabaseService.get_latest_model(db, user_id)
    
    total_time = sum(s.duration_seconds for s in sessions)
    target_time = sum(s.duration_seconds for s in sessions if s.is_target_app)
    
    # Calculate compliance rate and vibrations
    vibration_sessions = [s for s in sessions if s.num_vibrations > 0]
    total_vibrations_count = sum(s.num_vibrations for s in sessions)
    compliance_rate = 0
    if vibration_sessions:
        complied = sum(1 for s in vibration_sessions if s.user_complied)
        compliance_rate = (complied / len(vibration_sessions)) * 100
    
    return {
        "user_id": user_id,
        "current_day": DatabaseService.get_user(db, user_id).current_day,
        "total_sessions": len(sessions),
        "total_time_hours": total_time / 3600,
        "social_media_time_hours": target_time / 3600,
        "compliance_rate": compliance_rate,
        "total_vibrations": total_vibrations_count,
        "complied_vibrations": sum(1 for s in vibration_sessions if s.user_complied),
        "baseline_stats": {
            "median_target_app_usage_seconds": baseline_stats.median_target_app_usage_seconds,
            "median_session_usage_seconds": baseline_stats.median_session_usage_seconds,
            "query_interval_seconds": baseline_stats.query_interval_seconds
        } if baseline_stats else None,
        "model_stats": {
            "training_steps": checkpoint.training_step if checkpoint else 0,
            "q_table_states": len(checkpoint.q_table_json) if checkpoint else 0,
            "last_update": checkpoint.created_at.isoformat() if checkpoint else None
        },
        "recent_training_updates": [
            {
                "date": log.date,
                "reward": log.reward,
                "action": log.action,
                "q_value_delta": log.q_value_after - log.q_value_before
            }
            for log in training_logs[:10]
        ]
    }

@app.get("/api/v1/home/weekly-usage/{user_id}")
async def get_weekly_target_app_usage(user_id: str, db: SQLSession = Depends(get_db)):
    """
    Get cumulative target app usage for the last 7 days.
    Returns all available data within the 7-day window (today to 6 days ago).
    """
    if not DatabaseService.user_exists(db, user_id):
        raise HTTPException(status_code=404, detail="User not found")
    
    # Get user's apps to monitor (target apps)
    user = DatabaseService.get_user(db, user_id)
    apps_to_monitor = user.apps_to_monitor if user and user.apps_to_monitor else []
    
    # Calculate date range for last 7 days
    today = datetime.now().date()
    max_week_ago = today - timedelta(days=6)  # 7 days total: today and 6 days before
    
    if not apps_to_monitor:
        return {
            "user_id": user_id,
            "period_days": 0,
            "max_period_days": 7,
            "date_range": {
                "start_date": max_week_ago.isoformat(),
                "end_date": today.isoformat()
            },
            "apps_to_monitor": [],
            "daily_usage": [],
            "per_app_usage": {},
            "total_usage_seconds": 0,
            "total_usage_formatted": "0s",
            "message": "No target apps configured"
        }
    
    # Get all sessions for this user
    all_sessions = DatabaseService.get_all_sessions(db, user_id)
    
    # Filter sessions to last 7 days and target apps only
    daily_usage = {}  # {date: {app: seconds}}
    per_app_totals = {}  # {app: total_seconds}
    
    for session in all_sessions:
        # Parse session date
        try:
            session_date = datetime.strptime(session.date, "%Y-%m-%d").date()
        except (ValueError, TypeError):
            continue
        
        # Check if within last 7 days (maximum)
        if session_date < max_week_ago or session_date > today:
            continue
        
        # Check if target app
        if session.app_name not in apps_to_monitor:
            continue
        
        date_str = session.date
        app_name = session.app_name
        duration = session.duration_seconds or 0
        
        # Aggregate by date and app
        if date_str not in daily_usage:
            daily_usage[date_str] = {}
        if app_name not in daily_usage[date_str]:
            daily_usage[date_str][app_name] = 0
        daily_usage[date_str][app_name] += duration
        
        # Aggregate per-app totals
        if app_name not in per_app_totals:
            per_app_totals[app_name] = 0
        per_app_totals[app_name] += duration
    
    # Calculate grand total
    total_seconds = sum(per_app_totals.values())
    
    # Date range is always last 7 days from today
    # period_days reflects how many days actually have data
    available_dates = sorted(daily_usage.keys())
    actual_days = len(available_dates)
    
    # Format daily usage for response (sorted by date)
    daily_usage_list = []
    for date_str in sorted(daily_usage.keys()):
        day_total = sum(daily_usage[date_str].values())
        daily_usage_list.append({
            "date": date_str,
            "apps": daily_usage[date_str],
            "total_seconds": day_total,
            "total_formatted": format_duration(day_total)
        })
    
    # Format per-app usage (sorted by usage descending)
    per_app_formatted = {}
    for app, seconds in sorted(per_app_totals.items(), key=lambda x: x[1], reverse=True):
        per_app_formatted[app] = {
            "total_seconds": seconds,
            "total_formatted": format_duration(seconds)
        }
    
    return {
        "user_id": user_id,
        "period_days": actual_days,
        "max_period_days": 7,
        "date_range": {
            "start_date": max_week_ago.isoformat(),
            "end_date": today.isoformat()
        },
        "apps_to_monitor": apps_to_monitor,
        "daily_usage": daily_usage_list,
        "per_app_usage": per_app_formatted,
        "total_usage_seconds": total_seconds,
        "total_usage_formatted": format_duration(total_seconds)
    }

def format_duration(total_seconds: float) -> str:
    """Format duration in seconds to human-readable format (e.g., '1h 23m', '45m', '30s')"""
    total_seconds = int(total_seconds)
    if total_seconds < 60:
        return f"{total_seconds}s"
    elif total_seconds < 3600:
        minutes = total_seconds // 60
        seconds = total_seconds % 60
        if seconds > 0:
            return f"{minutes}m {seconds}s"
        return f"{minutes}m"
    else:
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        if minutes > 0:
            return f"{hours}h {minutes}m"
        return f"{hours}h"

@app.get("/api/v1/queries/{user_id}")
async def get_queries(user_id: str, date: Optional[str] = None, limit: int = 100, offset: int = 0, db: SQLSession = Depends(get_db)):
    """
    Get query logs for a user
    Shows all queries made by the device with their state, action, and compliance
    """
    if not DatabaseService.user_exists(db, user_id):
        raise HTTPException(status_code=404, detail="User not found")
    
    # Get queries (optionally filtered by date)
    if date:
        queries = DatabaseService.get_queries(db, user_id, date)
    else:
        queries = DatabaseService.get_all_queries(db, user_id)
    
    # Apply pagination
    total_queries = len(queries)
    queries = queries[offset:offset + limit]
    
    # Format queries for response
    query_list = []
    for query in queries:
        query_list.append({
            "id": query.id,
            "group_id": query.group_id,
            "date": query.date,
            "timestamp": query.timestamp.isoformat() if query.timestamp else None,
            "current_app": query.current_app,
            "state": query.state,
            "action": query.action,
            "compliance": query.compliance,
            "created_at": query.created_at.isoformat() if query.created_at else None
        })
    
    return {
        "user_id": user_id,
        "total_queries": total_queries,
        "returned": len(query_list),
        "offset": offset,
        "limit": limit,
        "date_filter": date,
        "queries": query_list
    }

def generate_realistic_qtable(user_id: str) -> dict:
    """Generate a Q-table with proper RL initialization (not pre-learned values)"""
    import random
    random.seed(hash(user_id) % 1000)  # User-specific but reproducible
    
    qtable = {}
    
    # New state space: num_queries (0-10), num_vibrations (0-5), is_target_app (0,1), day_quarter (0-3)
    for num_queries in range(0, 11):
        for num_vibrations in range(0, 6):
            for is_target_app in [0, 1]:
                for day_quarter in range(0, 4):
                    state = (num_queries, num_vibrations, is_target_app, day_quarter)
                    
                    # Using small random initialization around 0 (common practice)
                    no_vibrate_q = round(random.uniform(-0.1, 0.1), 3)
                    vibrate_q = round(random.uniform(-0.1, 0.1), 3)
                    
                    qtable[str(list(state))] = [no_vibrate_q, vibrate_q]
    
    # Add terminal state with zero values (unreachable state)
    terminal_state = (-1, -1, -1, -1)
    qtable[str(list(terminal_state))] = [0.0, 0.0]
    
    return qtable

def generate_learned_qtable(user_id: str) -> dict:
    """Generate a Q-table that simulates what it might look like after training"""
    import random
    random.seed(hash(user_id) % 1000)  # User-specific but reproducible
    
    qtable = {}
    
    # New state space: num_queries (0-10), num_vibrations (0-5), is_target_app (0,1), day_quarter (0-3)
    for num_queries in range(0, 11):
        for num_vibrations in range(0, 6):
            for is_target_app in [0, 1]:
                for day_quarter in range(0, 4):
                    state = (num_queries, num_vibrations, is_target_app, day_quarter)
                    
                    # Simulate learned Q-values based on expected reward patterns
                    intervention_score = 0.0
                    if is_target_app == 1: intervention_score += 0.4
                    if num_vibrations >= 3: intervention_score += 0.3
                    if day_quarter == 2: intervention_score += 0.2  # Afternoon
                    elif day_quarter == 3: intervention_score += 0.3  # Evening
                    elif day_quarter == 1: intervention_score += 0.1  # Morning
                    if num_queries >= 5: intervention_score += 0.2
                    
                    # Learned values after training (what algorithm might discover)
                    if intervention_score >= 0.7:
                        no_vibrate_q = round(random.uniform(-0.6, -0.2), 3)
                        vibrate_q = round(random.uniform(0.6, 1.0), 3)
                    elif intervention_score >= 0.4:
                        no_vibrate_q = round(random.uniform(-0.3, 0.1), 3)
                        vibrate_q = round(random.uniform(0.3, 0.7), 3)
                    elif intervention_score >= 0.2:
                        no_vibrate_q = round(random.uniform(0.0, 0.4), 3)
                        vibrate_q = round(random.uniform(0.2, 0.5), 3)
                    else:
                        no_vibrate_q = round(random.uniform(0.4, 0.8), 3)
                        vibrate_q = round(random.uniform(-0.2, 0.2), 3)
                    
                    qtable[str(list(state))] = [no_vibrate_q, vibrate_q]
    
    # Add terminal state with zero values (unreachable state)
    terminal_state = (-1, -1, -1, -1)
    qtable[str(list(terminal_state))] = [0.0, 0.0]
    
    return qtable

def generate_basic_qtable_for_new_user(user_id: str) -> dict:
    """Generate a basic Q-table with zero initialization for new users"""
    qtable = {}
    # New state space: num_queries (0-10), num_vibrations (0-5), is_target_app (0,1), day_quarter (0-3)
    for num_queries in range(0, 11):
        for num_vibrations in range(0, 6):
            for is_target_app in [0, 1]:
                for day_quarter in range(0, 4):
                    state = (num_queries, num_vibrations, is_target_app, day_quarter)
                    qtable[str(list(state))] = [0.0, 0.0]
    
    # Add terminal state with zero values (unreachable state)
    terminal_state = (-1, -1, -1, -1)
    qtable[str(list(terminal_state))] = [0.0, 0.0]
    
    return qtable

@app.post("/api/v1/baseline/generate-sample")
async def generate_sample_data(request_data: Optional[Dict] = None):
    """Generate sample baseline stats and session data for testing"""
    from datetime import datetime, timedelta
    import random
    
    # Parse request parameters with defaults
    if request_data is None:
        request_data = {}
    
    user_id = request_data.get('user_id')
    days = request_data.get('days', 7)
    sessions_per_day = request_data.get('sessions_per_day', 8)
    
    # Validate parameters
    days = max(1, min(30, days))  # Between 1-30 days
    sessions_per_day = max(1, min(20, sessions_per_day))  # Between 1-20 sessions per day
    
    # Generate a sample user ID if not provided
    sample_user_id = user_id if user_id else f"sample_user_{random.randint(1000, 9999)}"
    
    # Sample apps
    apps = ["Instagram", "TikTok", "Facebook", "Twitter", "YouTube", "Netflix", "Spotify", "Chrome", "WhatsApp", "Slack"]
    target_apps = ["Instagram", "TikTok", "Facebook", "Twitter", "YouTube"]
    
    # Generate baseline stats
    total_sessions = random.randint(50, 200)
    total_usage_minutes = random.randint(300, 800)  # 5-13 hours
    unique_apps = random.randint(5, 10)
    most_used_app = random.choice(target_apps)
    avg_session_duration = total_usage_minutes * 60 / total_sessions  # in seconds
    peak_usage_hour = random.randint(9, 22)  # 9 AM to 10 PM
    
    baseline_stats = CustomBaselineStats(
        user_id=sample_user_id,
        total_sessions=total_sessions,
        total_usage_time_minutes=total_usage_minutes,
        unique_apps=unique_apps,
        most_used_app=most_used_app,
        avg_session_duration=avg_session_duration,
        peak_usage_hour=peak_usage_hour
    )
    
    # Generate sample sessions
    sample_sessions = []
    base_date = datetime.now() - timedelta(days=days)
    
    for day in range(days):
        current_date = base_date + timedelta(days=day)
        sessions_today = random.randint(max(1, sessions_per_day - 3), sessions_per_day + 3)
        
        for session_num in range(sessions_today):
            app = random.choice(apps)
            is_target_app = app in target_apps
            
            # Generate session times
            hour = random.randint(8, 23)
            minute = random.randint(0, 59)
            start_time = current_date.replace(hour=hour, minute=minute, second=0, microsecond=0)
            
            # Duration varies by app type
            if is_target_app:
                duration = random.randint(60, 1800)  # 1-30 minutes for target apps
            else:
                duration = random.randint(30, 600)   # 30 seconds - 10 minutes for others
            
            end_time = start_time + timedelta(seconds=duration)
            
            # Simulate vibrations and compliance for target apps
            num_vibrations = 0
            user_complied = False
            if is_target_app and duration > 300:  # Vibrate if session > 5 minutes
                num_vibrations = random.randint(1, 3)
                user_complied = random.choice([True, False])
            
            session = Session(
                app_name=app,
                start_time=start_time.isoformat(),
                end_time=end_time.isoformat(),
                duration_seconds=duration,
                num_vibrations=num_vibrations,
                user_complied=user_complied,
                group_id=random.randint(1, 5)
            )
            sample_sessions.append(session)
    
    # Generate agent parameters
    agent_parameters = {
        "alpha": 0.1,
        "gamma": 0.95,
        "epsilon": random.uniform(0.1, 0.5),
        "training_steps": random.randint(0, 1000),
        "q_table_size": random.randint(10, 100)
    }
    
    # Generate Q-table (proper RL initialization by default, learned version optional)
    q_table_type = request_data.get('q_table_type', 'initialized')  # 'initialized' or 'learned'
    
    if q_table_type == 'learned':
        q_table = generate_learned_qtable(sample_user_id)
        q_table_description = "Simulated post-training Q-table with learned values"
    else:
        q_table = generate_realistic_qtable(sample_user_id)
        q_table_description = "Properly initialized Q-table (small random values around 0)"
    
    return {
        "user_id": sample_user_id,
        "baseline_stats": baseline_stats.dict(),
        "sample_sessions": [session.dict() for session in sample_sessions],
        "q_table": q_table,
        "agent_parameters": agent_parameters,
        "summary": {
            "total_sessions": len(sample_sessions),
            "days_covered": days,
            "target_app_sessions": sum(1 for s in sample_sessions if s.app_name in target_apps),
            "total_vibrations": sum(s.num_vibrations for s in sample_sessions),
            "compliance_rate": sum(1 for s in sample_sessions if s.user_complied and s.num_vibrations > 0),
            "q_table_states": len(q_table),  # Should be 32 states
            "q_table_actions": 2,  # vibrate vs no_vibrate
            "q_table_type": q_table_type,
            "q_table_description": q_table_description,
            "state_variables": {
                "monitored_app": "0=not_monitored, 1=monitored",
                "session_length": "0=short, 1=long", 
                "time_of_day": "0=morning, 1=afternoon, 2=evening, 3=night",
                "day_type": "0=weekday, 1=weekend"
            }
        }
    }

@app.post("/api/v1/baseline/upload-custom")
async def upload_custom_baseline(data: SampleDataUpload, db: SQLSession = Depends(get_db)):
    """Upload custom baseline stats and sample data for a user"""
    
    # Create or update user
    if not DatabaseService.user_exists(db, data.user_id):
        DatabaseService.create_user(db, data.user_id)
        
        # Create basic Q-table model for new user if they don't have one
        try:
            basic_qtable = generate_basic_qtable_for_new_user(data.user_id)
            DatabaseService.save_model_checkpoint(
                db=db,
                user_id=data.user_id,
                training_step=0,
                epsilon=0.9,  # High exploration for new user
                alpha=0.1,
                gamma=0.95,
                q_table=basic_qtable
            )
            print(f"  ✅ Basic Q-table model created for new user {data.user_id}")
        except Exception as e:
            print(f"  ⚠️  Warning: Could not create basic model: {e}")
    
    # Upload baseline stats
    baseline_stats = data.baseline_stats
    stats_dict = {
        "median_target_app_usage_seconds": (baseline_stats.total_usage_time_minutes * 60) // max(baseline_stats.unique_apps, 1),
        "median_session_usage_seconds": 300,  # Default
        "query_interval_seconds": 300  # Default
    }
    DatabaseService.save_baseline_stats(
        db=db,
        user_id=data.user_id,
        stats=stats_dict
    )
    
    # Upload sample sessions using save_sessions method
    if data.sample_sessions:
        try:
            # Convert session data to list format expected by save_sessions
            sessions_list = []
            for session_data in data.sample_sessions:
                sessions_list.append(session_data)
            
            # Use save_sessions method with current date
            DatabaseService.save_sessions(
                db=db,
                user_id=data.user_id,
                date=datetime.now().strftime("%Y-%m-%d"),
                sessions=sessions_list
            )
            uploaded_count = len(sessions_list)
        except Exception as e:
            print(f"Error uploading sessions: {e}")
            uploaded_count = 0
    else:
        uploaded_count = 0
    
    # Save agent parameters as model checkpoint with Q-table
    qtable_data = None
    
    # First priority: use Q-table from sample data if provided
    if hasattr(data, 'q_table') and data.q_table:
        qtable_data = data.q_table
        print(f"Using provided Q-table with {len(qtable_data)} states")
    
    # Second priority: generate Q-table if agent parameters exist but no Q-table
    elif data.agent_parameters:
        qtable_data = generate_realistic_qtable(data.user_id)
        print(f"Generated Q-table for user {data.user_id} with {len(qtable_data)} states")
    
    # Third priority: create basic zero-initialized Q-table for new user
    else:
        qtable_data = generate_basic_qtable_for_new_user(data.user_id)
        print(f"Created basic Q-table for new user {data.user_id} with {len(qtable_data)} states")
    
    # Save the model checkpoint with Q-table
    try:
        DatabaseService.save_model_checkpoint(
            db=db,
            user_id=data.user_id,
            training_step=data.agent_parameters.get("training_steps", 0) if data.agent_parameters else 0,
            epsilon=data.agent_parameters.get("epsilon", 0.9) if data.agent_parameters else 0.9,
            alpha=data.agent_parameters.get("alpha", 0.1) if data.agent_parameters else 0.1,
            gamma=data.agent_parameters.get("gamma", 0.95) if data.agent_parameters else 0.95,
            q_table=qtable_data
        )
        model_saved = True
        print(f"✅ Model checkpoint saved for user {data.user_id}")
    except Exception as e:
        print(f"❌ Error saving model checkpoint: {e}")
        model_saved = False
    
    return {
        "message": "Custom baseline data uploaded successfully",
        "user_id": data.user_id,
        "sessions_uploaded": uploaded_count,
        "baseline_stats_created": True,
        "agent_parameters_saved": bool(data.agent_parameters),
        "model_saved": model_saved,
        "q_table_states": len(qtable_data) if qtable_data else 0,
        "q_table_source": (
            "provided_sample_data" if hasattr(data, 'q_table') and data.q_table
            else "generated_initialized" if data.agent_parameters
            else "basic_zero_init"
        )
    }

# ============================================================================
# BACKGROUND TRAINING TASK (RUNS DAILY)
# ============================================================================

async def train_model_daily(user_id: str, date: str, queries: List[Query], db: SQLSession):
    """
    Background task to train the model with today's uploaded queries.
    Uses ACTUAL action and compliance data from device queries.
    Logs all training updates to database.
    Trains daily regardless of baseline/intervention period.
    """
    print(f"Training model for user {user_id} from date {date}")
    
    # Use queries passed from upload endpoint
    if not queries:
        print(f"No queries for user {user_id} on date {date} - skipping training (no learning data)")
        
        # Skip training entirely when there are no queries to learn from
        # This is especially important for sample data uploads where sessions exist but no real queries
        return {
            "status": "skipped",
            "reason": "no_queries_to_learn_from",
            "learned_transitions": 0,
            "q_table_size": 0,
            "training_steps": 0,
            "checkpoint_saved": False
        }
        
    print(f"Training model for user {user_id} with {len(queries)} queries from {date}")
    
    # Load agent from latest checkpoint using centralized loading method
    checkpoint = DatabaseService.get_latest_model(db, user_id)
    agent = EdgeQLearningAgent.load_from_checkpoint(checkpoint)
    
    # Sort queries by group_id first, then timestamp for proper Markov sequential learning
    # This ensures we learn from consecutive state transitions within the same session context
    # Handle both string timestamps (from Pydantic) and datetime objects (from DB)
    def get_sort_key(q):
        timestamp = datetime.fromisoformat(q.timestamp) if isinstance(q.timestamp, str) else q.timestamp
        return (q.group_id, timestamp)
    
    sorted_queries = sorted(queries, key=get_sort_key)
    print(f"Sorted {len(sorted_queries)} queries for training (by group_id, then timestamp)")
    
    # Log query distribution by group for visibility
    from collections import Counter
    group_counts = Counter(q.group_id for q in sorted_queries)
    print(f"Query distribution: {dict(group_counts)}")
    
    # Learning loop - process each query with its next query
    learned_count = 0
    initial_training_steps = agent.training_steps
    
    for i, query in enumerate(sorted_queries):
        if i + 1 >= len(sorted_queries):
            break
        
        next_query = sorted_queries[i + 1]
        if query.group_id != next_query.group_id:
            print(f"\n🔄 GROUP BOUNDARY DETECTED: {query.group_id} → {next_query.group_id}")
            print(f"   Triggering TERMINAL STATE LEARNING for group transition")
            
            # Learn from terminal state (unreachable state) before group boundary
            terminal_state = (-1, -1, -1, -1)  # Terminal state with all Q-values = 0
            
            # Extract current state from query
            if isinstance(query.state, list):
                current_state = tuple(query.state)
            elif isinstance(query.state, str):
                current_state = tuple(json.loads(query.state))
            else:
                print(f"❌ Error: Unexpected type for query.state: {type(query.state)}")
                continue
            
            # Validate current state dimensions
            def is_valid_state(s):
                return (len(s) == 4 and 
                        0 <= s[0] <= 10 and  # num_queries
                        0 <= s[1] <= 5 and   # num_vibrations
                        s[2] in [0, 1] and   # is_target_app
                        0 <= s[3] <= 3)      # day_quarter
            
            if not is_valid_state(current_state):
                print(f"⚠️ Skipping invalid current_state for terminal learning: {current_state} (expected ranges: [0-10, 0-5, 0-1, 0-3])")
                continue
            
            # Get action and compliance from current query
            action = query.action
            compliance = query.compliance
            reward = calculate_reward(action, compliance)
            
            print(f"🎯 TERMINAL STATE LEARNING:")
            print(f"   Current State: {current_state} (queries={current_state[0]}, vibrations={current_state[1]}, target_app={bool(current_state[2])}, time_quarter={current_state[3]})")
            print(f"   Terminal State: {terminal_state} (all Q-values = 0.0)")
            print(f"   Action: {'VIBRATE' if action == 1 else 'NO_VIBRATE'} ({action})")
            print(f"   Compliance: {'COMPLIED' if compliance == 1 else 'DID_NOT_COMPLY'} ({compliance})")
            print(f"   Reward: {reward:.2f}")
            
            # Get Q-values before terminal learning
            q_before = agent.q_table[current_state][action]
            print(f"   Q-value before: {q_before:.4f}")
            
            # Learn from terminal state (this will increment agent.training_steps internally)
            print(f"   Learning parameters: α={agent.alpha}, γ={agent.gamma}")
            agent.learn(current_state, terminal_state, action, reward)
            
            # Get Q-values after terminal learning
            q_after = agent.q_table[current_state][action]
            q_change = q_after - q_before
            print(f"   Q-value after: {q_after:.4f} (change: {q_change:+.4f})")
            print(f"   Training steps: {agent.training_steps}")
            
            # Log terminal state learning to database
            DatabaseService.log_training(
                db, user_id, date, current_state, terminal_state, action, reward,
                q_before, q_after, agent.alpha, agent.gamma
            )
            
            learned_count += 1
            print(f"✅ Terminal learning completed and logged to database")
            print(f"   Total learned transitions so far: {learned_count}")
            
            continue
        print(f"\n=== Processing query pair {i} ===")
        print(f"About to enter try block")
        
        # Extract state and next_state from queries
        # Handle both string (from DB) and list (from Pydantic) formats  
        # query.state is already a list when passed from upload endpoint
        if isinstance(query.state, list):
            state = tuple(query.state)
        elif isinstance(query.state, str):
            state = tuple(json.loads(query.state))
        else:
            print(f"Error: Unexpected type for query.state: {type(query.state)}")
            continue
            
        if isinstance(next_query.state, list):
            next_state = tuple(next_query.state)
        elif isinstance(next_query.state, str):
            next_state = tuple(json.loads(next_query.state))
        else:
            print(f"Error: Unexpected type for next_query.state: {type(next_query.state)}")
            continue
        
        # Validate state dimensions to ensure Q-table doesn't exceed 528 states
        # Expected: (num_queries: 0-10, num_vibrations: 0-5, is_target_app: 0-1, day_quarter: 0-3)
        def is_valid_state(s):
            return (len(s) == 4 and 
                    0 <= s[0] <= 10 and  # num_queries
                    0 <= s[1] <= 5 and   # num_vibrations
                    s[2] in [0, 1] and   # is_target_app
                    0 <= s[3] <= 3)      # day_quarter
        
        if not is_valid_state(state):
            print(f"⚠️ Skipping invalid state: {state} (expected ranges: [0-10, 0-5, 0-1, 0-3])")
            continue
        
        if not is_valid_state(next_state):
            print(f"⚠️ Skipping invalid next_state: {next_state} (expected ranges: [0-10, 0-5, 0-1, 0-3])")
            continue
        
        # Get action and compliance from query
        action = query.action
        compliance = query.compliance
        
        # Calculate reward using simplified logic
        reward = calculate_reward(action, compliance)
        
        # Get Q-values before learning
        q_before = agent.q_table[state][action]
        
        # Learn (this will increment agent.training_steps internally)
        agent.learn(state, next_state, action, reward)
        
        # Get Q-values after learning
        q_after = agent.q_table[state][action]
        
        # LOG TO DATABASE
        DatabaseService.log_training(
            db, user_id, date, state, next_state, action, reward,
            q_before, q_after, agent.alpha, agent.gamma
        )
        
        learned_count += 1
        print(f"Learned: state={state}, action={action}, compliance={compliance}, reward={reward:.2f}, next_state={next_state}")

        
    # ALWAYS save model checkpoint after training attempt (even if no learning occurred)
    # This ensures the model is updated every time the upload endpoint is called
    checkpoint_saved = False
    print(f"\n💾 Saving checkpoint: {len(agent.q_table)} states, training_step={agent.training_steps}, ε={agent.epsilon:.4f}")
    try:
        # padded_qtable = pad_qtable_to_full_shape({json.dumps(list(k)): v for k, v in agent.q_table.items()})
        DatabaseService.save_model_checkpoint(
            db, user_id, agent.training_steps, agent.epsilon,
            agent.alpha, agent.gamma, q_table={json.dumps(list(k)): v for k, v in agent.q_table.items()}    
        )
        # DatabaseService.save_model_checkpoint(
        #     db, user_id, agent.training_steps, agent.epsilon,
        #     agent.alpha, agent.gamma, {json.dumps(list(k)): v for k, v in agent.q_table.items()}
        # )
        checkpoint_saved = True
        print(f"✅ Model checkpoint saved successfully for user {user_id}")
        print(f"   Q-table size: {len(agent.q_table)} states")
    except Exception as e:
        print(f"❌ Error saving model checkpoint for user {user_id}: {e}")
    
    # Increment training steps if no learning occurred (to track training attempts)
    if learned_count == 0 and agent.training_steps == initial_training_steps:
        agent.training_steps += 1
        print(f"No learning transitions found, but incrementing training steps to track attempt")
        # Save again with incremented training steps
        try:
            DatabaseService.save_model_checkpoint(
                db, user_id, agent.training_steps, agent.epsilon,
                agent.alpha, agent.gamma, {json.dumps(list(k)): v for k, v in agent.q_table.items()}
            )
            checkpoint_saved = True
            print(f"✅ Updated checkpoint with incremented training steps")
        except Exception as e:
            print(f"❌ Error saving updated checkpoint: {e}")
    
    print(f"Training complete. Learned from {learned_count} transitions. Q-table size: {len(agent.q_table)}, Training steps: {agent.training_steps}")
    print(f"Checkpoint saved: {checkpoint_saved}")
    
    # Prepare Q-table for response
    q_table_data = {json.dumps(list(k)): v for k, v in agent.q_table.items()}
    
    return {
        "learned_transitions": learned_count,
        "q_table_size": len(agent.q_table),
        "training_steps": agent.training_steps,
        "checkpoint_saved": checkpoint_saved,
        "q_table": q_table_data,
        "model_metadata": {
            "epsilon": agent.epsilon,
            "alpha": agent.alpha,
            "gamma": agent.gamma,
            "training_steps": agent.training_steps,
            "q_table_states": len(agent.q_table),
            "last_updated": datetime.now().isoformat()
        }
    }

@app.get("/")
async def root():
    """Serve the API tester UI"""
    ui_path = os.path.join(os.path.dirname(__file__), "static", "index.html")
    if os.path.exists(ui_path):
        return FileResponse(ui_path)
    return {
        "service": "SmartQuit Edge ML API",
        "version": "2.0.0 - Database Enabled",
        "status": "running",
        "docs": "/docs",
        "ui": "/static/index.html",
        "database": "Neon PostgreSQL",
        "update_schedule": "Daily at 3 AM per user - trains from day 1",
        "training_policy": "Trains Q-learning model daily regardless of baseline/intervention period",
        "persistence": "All data points stored in PostgreSQL",
        "baseline_period": "2 uploads (Day 1 collects data, Day 2 calculates baseline)",
        "ab_testing": "Random test/production mode allocation (50/50 split)",
        "features": ["baseline_optimization", "random_mode_allocation", "daily_training"]
    }

# ============================================================================
# RUN SERVER
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)