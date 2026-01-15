"""
SmartQuit Edge ML API
Backend API for training and serving Q-Learning models to Android devices
Daily model updates at 3 AM per user
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
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

from database import get_db, init_db
from db_service import DatabaseService
from sqlalchemy.orm import Session as SQLSession

app = FastAPI(title="SmartQuit API", version="2.0.0 - Database Enabled")

# CORS middleware for Android app
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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

class DailyUpload(BaseModel):
    user_id: str
    sessions: List[Session]
    date: str  # The date these sessions are from (YYYY-MM-DD format)

class UserRegistration(BaseModel):
    user_id: str
    device_info: Optional[Dict] = None
    apps_to_monitor: Optional[List[str]] = None  # Apps user wants to monitor (empty list = all apps)

class ModelDownload(BaseModel):
    user_id: str
    day_number: int

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
            "q_table": {str(k): v for k, v in dict(self.q_table).items()},
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
            agent.q_table[eval(k)] = v
        agent.epsilon = data["epsilon"]
        agent.training_steps = data["training_steps"]
        return agent
    
    @staticmethod
    def from_compact_binary(data: str):
        """Reconstruct from binary"""
        return pickle.loads(base64.b64decode(data))

# ============================================================================
# DATABASE STORAGE (Using Neon PostgreSQL via SQLAlchemy)
# See database.py and db_service.py for database schema and operations
# ============================================================================

# ============================================================================
# CONFIGURATION
# ============================================================================

SOCIAL_MEDIA_APPS = [
    "com.facebook.katana", "com.instagram.android", "com.whatsapp",
    "com.facebook.orca", "com.google.android.youtube", "com.snapchat.android",
    "com.twitter.android", "com.reddit.frontpage", "com.pinterest",
    "com.tiktok.android", "com.linkedin.android", "org.telegram.messenger",
    "com.threads", "com.signal.android", "com.discord", "tv.twitch.android.app",
    "com.quora.android", "com.imo.android.imoim", "com.viber.voip", "com.tumblr",
    "com.rumble.video", "com.triller.android", "app.clapper.social",
    "com.spotify.music", "com.vevo.android", "com.teamx.android",
    "com.linecorp.line", "com.bsky.app", "com.beatreal.android",
    "com.xiaohongshu.app", "com.lemon8.android", "com.zigazoo.android",
    "com.clapper.android", "com.bumble.app", "com.meetup",
    "com.gab.android", "com.patreon.android", "com.sclub.community"
]

# EXACTLY matches config from original SmartQuit.ipynb
REWARD_CONFIG = {
    'vibration_penalty': -40,
    'vibration_compliance_reward': 60,
    'session_break_reward': 40,
    'long_session_penalty': -40
}

# Session division from original (test_type A or C = 120 seconds)
SESSION_DIVISION_SECONDS = 120

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def calculate_baseline_stats(sessions: List[Session]) -> Dict:
    """
    EXACTLY matches original baseline calculation from SmartQuit.ipynb
    - calculate_median_target_app_usage
    - calculate_short_session_length (50th percentile)
    - calculate_75_percentile_length (for query interval)
    """
    session_durations = []
    target_app_times = []
    
    for session in sessions:
        duration_seconds = session.duration_seconds
        session_durations.append(duration_seconds)
        
        if session.app_name in SOCIAL_MEDIA_APPS:
            target_app_times.append(duration_seconds)
    
    if not session_durations:
        return {
            "median_target_usage_minutes": 5,
            "short_session_threshold_seconds": 120,
            "query_interval_seconds": 300
        }
    
    session_durations.sort()
    target_app_times.sort()
    
    # 50th percentile for short session threshold
    median_duration = session_durations[len(session_durations) // 2]
    
    # 75th percentile for query interval (matches original)
    percentile_75_index = int(len(session_durations) * 0.75)
    percentile_75 = session_durations[percentile_75_index] if percentile_75_index < len(session_durations) else session_durations[-1]
    
    # Median target app usage in MINUTES (floor like original)
    median_target = 0
    if target_app_times:
        median_target_seconds = target_app_times[len(target_app_times) // 2]
        median_target = math.floor(median_target_seconds / 60)
    
    return {
        "median_target_usage_minutes": median_target,
        "short_session_threshold_seconds": median_duration,
        "query_interval_seconds": percentile_75
    }

def add_group_ids_to_sessions(sessions: List[Dict], baseline_stats: Dict) -> List[Dict]:
    """
    EXACTLY matches original add_group_ids function from SmartQuit.ipynb
    Groups sessions with gaps less than session_division_seconds (120 seconds default)
    """
    session_division_seconds = 120  # From original config
    
    group_id = 0
    group_time = timedelta(seconds=0)
    previous_end_time = None
    
    for session in sessions:
        if previous_end_time is None or (session['start_time'] - previous_end_time) > timedelta(seconds=session_division_seconds):
            group_id += 1
            group_time = timedelta(seconds=0)
        group_time += session['duration']
        session['group_time'] = group_time
        session['group_id'] = group_id
        previous_end_time = session['end_time']
    
    return sessions

def get_first_app_in_group(sessions: List[Dict], group_id: int) -> str:
    """EXACTLY matches original get_first_app_in_group"""
    for session in sessions:
        if session['group_id'] == group_id:
            return session['app_name']
    return None

def get_grouped_day_sessions(sessions: List[Dict], baseline_stats: Dict) -> List[Dict]:
    """
    EXACTLY matches original get_grouped_day_sessions function
    Groups sessions and handles compliance
    """
    session_division_seconds = 120
    
    grouped_sessions = []
    current_group_sessions = []
    previous_end_time = None
    group_id = 1
    
    for session in sessions:
        if previous_end_time is None or \
           (session['start_time'] - previous_end_time) > timedelta(seconds=session_division_seconds) or \
           session.get('complied') == 1:
            # Start a new group
            if current_group_sessions:
                grouped_session = create_grouped_session(current_group_sessions, group_id)
                grouped_sessions.append(grouped_session)
                group_id += 1
            current_group_sessions = [session]
        else:
            current_group_sessions.append(session)
        previous_end_time = session['end_time']
    
    # Add the last group
    if current_group_sessions:
        grouped_session = create_grouped_session(current_group_sessions, group_id)
        grouped_sessions.append(grouped_session)
    
    return grouped_sessions

def create_grouped_session(sessions: List[Dict], group_id: int) -> Dict:
    """Helper to create grouped session dict"""
    return {
        'group_id': group_id,
        'sessions': sessions,
        'duration': sum((s['duration'] for s in sessions), timedelta(0)),
        'target_app_duration': sum((s['duration'] for s in sessions if s['app_name'] in SOCIAL_MEDIA_APPS), timedelta(0)),
        'action': 1 if any(s.get('action', 0) == 1 for s in sessions) else 0,
        'start_time': sessions[0]['start_time'],
        'end_time': sessions[-1]['end_time'],
        'app_ids': [s['app_name'] for s in sessions],
        'date': sessions[0]['date'],
        'total_vibrations': sum((1 for s in sessions if s.get('action', 0) == 1), 0),
        'complied_vibrations': sum((1 for s in sessions if s.get('complied', 0) == 1 and s.get('action', 0) == 1), 0),
        'updated_duration': sum(
            ((s.get('updated_duration', s['duration']) if s.get('complied', 0) == 1 and s.get('action', 0) == 1 else s['duration'])
             for s in sessions), timedelta(0)
        ),
        'total_action_taken': sum((1 for s in sessions if s.get('action_taken', False)), 0)
    }

def extract_state_from_session(session: Dict, first_app: str, is_first_query: bool, baseline_stats: Dict) -> Tuple:
    """Extract state for action selection during simulation"""
    hour = session['start_time'].hour
    if 0 <= hour < 6:
        day_quarter = 0
    elif 6 <= hour < 12:
        day_quarter = 1
    elif 12 <= hour < 18:
        day_quarter = 2
    else:
        day_quarter = 3
    
    is_weekday = int(session['start_time'].weekday() < 5)
    is_target = int(first_app in SOCIAL_MEDIA_APPS)
    is_short = 1 if is_first_query else 0
    
    return (is_target, is_short, day_quarter, is_weekday)

def extract_state_from_first_app(first_app: str, group: Dict, is_short: int, baseline_stats: Dict) -> Tuple:
    """Extract state from grouped session for learning"""
    timestamp = group['start_time']
    
    hour = timestamp.hour
    if 0 <= hour < 6:
        day_quarter = 0
    elif 6 <= hour < 12:
        day_quarter = 1
    elif 12 <= hour < 18:
        day_quarter = 2
    else:
        day_quarter = 3
    
    is_weekday = int(timestamp.weekday() < 5)
    is_target = int(first_app in SOCIAL_MEDIA_APPS)
    
    return (is_target, is_short, day_quarter, is_weekday)

def get_grouped_day_sessions_from_actual_data(sessions: List[Dict], baseline_stats: Dict) -> List[Dict]:
    """
    Group sessions based on ACTUAL vibration and compliance data from device.
    When user complies with vibration, it starts a new group.
    """
    session_division_seconds = 120
    
    grouped_sessions = []
    current_group_sessions = []
    previous_end_time = None
    group_id = 1
    
    for session in sessions:
        # Start new group if:
        # 1. Time gap > 120 seconds, OR
        # 2. User complied with vibration in previous session
        if previous_end_time is None or \
           (session['start_time'] - previous_end_time) > timedelta(seconds=session_division_seconds) or \
           (current_group_sessions and current_group_sessions[-1].get('user_complied', False)):
            
            # Save previous group
            if current_group_sessions:
                grouped_session = create_grouped_session_from_actual_data(current_group_sessions, group_id)
                grouped_sessions.append(grouped_session)
                group_id += 1
            current_group_sessions = [session]
        else:
            current_group_sessions.append(session)
        
        previous_end_time = session['end_time']
    
    # Add the last group
    if current_group_sessions:
        grouped_session = create_grouped_session_from_actual_data(current_group_sessions, group_id)
        grouped_sessions.append(grouped_session)
    
    return grouped_sessions

def create_grouped_session_from_actual_data(sessions: List[Dict], group_id: int) -> Dict:
    """Create grouped session with ACTUAL vibration and compliance data"""
    
    # Count actual vibrations and compliances
    total_vibrations = sum(s.get('num_vibrations', 0) for s in sessions)
    complied_vibrations = sum(1 for s in sessions if s.get('user_complied', False))
    
    # Determine group action (1 if ANY vibration occurred in group)
    group_action = 1 if any(s.get('num_vibrations', 0) > 0 for s in sessions) else 0
    
    return {
        'group_id': group_id,
        'sessions': sessions,
        'duration': sum((s['duration'] for s in sessions), timedelta(0)),
        'target_app_duration': sum((s['duration'] for s in sessions if s['app_name'] in SOCIAL_MEDIA_APPS), timedelta(0)),
        'action': group_action,
        'start_time': sessions[0]['start_time'],
        'end_time': sessions[-1]['end_time'],
        'app_ids': [s['app_name'] for s in sessions],
        'date': sessions[0]['date'],
        'total_vibrations': total_vibrations,
        'complied_vibrations': complied_vibrations,
        'has_social_media': len(set([s['app_name'] for s in sessions]) & set(SOCIAL_MEDIA_APPS)) > 0
    }

def calculate_reward_from_actual_data(group: Dict, next_group: Dict, median_target_usage_minutes: float) -> float:
    """
    Calculate reward based on ACTUAL user behavior from device.
    Uses real vibration and compliance data.
    
    Reward components:
    1. Vibration penalty: -40 per vibration
    2. Compliance reward: +60 per complied vibration
    3. Session break reward: +40 if user took 2+ min break after vibration
    4. Long session penalty: -40 if no action on long social media session
    """
    reward = 0.0
    
    # 1. Vibration penalty - penalize each vibration
    reward += group['total_vibrations'] * REWARD_CONFIG['vibration_penalty']
    
    # 2. Compliance reward - reward when user actually complied
    reward += group['complied_vibrations'] * REWARD_CONFIG['vibration_compliance_reward']
    
    # 3. Long session penalty - penalize if no vibration on long social media session
    if group['action'] == 0 and group['has_social_media']:
        if group['duration'] > timedelta(minutes=median_target_usage_minutes):
            reward += REWARD_CONFIG['long_session_penalty']
    
    # 4. Session break reward - reward if user took break after vibration
    if group['action'] == 1 and next_group:
        break_duration = (next_group['start_time'] - group['end_time']).total_seconds() / 60.0
        if break_duration >= 2.0:  # 2 minutes break
            reward += REWARD_CONFIG['session_break_reward']
    
    return reward

def extract_state(session: Session, is_short: bool, baseline_stats: Dict) -> Tuple:
    """Extract state features for Q-learning"""
    timestamp = datetime.fromisoformat(session.start_time)
    
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
    is_target = int(session.app_name in SOCIAL_MEDIA_APPS)
    
    # Is short session
    is_short_session = int(is_short)
    
    return (is_target, is_short_session, day_quarter, is_weekday)

def process_sessions_into_groups(sessions: List[Session], gap_seconds: int = 120) -> List[Dict]:
    """Group sessions with small time gaps"""
    if not sessions:
        return []
    
    # Sort by time
    sorted_sessions = sorted(sessions, key=lambda x: x.start_time)
    
    groups = []
    current_group = [sorted_sessions[0]]
    
    for i in range(1, len(sorted_sessions)):
        prev_end = datetime.fromisoformat(sorted_sessions[i-1].end_time)
        curr_start = datetime.fromisoformat(sorted_sessions[i].start_time)
        
        gap = (curr_start - prev_end).total_seconds()
        
        if gap <= gap_seconds:
            current_group.append(sorted_sessions[i])
        else:
            groups.append(current_group)
            current_group = [sorted_sessions[i]]
    
    groups.append(current_group)
    
    # Convert to group dicts
    group_dicts = []
    for idx, group in enumerate(groups):
        total_duration = sum(s.duration_seconds for s in group)
        target_duration = sum(
            s.duration_seconds for s in group 
            if s.app_name in SOCIAL_MEDIA_APPS
        )
        
        group_dicts.append({
            "group_id": idx,
            "sessions": group,
            "total_duration": total_duration,
            "target_duration": target_duration,
            "start_time": group[0].start_time,
            "end_time": group[-1].end_time,
            "first_app": group[0].app_name
        })
    
    return group_dicts

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
    
    if DatabaseService.user_exists(db, registration.user_id):
        raise HTTPException(status_code=400, detail="User already exists")
    
    # If no apps specified, use all social media apps by default
    apps_to_monitor = registration.apps_to_monitor if registration.apps_to_monitor else SOCIAL_MEDIA_APPS
    
    print(f"  Final apps_to_monitor to save: {apps_to_monitor}")
    
    user = DatabaseService.create_user(db, registration.user_id, registration.device_info, apps_to_monitor)
    
    print(f"  ✅ User created successfully")
    print(f"  Saved apps_to_monitor: {user.apps_to_monitor}\n")
    
    return {
        "status": "success",
        "user_id": registration.user_id,
        "apps_to_monitor": apps_to_monitor,
        "message": "User registered. Please complete baseline week (days 0-6) before intervention."
    }

@app.post("/api/v1/sessions/upload")
async def upload_daily_sessions(
    batch: DailyUpload,
    background_tasks: BackgroundTasks,
    db: SQLSession = Depends(get_db)
):
    """
    Upload daily session data from device.
    Day number is automatically calculated from the date.
    Days 0-6 = baseline (no training, just stats)
    Days 7-34 = intervention (train model daily)
    
    Device should call this every day at 3 AM with yesterday's sessions
    """
    if not DatabaseService.user_exists(db, batch.user_id):
        raise HTTPException(status_code=404, detail="User not found")
    
    try:
        # Validate date format
        print(f"📥 Upload received: User={batch.user_id}, Date={batch.date}, Sessions={len(batch.sessions)}")
        print(f"   Date format: {batch.date}")
        if len(batch.sessions) > 0:
            print(f"   Sample session start_time: {batch.sessions[0].start_time}")
        
        # Automatically calculate day number from date
        day_number = DatabaseService.calculate_day_number(db, batch.user_id, batch.date)
        
        print(f"✅ Calculated day number: {day_number} for date {batch.date}")
        
        # Save sessions to database
        DatabaseService.save_sessions(db, batch.user_id, batch.date, batch.sessions)
        
        response = {
            "status": "received",
            "sessions_count": len(batch.sessions),
            "day_number": day_number,
            "date": batch.date
        }
        
        # If we've completed baseline week (day 6), calculate stats
        if day_number == 6:
            baseline_sessions = DatabaseService.get_baseline_sessions(db, batch.user_id)
            print(f"Calculating baseline stats from {len(baseline_sessions)} sessions")
            stats = calculate_baseline_stats(baseline_sessions)
            DatabaseService.save_baseline_stats(db, batch.user_id, stats)
            response["baseline_stats"] = stats
            response["message"] = "Baseline week completed. Model training starts from day 7."
            print(f"Baseline stats: {stats}")
        
        # If intervention days (7-34), train model in background
        elif day_number >= 7:
            print(f"Scheduling training for user {batch.user_id}, day {day_number}")
            background_tasks.add_task(
                train_model_daily,
                batch.user_id,
                batch.date,
                db
            )
            response["message"] = f"Training model with day {day_number} data. Updated model ready for download."
        else:
            response["message"] = f"Baseline day {day_number} recorded. Continue uploading daily until day 6."
        
        # Update user's current day
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
    
    baseline_stats = DatabaseService.get_baseline_stats(db, user_id)
    
    if not baseline_stats:
        raise HTTPException(
            status_code=400,
            detail="Complete baseline week (days 0-6) first before downloading model"
        )
    
    # Get latest model checkpoint
    checkpoint = DatabaseService.get_latest_model(db, user_id)
    
    model_data = {
        "user_id": user_id,
        "model_version": checkpoint.training_step if checkpoint else 0,
        "updated_at": datetime.now().isoformat(),
        "baseline_stats": {
            "median_target_usage_minutes": baseline_stats.median_target_usage_minutes,
            "short_session_threshold_seconds": baseline_stats.short_session_threshold_seconds,
            "query_interval_seconds": baseline_stats.query_interval_seconds
        },
        "reward_config": REWARD_CONFIG,
        "social_media_apps": SOCIAL_MEDIA_APPS,
    }
    
    if format == "binary" and checkpoint and checkpoint.model_binary:
        model_data["agent_data"] = checkpoint.model_binary.hex()
        model_data["format"] = "binary"
    else:
        # Return Q-table as JSON
        model_data["agent_data"] = checkpoint.q_table_json if checkpoint else {}
        model_data["format"] = "json"
    
    return model_data

@app.get("/api/v1/model/status/{user_id}")
async def get_model_status(user_id: str, db: SQLSession = Depends(get_db)):
    """Check if model is ready for download and get current day"""
    if not DatabaseService.user_exists(db, user_id):
        raise HTTPException(status_code=404, detail="User not found")
    
    user = DatabaseService.get_user(db, user_id)
    baseline_stats = DatabaseService.get_baseline_stats(db, user_id)
    checkpoint = DatabaseService.get_latest_model(db, user_id)
    
    return {
        "user_id": user_id,
        "current_day": user.current_day,
        "baseline_completed": baseline_stats is not None,
        "model_ready": baseline_stats is not None and checkpoint is not None and checkpoint.training_step > 0,
        "training_steps": checkpoint.training_step if checkpoint else 0,
        "epsilon": checkpoint.epsilon if checkpoint else 1.0,
        "q_table_size": len(checkpoint.q_table_json) if checkpoint else 0,
        "last_model_update": checkpoint.created_at.isoformat() if checkpoint else None
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
            "median_target_usage_minutes": baseline_stats.median_target_usage_minutes,
            "short_session_threshold_seconds": baseline_stats.short_session_threshold_seconds,
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

# ============================================================================
# BACKGROUND TRAINING TASK (RUNS DAILY)
# ============================================================================

async def train_model_daily(user_id: str, date: str, db: SQLSession):
    """
    Background task to train the model with today's uploaded sessions.
    Uses ACTUAL vibration and compliance data from device.
    Logs all training updates to database.
    """
    # Get baseline stats
    baseline_stats = DatabaseService.get_baseline_stats(db, user_id)
    
    if not baseline_stats:
        print(f"No baseline stats for user {user_id}")
        return
    
    baseline_stats_dict = {
        "median_target_usage_minutes": baseline_stats.median_target_usage_minutes,
        "short_session_threshold_seconds": baseline_stats.short_session_threshold_seconds,
        "query_interval_seconds": baseline_stats.query_interval_seconds
    }
    
    # Get today's sessions from database
    today_sessions = DatabaseService.get_sessions_by_date(db, user_id, date)
    
    if not today_sessions:
        print(f"No sessions for user {user_id} on date {date}")
        return
    
    print(f"Training model for user {user_id} with {len(today_sessions)} sessions from {date}")
    
    # Reconstruct agent from latest checkpoint or create new
    checkpoint = DatabaseService.get_latest_model(db, user_id)
    agent = EdgeQLearningAgent()
    if checkpoint:
        agent.q_table = defaultdict(_default_q_values)
        for k, v in checkpoint.q_table_json.items():
            agent.q_table[eval(k)] = v
        agent.epsilon = checkpoint.epsilon
        agent.training_steps = checkpoint.training_step
    
    # Convert database sessions to dict format
    session_dicts = []
    for s in today_sessions:
        session_dicts.append({
            'app_name': s.app_name,
            'start_time': s.start_time,
            'end_time': s.end_time,
            'duration': timedelta(seconds=s.duration_seconds),
            'date': s.date,
            'num_vibrations': s.num_vibrations,
            'user_complied': s.user_complied,
            'action': 1 if s.num_vibrations > 0 else 0,
            'complied': 1 if s.user_complied else 0,
        })
    
    # Add group IDs
    session_dicts = add_group_ids_to_sessions(session_dicts, baseline_stats_dict)
    
    # Group sessions
    grouped_sessions = get_grouped_day_sessions_from_actual_data(session_dicts, baseline_stats_dict)
    
    print(f"Created {len(grouped_sessions)} grouped sessions for learning")
    
    # Learning loop
    learned_count = 0
    for i, group in enumerate(grouped_sessions):
        if i + 1 >= len(grouped_sessions):
            break
        
        next_group = grouped_sessions[i + 1]
        
        # Extract states
        first_app = get_first_app_in_group(session_dicts, group['group_id'])
        is_short = 1 if group['duration'] <= timedelta(seconds=baseline_stats_dict["short_session_threshold_seconds"]) else 0
        state = extract_state_from_first_app(first_app, group, is_short, baseline_stats_dict)
        
        next_first_app = get_first_app_in_group(session_dicts, next_group['group_id'])
        next_is_short = 1 if next_group['duration'] <= timedelta(seconds=baseline_stats_dict["short_session_threshold_seconds"]) else 0
        next_state = extract_state_from_first_app(next_first_app, next_group, next_is_short, baseline_stats_dict)
        
        # Calculate reward
        reward = calculate_reward_from_actual_data(
            group,
            next_group,
            baseline_stats_dict["median_target_usage_minutes"]
        )
        
        # Get Q-values before learning
        q_before = agent.q_table[state][group['action']]
        
        # Learn
        agent.learn(state, next_state, group['action'], reward)
        
        # Get Q-values after learning
        q_after = agent.q_table[state][group['action']]
        
        # LOG TO DATABASE
        DatabaseService.log_training(
            db, user_id, date, state, next_state, group['action'], reward,
            q_before, q_after, agent.alpha, agent.gamma
        )
        
        learned_count += 1
        print(f"Learned: state={state}, action={group['action']}, reward={reward:.2f}, next_state={next_state}")
    
    # Save model checkpoint to database
    DatabaseService.save_model_checkpoint(
        db, user_id, agent.training_steps, agent.epsilon,
        agent.alpha, agent.gamma, dict(agent.q_table)
    )
    
    print(f"Training complete. Learned from {learned_count} transitions. Q-table size: {len(agent.q_table)}, Training steps: {agent.training_steps}")

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
        "update_schedule": "Daily at 3 AM per user",
        "persistence": "All data points stored in PostgreSQL"
    }

# ============================================================================
# RUN SERVER
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)