"""
Database operations service - replaces InMemoryStore
"""

from sqlalchemy.orm import Session
from sqlalchemy import desc, and_
from datetime import datetime, timedelta
import json
from database import SessionLocal, User, Session as SessionModel, GroupedSession, BaselineStats, ModelCheckpoint, TrainingLog, FeedbackLog, Query

# ============================================================================
# Q-TABLE STATE SPACE DEFINITION
# ============================================================================
# State: (num_queries, num_vibrations, is_target_app, day_quarter)
# - num_queries: 0-10
# - num_vibrations: 0-5 (capped at 5)
# - is_target_app: 0 (not target), 1 (target app)
# - day_quarter: 0 (night 0-6), 1 (morning 6-12), 2 (afternoon 12-18), 3 (evening 18-24)

STATE_SPACE = {
    'num_queries_range': range(0, 11),        # 0-10
    'num_vibrations_range': range(0, 6),      # 0-5
    'is_target_app_range': [0, 1],            # 0 or 1
    'day_quarter_range': range(0, 4)          # 0-3
}

# Total states: 11 * 6 * 2 * 4 = 528
TOTAL_Q_TABLE_STATES = 528
MAX_NUM_QUERIES = 10  # Cap for num_queries state component
MAX_NUM_VIBRATIONS = 5  # Cap for num_vibrations state component


def generate_complete_qtable() -> dict:
    """
    Generate a complete Q-table with all possible states initialized to [0.0, 0.0].
    This ensures consistent Q-table size across all model checkpoints.
    
    Returns:
        Dict with 528 states, each with format: "[num_queries, num_vibrations, is_target, day_quarter]": [0.0, 0.0]
    """
    qtable = {}
    for num_queries in STATE_SPACE['num_queries_range']:
        for num_vibrations in STATE_SPACE['num_vibrations_range']:
            for is_target_app in STATE_SPACE['is_target_app_range']:
                for day_quarter in STATE_SPACE['day_quarter_range']:
                    state = (num_queries, num_vibrations, is_target_app, day_quarter)
                    qtable[json.dumps(list(state))] = [0.0, 0.0]
    return qtable


def ensure_complete_qtable(qtable: dict) -> dict:
    """
    Ensure a Q-table has all possible states, filling missing ones with [0.0, 0.0].
    This function is called before saving any Q-table to the database.
    
    Args:
        qtable: Existing Q-table (may have missing states)
    
    Returns:
        Complete Q-table with all 528 states
    """
    # Start with complete Q-table (all zeros)
    complete = generate_complete_qtable()
    
    # Update with existing values (preserving learned Q-values)
    if qtable:
        for key, value in qtable.items():
            # Normalize key format to ensure consistency
            try:
                if isinstance(key, str):
                    # Parse the state from string format
                    parsed = json.loads(key) if key.startswith('[') else [int(x) for x in key.split('_')]
                    state_tuple = tuple(parsed)
                else:
                    state_tuple = tuple(key)
                
                # Cap num_queries and num_vibrations if needed
                if len(state_tuple) == 4:
                    num_queries = min(state_tuple[0], MAX_NUM_QUERIES)
                    num_vib = min(state_tuple[1], MAX_NUM_VIBRATIONS)
                    normalized_state = (num_queries, num_vib, state_tuple[2], state_tuple[3])
                    normalized_key = json.dumps(list(normalized_state))
                    
                    # Only update if this is a valid state in our state space
                    if normalized_key in complete:
                        complete[normalized_key] = value
            except (json.JSONDecodeError, ValueError, TypeError) as e:
                print(f"⚠️  Skipping invalid Q-table key '{key}': {e}")
                continue
    
    return complete


class DatabaseService:
    @staticmethod
    def get_upload_count(db: Session, user_id: str) -> int:
        """Return the number of unique session upload dates for a user (i.e., number of uploads)."""
        return db.query(SessionModel.date).filter(SessionModel.user_id == user_id).distinct().count()

    
    """Service layer for all database operations"""
    @staticmethod
    def create_user(db: Session, user_id: str, device_info: dict = None, apps_to_monitor: list = None, is_test_mode: bool = False):
        """Create a new user"""
        user = User(
            id=user_id,
            device_info=device_info,
            apps_to_monitor=apps_to_monitor,  # Store list of apps to monitor
            current_day=0,
            is_test_mode=is_test_mode,
            created_at=datetime.utcnow()
        )
        db.add(user)
        db.commit()
        db.refresh(user)
        return user
    
    @staticmethod
    def get_user(db: Session, user_id: str):
        """Get user by ID"""
        return db.query(User).filter(User.id == user_id).first()
    
    @staticmethod
    def user_exists(db: Session, user_id: str) -> bool:
        """Check if user exists"""
        return db.query(User).filter(User.id == user_id).first() is not None
    
    @staticmethod
    def save_sessions(db: Session, user_id: str, date: str, sessions: list):
        """Save daily sessions to database"""
        # Validate date format
        try:
            datetime.strptime(date, "%Y-%m-%d")
        except ValueError as e:
            print(f"❌ Invalid date format: {date}. Expected YYYY-MM-DD. Error: {e}")
            raise ValueError(f"Invalid date format: {date}. Expected format YYYY-MM-DD (e.g., 2025-01-15)")
        
        # Get user's apps_to_monitor for target app checking
        user = db.query(User).filter(User.id == user_id).first()
        apps_to_monitor = user.apps_to_monitor if user and user.apps_to_monitor else []
        
        # Delete existing sessions for this date (in case of re-upload)
        db.query(SessionModel).filter(
            and_(SessionModel.user_id == user_id, SessionModel.date == date)
        ).delete()
        db.commit()
        
        # Insert new sessions
        for i, session in enumerate(sessions):
            start_time_str = getattr(session, 'start_time', None)
            end_time_str = getattr(session, 'end_time', None)
            
            if not start_time_str or not end_time_str:
                raise ValueError(f"Session {i}: Missing start_time or end_time")
            
            try:
                # Debug logging
                print(f"   [Session {i}] Parsing start_time: '{start_time_str}' (type: {type(start_time_str).__name__}, len: {len(str(start_time_str))})")
                print(f"   [Session {i}] Parsing end_time: '{end_time_str}' (type: {type(end_time_str).__name__}, len: {len(str(end_time_str))})")
                
                # Try parsing with timezone first, then without
                try:
                    start_dt = datetime.fromisoformat(start_time_str)
                except ValueError as e1:
                    # Fallback: try parsing as naive datetime if fromisoformat fails
                    try:
                        start_dt = datetime.strptime(start_time_str, "%Y-%m-%dT%H:%M:%S")
                    except ValueError as e2:
                        raise ValueError(f"Could not parse start_time with fromisoformat or strptime. Tried formats: ISO (e.g., 2025-01-15T14:30:00+05:30) and YYYY-MM-DDTHH:MM:SS. Error1: {e1}. Error2: {e2}")
                
                try:
                    end_dt = datetime.fromisoformat(end_time_str)
                except ValueError as e1:
                    # Fallback: try parsing as naive datetime if fromisoformat fails
                    try:
                        end_dt = datetime.strptime(end_time_str, "%Y-%m-%dT%H:%M:%S")
                    except ValueError as e2:
                        raise ValueError(f"Could not parse end_time with fromisoformat or strptime. Tried formats: ISO (e.g., 2025-01-15T14:30:00+05:30) and YYYY-MM-DDTHH:MM:SS. Error1: {e1}. Error2: {e2}")
                
                print(f"   [Session {i}] ✅ Parsed successfully: {start_dt} to {end_dt}")
                    
            except ValueError as e:
                print(f"❌ Invalid datetime format in session {i}: start_time={start_time_str}, end_time={end_time_str}. Error: {e}")
                raise ValueError(f"Session {i}: Invalid datetime format. Error: {e}")
            
            db_session = SessionModel(
                user_id=user_id,
                app_name=session.app_name,
                start_time=start_dt,
                end_time=end_dt,
                duration_seconds=session.duration_seconds,
                date=date,
                num_vibrations=session.num_vibrations,
                user_complied=session.user_complied,
                group_id=session.group_id,
                is_target_app=session.app_name in apps_to_monitor
            )
            db.add(db_session)
        
        db.commit()
        print(f"✅ Saved {len(sessions)} sessions for user {user_id} on date {date}")
    
    @staticmethod
    def get_sessions_by_date(db: Session, user_id: str, date: str) -> list:
        """Get sessions for a specific date"""
        return db.query(SessionModel).filter(
            and_(SessionModel.user_id == user_id, SessionModel.date == date)
        ).all()
    
    @staticmethod
    def get_baseline_sessions(db: Session, user_id: str) -> list:
        """Get first 2 days of sessions for baseline calculation"""
        # Get unique dates for this user, ordered
        dates = db.query(SessionModel.date).filter(
            SessionModel.user_id == user_id
        ).distinct().order_by(SessionModel.date).limit(2).all()
        
        baseline_dates = [d[0] for d in dates]
        
        return db.query(SessionModel).filter(
            and_(SessionModel.user_id == user_id, SessionModel.date.in_(baseline_dates))
        ).all()
    
    @staticmethod
    def get_all_sessions(db: Session, user_id: str) -> list:
        """Get all sessions for a user"""
        return db.query(SessionModel).filter(SessionModel.user_id == user_id).all()
    
    @staticmethod
    def save_baseline_stats(db: Session, user_id: str, stats: dict):
        """Save baseline statistics"""
        # Delete existing baseline stats for this user
        db.query(BaselineStats).filter(BaselineStats.user_id == user_id).delete()
        db.commit()
        
        # Insert new baseline stats
        baseline = BaselineStats(
            user_id=user_id,
            median_target_app_usage_seconds=stats["median_target_app_usage_seconds"],
            median_session_usage_seconds=stats["median_session_usage_seconds"],
            query_interval_seconds=stats["query_interval_seconds"]
        )
        db.add(baseline)
        db.commit()
        db.refresh(baseline)
        return baseline
    
    @staticmethod
    def get_baseline_stats(db: Session, user_id: str):
        """Get baseline stats for user"""
        return db.query(BaselineStats).filter(BaselineStats.user_id == user_id).first()
    
    @staticmethod
    def save_model_checkpoint(db: Session, user_id: str, training_step: int, epsilon: float, 
                            alpha: float, gamma: float, q_table: dict, model_binary: bytes = None):
        """Save model checkpoint with complete Q-table (all 96 states)"""
        # Ensure Q-table has all possible states (fills missing with [0.0, 0.0])
        complete_qtable = ensure_complete_qtable(q_table)
        
        # Mark previous checkpoint as not latest
        db.query(ModelCheckpoint).filter(
            and_(ModelCheckpoint.user_id == user_id, ModelCheckpoint.is_latest == True)
        ).update({ModelCheckpoint.is_latest: False})
        
        checkpoint = ModelCheckpoint(
            user_id=user_id,
            training_step=training_step,
            epsilon=epsilon,
            alpha=alpha,
            gamma=gamma,
            q_table_json=complete_qtable,
            model_binary=model_binary,
            is_latest=True
        )
        db.add(checkpoint)
        db.commit()
        db.refresh(checkpoint)
        print(f"💾 Saved model checkpoint for {user_id}: {len(complete_qtable)} states (expected {TOTAL_Q_TABLE_STATES})")
        return checkpoint
    
    @staticmethod
    def get_latest_model(db: Session, user_id: str):
        """Get latest model checkpoint"""
        return db.query(ModelCheckpoint).filter(
            and_(ModelCheckpoint.user_id == user_id, ModelCheckpoint.is_latest == True)
        ).first()
    
    @staticmethod
    def log_training(db: Session, user_id: str, date: str, state: tuple, next_state: tuple,
                    action: int, reward: float, q_value_before: float, q_value_after: float,
                    alpha: float, gamma: float):
        """Log a training update"""
        log = TrainingLog(
            user_id=user_id,
            date=date,
            state=list(state),
            next_state=list(next_state),
            action=action,
            reward=reward,
            q_value_before=q_value_before,
            q_value_after=q_value_after,
            learning_rate_alpha=alpha,
            discount_factor_gamma=gamma
        )
        db.add(log)
        db.commit()
    
    @staticmethod
    def get_training_history(db: Session, user_id: str, limit: int = 100):
        """Get recent training history"""
        return db.query(TrainingLog).filter(
            TrainingLog.user_id == user_id
        ).order_by(desc(TrainingLog.created_at)).limit(limit).all()
    
    @staticmethod
    def log_feedback(db: Session, user_id: str, state: tuple, next_state: tuple, action: int, reward: float):
        """Log on-device feedback"""
        feedback = FeedbackLog(
            user_id=user_id,
            state=list(state),
            next_state=list(next_state),
            action=action,
            reward=reward
        )
        db.add(feedback)
        db.commit()
    
    @staticmethod
    def save_grouped_session(db: Session, user_id: str, date: str, group_id: int, group_data: dict):
        """Save a grouped session"""
        grouped = GroupedSession(
            user_id=user_id,
            group_id=group_id,
            date=date,
            session_count=len(group_data['sessions']),
            start_time=group_data['start_time'],
            end_time=group_data['end_time'],
            duration_seconds=group_data['duration'].total_seconds(),
            target_app_duration_seconds=group_data['target_app_duration'].total_seconds(),
            total_vibrations=group_data['total_vibrations'],
            complied_vibrations=group_data['complied_vibrations'],
            action_taken=group_data['action'] == 1,
            reward=group_data.get('reward', None)
        )
        db.add(grouped)
        db.commit()
    
    @staticmethod
    def update_user_day(db: Session, user_id: str, day_number: int):
        """Update user's current day"""
        user = db.query(User).filter(User.id == user_id).first()
        if user:
            user.current_day = day_number
            if day_number == 0:
                user.start_date = datetime.utcnow()
            db.commit()
    
    @staticmethod
    def calculate_day_number(db: Session, user_id: str, date_str: str) -> int:
        """Calculate day number from date"""
        user = db.query(User).filter(User.id == user_id).first()
        
        # Validate and parse date with multiple formats
        current_date = None
        for fmt in ["%Y-%m-%d", "%m/%d/%Y", "%d/%m/%Y"]:
            try:
                current_date = datetime.strptime(date_str, fmt)
                break
            except ValueError:
                continue
        
        if current_date is None:
            raise ValueError(f"Unable to parse date: {date_str}. Expected format: YYYY-MM-DD")
        
        if not user.start_date:
            # First upload - set start date
            user.start_date = current_date
            db.commit()
            return 0
        
        day_number = (current_date.date() - user.start_date.date()).days
        return day_number
    
    @staticmethod
    def save_queries(db: Session, user_id: str, date: str, queries: list):
        """Save daily queries to database"""
        # Validate date format
        try:
            datetime.strptime(date, "%Y-%m-%d")
        except ValueError as e:
            print(f"❌ Invalid date format: {date}. Expected YYYY-MM-DD. Error: {e}")
            raise ValueError(f"Invalid date format: {date}. Expected format YYYY-MM-DD (e.g., 2025-01-15)")
        
        # Delete existing queries for this date (in case of re-upload)
        db.query(Query).filter(
            and_(Query.user_id == user_id, Query.date == date)
        ).delete()
        db.commit()
        
        # Insert new queries
        for i, query in enumerate(queries):
            timestamp_str = getattr(query, 'timestamp', None)
            
            if not timestamp_str:
                raise ValueError(f"Query {i}: Missing timestamp")
            
            try:
                # Parse timestamp
                try:
                    timestamp_dt = datetime.fromisoformat(timestamp_str)
                except ValueError as e1:
                    # Fallback: try parsing as naive datetime
                    try:
                        timestamp_dt = datetime.strptime(timestamp_str, "%Y-%m-%dT%H:%M:%S")
                    except ValueError as e2:
                        raise ValueError(f"Could not parse timestamp. Error1: {e1}. Error2: {e2}")
                
                print(f"   [Query {i}] ✅ Parsed timestamp: {timestamp_dt}")
                    
            except ValueError as e:
                print(f"❌ Invalid datetime format in query {i}: timestamp={timestamp_str}. Error: {e}")
                raise ValueError(f"Query {i}: Invalid datetime format. Error: {e}")
            
            # Validate state is a list
            if not isinstance(query.state, list):
                print(f"❌ Query {i}: state is not a list: {type(query.state)} = {query.state}")
                raise ValueError(f"Query {i}: state must be a list, got {type(query.state)}")
            
            db_query = Query(
                user_id=user_id,
                group_id=query.group_id,
                date=date,
                timestamp=timestamp_dt,
                current_app=query.current_app,
                state=json.dumps(query.state),  # Convert list to JSON string
                action=query.action,
                compliance=query.compliance,
                is_exploit=getattr(query, 'is_exploit', 0)  # 0 = random/explore, 1 = Q-table exploit
            )
            db.add(db_query)
        
        db.commit()
        print(f"💾 Saved {len(queries)} queries for user {user_id} on {date}")
    
    @staticmethod
    def get_queries(db: Session, user_id: str, date: str = None):
        """Get queries for a user, optionally filtered by date"""
        query = db.query(Query).filter(Query.user_id == user_id)
        if date:
            query = query.filter(Query.date == date)
        return query.order_by(Query.timestamp.asc()).all()
    
    @staticmethod
    def get_all_queries(db: Session, user_id: str):
        """Get all queries for a user"""
        return db.query(Query).filter(Query.user_id == user_id).order_by(Query.timestamp.asc()).all()

