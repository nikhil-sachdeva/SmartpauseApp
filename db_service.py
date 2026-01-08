"""
Database operations service - replaces InMemoryStore
"""

from sqlalchemy.orm import Session
from sqlalchemy import desc, and_
from datetime import datetime, timedelta
import json
from database import SessionLocal, User, Session as SessionModel, GroupedSession, BaselineStats, ModelCheckpoint, TrainingLog, FeedbackLog

class DatabaseService:
    """Service layer for all database operations"""
    
    @staticmethod
    def create_user(db: Session, user_id: str, device_info: dict = None):
        """Create a new user"""
        user = User(
            id=user_id,
            device_info=device_info,
            current_day=0,
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
        # Delete existing sessions for this date (in case of re-upload)
        db.query(SessionModel).filter(
            and_(SessionModel.user_id == user_id, SessionModel.date == date)
        ).delete()
        db.commit()
        
        # Insert new sessions
        for session in sessions:
            db_session = SessionModel(
                user_id=user_id,
                app_name=session.app_name,
                start_time=datetime.fromisoformat(session.start_time),
                end_time=datetime.fromisoformat(session.end_time),
                duration_seconds=session.duration_seconds,
                date=date,
                vibration_occurred=session.vibration_occurred,
                user_complied=session.user_complied,
                is_target_app=session.app_name in get_social_media_apps()
            )
            db.add(db_session)
        
        db.commit()
    
    @staticmethod
    def get_sessions_by_date(db: Session, user_id: str, date: str) -> list:
        """Get sessions for a specific date"""
        return db.query(SessionModel).filter(
            and_(SessionModel.user_id == user_id, SessionModel.date == date)
        ).all()
    
    @staticmethod
    def get_baseline_sessions(db: Session, user_id: str) -> list:
        """Get first 7 days of sessions"""
        # Get unique dates for this user, ordered
        dates = db.query(SessionModel.date).filter(
            SessionModel.user_id == user_id
        ).distinct().order_by(SessionModel.date).limit(7).all()
        
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
            median_target_usage_minutes=stats["median_target_usage_minutes"],
            short_session_threshold_seconds=stats["short_session_threshold_seconds"],
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
        """Save model checkpoint"""
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
            q_table_json=q_table,
            model_binary=model_binary,
            is_latest=True
        )
        db.add(checkpoint)
        db.commit()
        db.refresh(checkpoint)
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
        
        if not user.start_date:
            # First upload - set start date
            user.start_date = datetime.strptime(date_str, "%Y-%m-%d")
            db.commit()
            return 0
        
        # Parse date with multiple formats
        for fmt in ["%Y-%m-%d", "%m/%d/%Y", "%d/%m/%Y"]:
            try:
                current_date = datetime.strptime(date_str, fmt)
                break
            except ValueError:
                continue
        else:
            raise ValueError(f"Unable to parse date: {date_str}")
        
        day_number = (current_date.date() - user.start_date.date()).days
        return day_number


def get_social_media_apps():
    """Get list of social media apps"""
    return [
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
