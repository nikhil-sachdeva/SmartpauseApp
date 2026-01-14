"""
Database models and configuration for SmartQuit using Neon PostgreSQL
Supports both PostgreSQL (production) and SQLite (local development)
"""

import os
from dotenv import load_dotenv
from sqlalchemy import create_engine, Column, String, Integer, Float, DateTime, Boolean, JSON, LargeBinary, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from datetime import datetime
import json

# Load environment variables from .env file
load_dotenv()

# Detect environment and use appropriate database
DATABASE_URL = os.getenv("DATABASE_URL")
ENVIRONMENT = os.getenv("ENVIRONMENT", "development")

print(f"Environment: {ENVIRONMENT}")
print(f"DATABASE_URL: {DATABASE_URL if DATABASE_URL else 'Not Set'}")
if not DATABASE_URL:
    # Local development: use SQLite
    DATABASE_URL = "sqlite:///./smartquit.db"
    print("⚠️  No DATABASE_URL set. Using SQLite for local development.")
    print("   To use PostgreSQL, set DATABASE_URL environment variable.")
else:
    # Production: use PostgreSQL
    print("✅ Using PostgreSQL from DATABASE_URL")

# Create engine with appropriate settings
if "sqlite" in DATABASE_URL:
    # SQLite settings
    engine = create_engine(
        DATABASE_URL,
        connect_args={"check_same_thread": False},
        echo=False
    )
else:
    # PostgreSQL settings
    engine = create_engine(
        DATABASE_URL,
        pool_size=10,
        max_overflow=20,
        pool_pre_ping=True,  # Verify connections before using
        echo=False
    )

# Session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base class for models
Base = declarative_base()

# ============================================================================
# DATABASE MODELS
# ============================================================================

class User(Base):
    """User account and metadata"""
    __tablename__ = "users"
    
    id = Column(String, primary_key=True, index=True)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    device_info = Column(JSON, nullable=True)
    apps_to_monitor = Column(JSON, nullable=True)  # List of apps user wants to monitor
    current_day = Column(Integer, default=0)
    baseline_completed = Column(Boolean, default=False)
    start_date = Column(DateTime, nullable=True)
    
    # Relationships
    sessions = relationship("Session", back_populates="user", cascade="all, delete-orphan")
    baseline_stats = relationship("BaselineStats", back_populates="user", uselist=False, cascade="all, delete-orphan")
    model_checkpoints = relationship("ModelCheckpoint", back_populates="user", cascade="all, delete-orphan")
    training_logs = relationship("TrainingLog", back_populates="user", cascade="all, delete-orphan")


class Session(Base):
    """Individual app usage session"""
    __tablename__ = "sessions"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String, ForeignKey("users.id"), index=True)
    app_name = Column(String, index=True)
    start_time = Column(DateTime, index=True)
    end_time = Column(DateTime)
    duration_seconds = Column(Float)
    date = Column(String, index=True)  # YYYY-MM-DD
    
    # Real device data
    num_vibrations = Column(Integer, default=0)  # Number of vibrations in this session
    user_complied = Column(Boolean, default=False)
    
    # Processed data
    group_id = Column(Integer, nullable=True)
    group_time_seconds = Column(Float, nullable=True)
    is_target_app = Column(Boolean, index=True)
    
    # Relationships
    user = relationship("User", back_populates="sessions")


class GroupedSession(Base):
    """Grouped sessions for reward calculation"""
    __tablename__ = "grouped_sessions"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String, ForeignKey("users.id"), index=True)
    group_id = Column(Integer)
    date = Column(String, index=True)
    
    # Session details
    session_count = Column(Integer)
    start_time = Column(DateTime)
    end_time = Column(DateTime)
    duration_seconds = Column(Float)
    target_app_duration_seconds = Column(Float)
    
    # Actual device data
    total_vibrations = Column(Integer, default=0)
    complied_vibrations = Column(Integer, default=0)
    action_taken = Column(Boolean)
    
    # Reward data
    reward = Column(Float, nullable=True)
    
    # Relationships
    user_id_ref = Column(String, ForeignKey("users.id"))


class BaselineStats(Base):
    """Baseline statistics for a user (from first 7 days)"""
    __tablename__ = "baseline_stats"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String, ForeignKey("users.id"), unique=True, index=True)
    
    median_target_usage_minutes = Column(Float)
    short_session_threshold_seconds = Column(Float)
    query_interval_seconds = Column(Float)
    
    calculated_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    user = relationship("User", back_populates="baseline_stats")


class ModelCheckpoint(Base):
    """Saved Q-Learning model checkpoints"""
    __tablename__ = "model_checkpoints"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String, ForeignKey("users.id"), index=True)
    
    training_step = Column(Integer)
    epsilon = Column(Float)
    alpha = Column(Float)
    gamma = Column(Float)
    q_table_json = Column(JSON)  # Q-table stored as JSON
    model_binary = Column(LargeBinary, nullable=True)  # Optional: pickled model
    
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    is_latest = Column(Boolean, default=False, index=True)
    
    # Relationships
    user = relationship("User", back_populates="model_checkpoints")


class TrainingLog(Base):
    """Training history and learning updates"""
    __tablename__ = "training_logs"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String, ForeignKey("users.id"), index=True)
    
    date = Column(String, index=True)
    state = Column(JSON)  # State tuple as JSON
    next_state = Column(JSON)  # Next state tuple as JSON
    action = Column(Integer)
    reward = Column(Float)
    
    q_value_before = Column(Float)
    q_value_after = Column(Float)
    
    learning_rate_alpha = Column(Float)
    discount_factor_gamma = Column(Float)
    
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    
    # Relationships
    user = relationship("User", back_populates="training_logs")


class FeedbackLog(Base):
    """Optional: On-device feedback for online learning"""
    __tablename__ = "feedback_logs"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String, ForeignKey("users.id"), index=True)
    
    state = Column(JSON)
    next_state = Column(JSON)
    action = Column(Integer)
    reward = Column(Float)
    
    created_at = Column(DateTime, default=datetime.utcnow)


# ============================================================================
# DATABASE INITIALIZATION
# ============================================================================

def init_db():
    """Create all tables"""
    Base.metadata.create_all(bind=engine)
    print("Database initialized successfully")


def get_db():
    """Dependency to get database session"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
