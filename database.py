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
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    device_info = Column(JSON, nullable=True)
    apps_to_monitor = Column(JSON, nullable=True)  # List of apps user wants to monitor
    current_day = Column(Integer, default=0)
    baseline_completed = Column(Boolean, default=False)
    start_date = Column(DateTime, nullable=True)
    is_test_mode = Column(Boolean, default=False)  # Random allocation for A/B testing
    
    # Relationships
    sessions = relationship("Session", back_populates="user", cascade="all, delete-orphan")
    baseline_stats = relationship("BaselineStats", back_populates="user", uselist=False, cascade="all, delete-orphan")
    model_checkpoints = relationship("ModelCheckpoint", back_populates="user", cascade="all, delete-orphan")
    training_logs = relationship("TrainingLog", back_populates="user", cascade="all, delete-orphan")
    api_logs = relationship("APILog", back_populates="user", cascade="all, delete-orphan")
    queries = relationship("Query", back_populates="user", cascade="all, delete-orphan")


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
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
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
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
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
    """Baseline statistics for a user (from first 2 days)"""
    __tablename__ = "baseline_stats"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String, ForeignKey("users.id"), unique=True, index=True)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    median_target_app_usage_seconds = Column(Float)
    median_session_usage_seconds = Column(Float)
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
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
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
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class APILog(Base):
    """API call logs for tracking and debugging"""
    __tablename__ = "api_logs"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String, ForeignKey("users.id"), index=True)
    
    # API request details
    endpoint = Column(String, index=True)
    method = Column(String)  # GET, POST, etc.
    status_code = Column(Integer, index=True)
    
    # Request/Response info
    request_body = Column(JSON, nullable=True)
    response_body = Column(JSON, nullable=True)
    error_message = Column(String, nullable=True)
    
    # Timing
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    user = relationship("User", back_populates="api_logs")


class Query(Base):
    """Query logs from device - tracks each intervention decision and compliance"""
    __tablename__ = "queries"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String, ForeignKey("users.id"), index=True)
    
    # Query details
    group_id = Column(Integer, index=True)
    date = Column(String, index=True)  # YYYY-MM-DD
    timestamp = Column(DateTime, index=True)
    current_app = Column(String, index=True)
    state = Column(String)  # State tuple as JSON string (e.g., "[0, 1, 1, 1]")
    action = Column(Integer)  # 0 or 1 (binary)
    compliance = Column(Integer)  # 0 or 1 (binary)
    is_exploit = Column(Integer, default=0)  # 0 = random/explore, 1 = Q-table exploit
    
    # Timing
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    user = relationship("User", back_populates="queries")


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

def create_tables():
    """Create all database tables with current schema"""
    try:
        print("🔄 Creating database tables...")
        Base.metadata.create_all(bind=engine)
        print("✅ Database tables created successfully")
        return True
    except Exception as e:
        print(f"❌ Error creating tables: {e}")
        return False

def run_migrations():
    """Run any pending database migrations"""
    try:
        from sqlalchemy import text
        print("🔄 Running database migrations...")
        
        # Check if is_test_mode column exists in users table
        with engine.connect() as conn:
            if "sqlite" in str(engine.url):
                result = conn.execute(text("PRAGMA table_info(users)"))
                columns = [row[1] for row in result.fetchall()]
                has_test_mode = 'is_test_mode' in columns
            else:
                result = conn.execute(text("""
                    SELECT column_name 
                    FROM information_schema.columns 
                    WHERE table_name = 'users' AND column_name = 'is_test_mode'
                """))
                has_test_mode = result.fetchone() is not None
        
        # Add column if it doesn't exist
        if not has_test_mode:
            print("📝 Adding is_test_mode column to users table...")
            
            # Use autocommit mode for DDL statements
            with engine.connect().execution_options(autocommit=True) as conn:
                if "sqlite" in str(engine.url):
                    conn.execute(text("ALTER TABLE users ADD COLUMN is_test_mode BOOLEAN DEFAULT FALSE"))
                else:
                    conn.execute(text("ALTER TABLE users ADD COLUMN is_test_mode BOOLEAN DEFAULT FALSE NOT NULL"))
            
            print("✅ is_test_mode column added successfully")
            print("   📝 All existing users automatically set to production mode (default: FALSE)")
        else:
            print("✅ is_test_mode column already exists")
        
        print("✅ Database migrations completed")
        return True
        
    except Exception as e:
        print(f"❌ Error running migrations: {e}")
        return False

def initialize_database():
    """Initialize database with tables and migrations"""
    print("🚀 Initializing database...")
    
    # Create tables first
    if not create_tables():
        return False
    
    # Run migrations
    if not run_migrations():
        return False
    
    print("🎉 Database initialization completed successfully!")
    return True
