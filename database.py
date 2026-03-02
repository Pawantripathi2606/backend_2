"""
Database configuration using SQLAlchemy with SQLite.
On Render: set MEDIA_ROOT env var to /opt/render/project/src/media (persistent disk).
Locally: defaults to fastapi_app/ directory.
"""
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Use MEDIA_ROOT env var if set (Render persistent disk), else local
MEDIA_ROOT = os.getenv("MEDIA_ROOT", os.path.join(BASE_DIR, "..", "media"))
os.makedirs(MEDIA_ROOT, exist_ok=True)

DATABASE_URL = f"sqlite:///{os.path.join(MEDIA_ROOT, 'face_recognition.db')}"

# Create SQLAlchemy engine
engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False}  # Required for SQLite
)

# Session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base class for ORM models
Base = declarative_base()


def get_db():
    """FastAPI dependency to provide a DB session per request."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
