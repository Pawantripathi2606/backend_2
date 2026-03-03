"""
Database configuration using SQLAlchemy.

- Production (Render/Cloud): Set DATABASE_URL env var to your PostgreSQL connection string.
  Example: postgresql://user:password@host:5432/dbname

- Local development: Falls back to SQLite (no setup needed).
"""
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv
import os

load_dotenv()  # Load .env file before reading any env vars

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Database URL ───────────────────────────────────────────────────────────────
# Priority: DATABASE_URL env var (PostgreSQL) → local SQLite fallback
DATABASE_URL = os.getenv("DATABASE_URL")

if DATABASE_URL:
    # Render provides URLs starting with "postgres://" but SQLAlchemy needs "postgresql://"
    if DATABASE_URL.startswith("postgres://"):
        DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)
    # PostgreSQL — no special connect_args needed
    engine = create_engine(DATABASE_URL)
    print(f"✅ Connected to PostgreSQL database.")
else:
    # Local fallback — SQLite
    MEDIA_ROOT = os.getenv("MEDIA_ROOT", os.path.join(BASE_DIR, "..", "media"))
    os.makedirs(MEDIA_ROOT, exist_ok=True)
    SQLITE_URL = f"sqlite:///{os.path.join(MEDIA_ROOT, 'face_recognition.db')}"
    engine = create_engine(
        SQLITE_URL,
        connect_args={"check_same_thread": False}  # Required for SQLite only
    )
    print(f"⚠️  DATABASE_URL not set. Using local SQLite: {SQLITE_URL}")

# ── Session factory ────────────────────────────────────────────────────────────
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# ── Base class for ORM models ──────────────────────────────────────────────────
Base = declarative_base()


def get_db():
    """FastAPI dependency — provides a DB session per request, auto-closed after."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
