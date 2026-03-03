"""
Database configuration using SQLAlchemy.

- Production (Render/Cloud): Set DATABASE_URL env var to your PostgreSQL connection string.
- Local development: Falls back to SQLite (no setup needed).
"""
from sqlalchemy import create_engine, text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv
import os

load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Database URL ───────────────────────────────────────────────────────────────
DATABASE_URL = os.getenv("DATABASE_URL")

if DATABASE_URL:
    if DATABASE_URL.startswith("postgres://"):
        DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)
    engine = create_engine(DATABASE_URL)
    print("Connected to PostgreSQL database.")
else:
    MEDIA_ROOT = os.getenv("MEDIA_ROOT", os.path.join(BASE_DIR, "..", "media"))
    os.makedirs(MEDIA_ROOT, exist_ok=True)
    SQLITE_URL = f"sqlite:///{os.path.join(MEDIA_ROOT, 'face_recognition.db')}"
    engine = create_engine(SQLITE_URL, connect_args={"check_same_thread": False})
    print(f"DATABASE_URL not set. Using local SQLite: {SQLITE_URL}")

# ── Session factory ────────────────────────────────────────────────────────────
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# ── Base class ─────────────────────────────────────────────────────────────────
Base = declarative_base()


def run_migrations():
    """
    Safely apply schema migrations to existing tables.
    Idempotent — safe to run on every deploy.
    """
    is_postgres = DATABASE_URL is not None

    with engine.connect() as conn:
        if is_postgres:
            # PostgreSQL: ADD COLUMN IF NOT EXISTS (safe to run multiple times)
            migrations = [
                "ALTER TABLE training_models ADD COLUMN IF NOT EXISTS model_data BYTEA",
                "ALTER TABLE training_models ALTER COLUMN model_file SET DEFAULT 'db'",
            ]
            for sql in migrations:
                try:
                    conn.execute(text(sql))
                    conn.commit()
                    print(f"Migration OK: {sql[:60]}")
                except Exception as e:
                    conn.rollback()
                    print(f"Migration skipped (already applied): {e}")
        else:
            # SQLite: check columns first
            try:
                result = conn.execute(text("PRAGMA table_info(training_models)"))
                cols = [row[1] for row in result.fetchall()]
                if "model_data" not in cols:
                    conn.execute(text("ALTER TABLE training_models ADD COLUMN model_data BLOB"))
                    conn.commit()
                    print("SQLite migration: added model_data column")
            except Exception as e:
                print(f"SQLite migration skipped: {e}")

    print("Database migrations complete.")


def get_db():
    """FastAPI dependency — provides a DB session per request."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
