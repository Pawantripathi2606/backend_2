"""
FastAPI entry point for the Face Recognition Attendance System.

Run with:
    uvicorn main:app --reload
    
API docs available at: http://127.0.0.1:8000/docs
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import os

from database import engine, Base
import models  # noqa: F401 - ensures models are registered with Base


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Create all database tables on startup."""
    # Ensure media directories exist
    base_dir = os.path.dirname(os.path.abspath(__file__))
    media_root = os.path.join(base_dir, '..', 'media')
    os.makedirs(os.path.join(media_root, 'training_data'), exist_ok=True)
    os.makedirs(os.path.join(media_root, 'models'), exist_ok=True)

    # Auto-create DB tables
    Base.metadata.create_all(bind=engine)
    print("✅ Database tables created / verified.")
    yield
    print("🛑 Shutting down FastAPI app.")


app = FastAPI(
    title="Face Recognition Attendance System",
    description=(
        "REST API for automated attendance marking using face recognition.\n\n"
        "## Features\n"
        "- **Students** — Register, update, and manage student profiles\n"
        "- **Recognition** — Capture training photos, train LBPH model, recognize faces\n"
        "- **Attendance** — View records, dashboard stats, export Excel & PDF reports\n\n"
        "Use the webcam test page or any HTTP client to interact with this API."
    ),
    version="2.0.0",
    lifespan=lifespan,
)

# ─── CORS ─────────────────────────────────────────────────────────────────────
# Allow all origins so the webcam HTML page and any frontend can call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Routers ──────────────────────────────────────────────────────────────────
from routers import students, recognition, attendance

app.include_router(students.router)
app.include_router(recognition.router)
app.include_router(attendance.router)


# ─── Root ─────────────────────────────────────────────────────────────────────
@app.get("/", tags=["Root"])
def root():
    """Health check and welcome endpoint."""
    return {
        "message": "Face Recognition Attendance System API",
        "version": "2.0.0",
        "docs": "http://127.0.0.1:8000/docs",
        "endpoints": {
            "students": "/students/",
            "recognition": "/recognition/",
            "attendance": "/attendance/",
        }
    }
