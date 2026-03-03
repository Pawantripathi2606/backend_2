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

from database import engine, Base, run_migrations
import models  # noqa: F401 - ensures models are registered with Base


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Create tables and apply migrations on startup."""
    # 1. Create any NEW tables (e.g. training_photos)
    Base.metadata.create_all(bind=engine)
    # 2. Apply column migrations to EXISTING tables (e.g. model_data on training_models)
    run_migrations()
    print("Database ready.")
    yield
    print("Shutting down.")


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
