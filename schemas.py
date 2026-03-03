"""
Pydantic schemas for request/response validation.
"""
from pydantic import BaseModel, EmailStr, field_validator
from typing import Optional, List
from datetime import datetime


# ─────────────────────────────────────────────
# Student Schemas
# ─────────────────────────────────────────────

class StudentCreate(BaseModel):
    student_id: str
    name: str
    department: str          # CS, CE, ME, EE
    course: str
    year: str
    semester: str
    division: str
    roll_no: Optional[str] = None
    gender: str              # M, F, O
    dob: str                 # YYYY-MM-DD
    email: str
    phone: str
    address: str
    teacher: str


class StudentUpdate(BaseModel):
    name: Optional[str] = None
    department: Optional[str] = None
    course: Optional[str] = None
    year: Optional[str] = None
    semester: Optional[str] = None
    division: Optional[str] = None
    roll_no: Optional[str] = None
    gender: Optional[str] = None
    dob: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    address: Optional[str] = None
    teacher: Optional[str] = None


class StudentOut(BaseModel):
    id: int
    student_id: str
    name: str
    department: str
    department_display: Optional[str] = None
    course: str
    course_display: Optional[str] = None
    year: str
    semester: str
    division: str
    roll_no: Optional[str] = None
    gender: str
    dob: str
    email: str
    phone: str
    address: str
    teacher: str
    photo_samples_taken: bool
    created_at: Optional[datetime] = None

    class Config:
        from_attributes = True


# ─────────────────────────────────────────────
# Attendance Schemas
# ─────────────────────────────────────────────

class AttendanceOut(BaseModel):
    id: int
    student_id: int
    student_name: Optional[str] = None
    student_uid: Optional[str] = None    # student.student_id string
    date: str
    time: str
    status: str
    status_display: Optional[str] = None
    confidence: float

    class Config:
        from_attributes = True


class AttendanceFilter(BaseModel):
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    department: Optional[str] = None
    course: Optional[str] = None
    semester: Optional[str] = None


class AttendanceStatsOut(BaseModel):
    total_students: int
    total_records: int
    today_attendance: int
    today_percentage: float
    week_attendance: int
    avg_confidence: float
    dept_stats: List[dict]


# ─────────────────────────────────────────────
# Recognition Schemas
# ─────────────────────────────────────────────

class SavePhotoRequest(BaseModel):
    student_id: str        # The student_id string (e.g. "CS001")
    image: str             # Base64-encoded image (data URI or raw base64)


class SavePhotoResponse(BaseModel):
    success: bool
    count: int = 0
    should_train: bool = False
    message: str = ""
    error: str = ""


class TrainResponse(BaseModel):
    success: bool
    message: str = ""
    num_images: int = 0
    num_students: int = 0
    accuracy: float = 0.0
    error: str = ""


class RecognizeResponse(BaseModel):
    success: bool
    recognized: bool = False
    student_id: Optional[str] = None
    student_name: Optional[str] = None
    department: Optional[str] = None
    confidence: float = 0.0
    attendance_marked: bool = False
    already_marked: bool = False
    marked_at: Optional[str] = None
    message: str = ""
    error: str = ""
    # Face bounding box — used by frontend to draw overlay rectangle
    bbox: Optional[dict] = None        # {x, y, w, h} in pixels of the captured frame
    img_w: Optional[int] = None        # captured frame width
    img_h: Optional[int] = None        # captured frame height
