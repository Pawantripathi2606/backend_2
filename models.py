"""
SQLAlchemy ORM models - ported from Django models.
"""
from sqlalchemy import (
    Column, Integer, String, Boolean, Float,
    DateTime, Text, ForeignKey, UniqueConstraint, LargeBinary
)
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from database import Base


class Student(Base):
    """Student information table."""
    __tablename__ = "students"

    DEPARTMENT_CHOICES = {
        'CS': 'Computer Science', 'CE': 'Civil Engineering',
        'ME': 'Mechanical Engineering', 'EE': 'Electrical Engineering',
    }
    COURSE_CHOICES = {
        'CS_GENERAL': 'Computer Science', 'CS_AI_ML': 'CSE - AI/ML',
        'CS_IOT': 'CSE - IOT', 'CS_AIDS': 'CSE - AIDS', 'CS_DS': 'CSE - DS',
        'CE_CONSTRUCTION': 'Construction Engineering', 'CE_STRUCTURAL': 'Structural Engineering',
        'CE_GEOTECHNICAL': 'Geotechnical Engineering',
        'ME': 'Mechanical Engineering', 'EE': 'Electrical Engineering',
    }
    YEAR_CHOICES = ['2020-21', '2021-22', '2022-23', '2023-24', '2024-25', '2025-26', '2026-27']
    SEMESTER_CHOICES = ['1', '2', '3', '4', '5', '6', '7', '8']
    DIVISION_CHOICES = ['A', 'B', 'C', 'D', 'E']
    GENDER_CHOICES = {'M': 'Male', 'F': 'Female', 'O': 'Others'}

    id = Column(Integer, primary_key=True, index=True)
    student_id = Column(String(50), unique=True, nullable=False, index=True)
    name = Column(String(100), nullable=False)
    department = Column(String(2), nullable=False)
    course = Column(String(20), nullable=False)
    year = Column(String(10), nullable=False)
    semester = Column(String(1), nullable=False)
    division = Column(String(1), nullable=False)
    roll_no = Column(String(50), nullable=True)
    gender = Column(String(1), nullable=False)
    dob = Column(String(10), nullable=False)
    email = Column(String(254), nullable=False)
    phone = Column(String(10), nullable=False)
    address = Column(Text, nullable=False)
    teacher = Column(String(100), nullable=False)
    photo_samples_taken = Column(Boolean, default=False)
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())

    attendances = relationship("Attendance", back_populates="student", cascade="all, delete-orphan")
    training_photos = relationship("TrainingPhoto", back_populates="student", cascade="all, delete-orphan")

    def get_department_display(self): return self.DEPARTMENT_CHOICES.get(self.department, self.department)
    def get_course_display(self): return self.COURSE_CHOICES.get(self.course, self.course)
    def get_gender_display(self): return self.GENDER_CHOICES.get(self.gender, self.gender)


class Attendance(Base):
    """Attendance record table."""
    __tablename__ = "attendance"

    STATUS_CHOICES = {'P': 'Present', 'A': 'Absent', 'L': 'Late'}

    id = Column(Integer, primary_key=True, index=True)
    student_id = Column(Integer, ForeignKey("students.id", ondelete="CASCADE"), nullable=False)
    date = Column(String(10), nullable=False)
    time = Column(String(8), nullable=False)
    status = Column(String(1), default='P', nullable=False)
    confidence = Column(Float, nullable=False)

    student = relationship("Student", back_populates="attendances")

    __table_args__ = (
        UniqueConstraint('student_id', 'date', name='unique_student_date'),
    )

    def get_status_display(self): return self.STATUS_CHOICES.get(self.status, self.status)


class TrainingPhoto(Base):
    """
    Stores training face photos as binary blobs in PostgreSQL.
    No disk required — survives Render redeployments forever.
    """
    __tablename__ = "training_photos"

    id = Column(Integer, primary_key=True, index=True)
    student_id = Column(Integer, ForeignKey("students.id", ondelete="CASCADE"),
                        nullable=False, index=True)
    photo_num = Column(Integer, nullable=False)
    image_data = Column(LargeBinary, nullable=False)   # Grayscale JPEG bytes
    created_at = Column(DateTime, server_default=func.now())

    student = relationship("Student", back_populates="training_photos")

    __table_args__ = (
        UniqueConstraint('student_id', 'photo_num', name='unique_student_photo_num'),
    )


class TrainingModel(Base):
    """Training model — stores LBPH XML bytes in DB (no disk)."""
    __tablename__ = "training_models"

    id = Column(Integer, primary_key=True, index=True)
    trained_at = Column(DateTime, server_default=func.now())
    model_file = Column(String(255), nullable=False, default='db')
    model_data = Column(LargeBinary, nullable=True)    # Raw LBPH XML bytes
    num_students = Column(Integer, nullable=False)
    num_images = Column(Integer, nullable=False)
    accuracy = Column(Float, nullable=True)
    notes = Column(Text, default='')
