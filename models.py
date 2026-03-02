"""
SQLAlchemy ORM models - ported from Django models.
"""
from sqlalchemy import (
    Column, Integer, String, Boolean, Float,
    Date, Time, DateTime, Text, ForeignKey, UniqueConstraint
)
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from database import Base


class Student(Base):
    """Student information table."""
    __tablename__ = "students"

    # Department choices
    DEPARTMENT_CHOICES = {
        'CS': 'Computer Science',
        'CE': 'Civil Engineering',
        'ME': 'Mechanical Engineering',
        'EE': 'Electrical Engineering',
    }

    # Course choices
    COURSE_CHOICES = {
        'CS_GENERAL': 'Computer Science',
        'CS_AI_ML': 'CSE - AI/ML',
        'CS_IOT': 'CSE - IOT',
        'CS_AIDS': 'CSE - AIDS',
        'CS_DS': 'CSE - DS',
        'CE_CONSTRUCTION': 'Construction Engineering',
        'CE_STRUCTURAL': 'Structural Engineering',
        'CE_GEOTECHNICAL': 'Geotechnical Engineering',
        'ME': 'Mechanical Engineering',
        'EE': 'Electrical Engineering',
    }

    # Year choices
    YEAR_CHOICES = ['2020-21', '2021-22', '2022-23', '2023-24', '2024-25', '2025-26', '2026-27']

    # Semester choices
    SEMESTER_CHOICES = ['1', '2', '3', '4', '5', '6', '7', '8']

    # Division choices
    DIVISION_CHOICES = ['A', 'B', 'C', 'D', 'E']

    # Gender choices
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
    dob = Column(String(10), nullable=False)  # Stored as ISO string YYYY-MM-DD

    # Contact information
    email = Column(String(254), nullable=False)
    phone = Column(String(10), nullable=False)
    address = Column(Text, nullable=False)

    # Additional
    teacher = Column(String(100), nullable=False)
    photo_samples_taken = Column(Boolean, default=False)

    # Timestamps
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())

    # Relationships
    attendances = relationship("Attendance", back_populates="student", cascade="all, delete-orphan")

    def get_department_display(self):
        return self.DEPARTMENT_CHOICES.get(self.department, self.department)

    def get_course_display(self):
        return self.COURSE_CHOICES.get(self.course, self.course)

    def get_gender_display(self):
        return self.GENDER_CHOICES.get(self.gender, self.gender)


class Attendance(Base):
    """Attendance record table."""
    __tablename__ = "attendance"

    STATUS_CHOICES = {'P': 'Present', 'A': 'Absent', 'L': 'Late'}

    id = Column(Integer, primary_key=True, index=True)
    student_id = Column(Integer, ForeignKey("students.id", ondelete="CASCADE"), nullable=False)
    date = Column(String(10), nullable=False)   # YYYY-MM-DD
    time = Column(String(8), nullable=False)    # HH:MM:SS
    status = Column(String(1), default='P', nullable=False)
    confidence = Column(Float, nullable=False)

    # Relationships
    student = relationship("Student", back_populates="attendances")

    # One attendance record per student per day
    __table_args__ = (
        UniqueConstraint('student_id', 'date', name='unique_student_date'),
    )

    def get_status_display(self):
        return self.STATUS_CHOICES.get(self.status, self.status)


class TrainingModel(Base):
    """Training model history table."""
    __tablename__ = "training_models"

    id = Column(Integer, primary_key=True, index=True)
    trained_at = Column(DateTime, server_default=func.now())
    model_file = Column(String(255), nullable=False)
    num_students = Column(Integer, nullable=False)
    num_images = Column(Integer, nullable=False)
    accuracy = Column(Float, nullable=True)
    notes = Column(Text, default='')
