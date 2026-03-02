"""
Students router - CRUD operations for students.
"""
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from sqlalchemy import or_
from typing import List, Optional

import models
import schemas
from database import get_db

router = APIRouter(prefix="/students", tags=["Students"])


def student_to_out(student: models.Student) -> schemas.StudentOut:
    """Convert ORM Student to StudentOut schema."""
    return schemas.StudentOut(
        id=student.id,
        student_id=student.student_id,
        name=student.name,
        department=student.department,
        department_display=student.get_department_display(),
        course=student.course,
        course_display=student.get_course_display(),
        year=student.year,
        semester=student.semester,
        division=student.division,
        roll_no=student.roll_no,
        gender=student.gender,
        dob=student.dob,
        email=student.email,
        phone=student.phone,
        address=student.address,
        teacher=student.teacher,
        photo_samples_taken=student.photo_samples_taken,
        created_at=student.created_at,
    )


@router.get("/", response_model=List[schemas.StudentOut], summary="List all students")
def list_students(
    search: Optional[str] = Query(None, description="Search by student_id, name, email, phone, or roll_no"),
    db: Session = Depends(get_db)
):
    """
    Retrieve all students.
    Optionally filter with a `search` query parameter.
    """
    query = db.query(models.Student)
    if search:
        like = f"%{search}%"
        query = query.filter(
            or_(
                models.Student.student_id.ilike(like),
                models.Student.name.ilike(like),
                models.Student.email.ilike(like),
                models.Student.phone.ilike(like),
                models.Student.roll_no.ilike(like),
            )
        )
    students = query.order_by(models.Student.student_id).all()
    return [student_to_out(s) for s in students]


@router.post("/", response_model=schemas.StudentOut, status_code=201, summary="Create a student")
def create_student(payload: schemas.StudentCreate, db: Session = Depends(get_db)):
    """Register a new student."""
    # Check for duplicate student_id
    existing = db.query(models.Student).filter(models.Student.student_id == payload.student_id).first()
    if existing:
        raise HTTPException(status_code=400, detail=f"Student with id '{payload.student_id}' already exists.")

    student = models.Student(**payload.model_dump())
    db.add(student)
    db.commit()
    db.refresh(student)
    return student_to_out(student)


@router.get("/{student_db_id}", response_model=schemas.StudentOut, summary="Get student by DB id")
def get_student(student_db_id: int, db: Session = Depends(get_db)):
    """Retrieve a student by their database primary key."""
    student = db.query(models.Student).filter(models.Student.id == student_db_id).first()
    if not student:
        raise HTTPException(status_code=404, detail="Student not found.")
    return student_to_out(student)


@router.get("/by-student-id/{student_id}", response_model=schemas.StudentOut, summary="Get student by student_id string")
def get_student_by_student_id(student_id: str, db: Session = Depends(get_db)):
    """Retrieve a student by their string student_id (e.g. 'CS001')."""
    student = db.query(models.Student).filter(models.Student.student_id == student_id).first()
    if not student:
        raise HTTPException(status_code=404, detail="Student not found.")
    return student_to_out(student)


@router.put("/{student_db_id}", response_model=schemas.StudentOut, summary="Update a student")
def update_student(student_db_id: int, payload: schemas.StudentUpdate, db: Session = Depends(get_db)):
    """Update an existing student's details."""
    student = db.query(models.Student).filter(models.Student.id == student_db_id).first()
    if not student:
        raise HTTPException(status_code=404, detail="Student not found.")

    update_data = payload.model_dump(exclude_unset=True)
    for key, value in update_data.items():
        setattr(student, key, value)

    db.commit()
    db.refresh(student)
    return student_to_out(student)


@router.delete("/{student_db_id}", summary="Delete a student")
def delete_student(student_db_id: int, db: Session = Depends(get_db)):
    """Delete a student and all their attendance records."""
    student = db.query(models.Student).filter(models.Student.id == student_db_id).first()
    if not student:
        raise HTTPException(status_code=404, detail="Student not found.")

    name = student.name
    db.delete(student)
    db.commit()
    return {"success": True, "message": f"Student '{name}' deleted successfully."}
