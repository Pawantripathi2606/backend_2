"""
Recognition router - face photo capture, model training, and face recognition.
"""
from fastapi import APIRouter, Depends, UploadFile, File, HTTPException
from sqlalchemy.orm import Session
from typing import Optional
from datetime import date, datetime
import base64
import os
import re
import json

import models
import schemas
from database import get_db
from utils.face_utils import FaceDetector, FaceRecognizer

router = APIRouter(prefix="/recognition", tags=["Recognition"])

# ─── Paths (resolved lazily so MEDIA_ROOT env var is always respected) ────────
def get_media_root():
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.getenv("MEDIA_ROOT", os.path.join(base, 'media'))

def get_training_dir():
    d = os.path.join(get_media_root(), 'training_data')
    os.makedirs(d, exist_ok=True)
    return d

def get_model_path():
    d = os.path.join(get_media_root(), 'models')
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, 'classifier.xml')


# ─── Helpers ─────────────────────────────────────────────────────────────────

def decode_base64_image(image_data: str):
    """Decode a base64 image (with or without data URI prefix) to a numpy array."""
    import numpy as np
    from PIL import Image
    from io import BytesIO
    import cv2

    if ',' in image_data:
        image_data = image_data.split(',')[1]

    img_bytes = base64.b64decode(image_data)
    pil_img = Image.open(BytesIO(img_bytes))
    img_cv = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    return img_cv


def count_existing_photos(student_db_id: int) -> int:
    """Count how many training photos already exist for a student db id."""
    training_dir = get_training_dir()
    if not os.path.exists(training_dir):
        return 0
    pattern = re.compile(rf'^user\.{student_db_id}\.\d+\.jpg$')
    return len([f for f in os.listdir(training_dir) if pattern.match(f)])


# ─── Endpoints ───────────────────────────────────────────────────────────────

@router.post("/save-photo", response_model=schemas.SavePhotoResponse, summary="Save a captured face photo")
def save_photo(payload: schemas.SavePhotoRequest, db: Session = Depends(get_db)):
    """
    Accept a base64-encoded image and student_id string.
    Detects the face, crops & saves it as a training sample.
    Auto-marks photo_samples_taken=True when >= 100 photos are collected.
    """
    import cv2

    try:
        training_dir = get_training_dir()

        # Look up student
        student = db.query(models.Student).filter(
            models.Student.student_id == payload.student_id
        ).first()
        if not student:
            return schemas.SavePhotoResponse(success=False, error=f"Student '{payload.student_id}' not found.")

        # Decode image
        img_cv = decode_base64_image(payload.image)

        # Detect face
        detector = FaceDetector()
        faces = detector.detect_faces(img_cv)

        if len(faces) == 0:
            return schemas.SavePhotoResponse(
                success=False,
                error="No face detected. Please ensure your face is clearly visible and well-lit."
            )

        # Get first face, preprocess
        x, y, w, h = faces[0]
        face_cropped = detector.crop_face(img_cv, (x, y, w, h))
        face_resized = cv2.resize(face_cropped, (450, 450))
        face_gray = cv2.cvtColor(face_resized, cv2.COLOR_BGR2GRAY)

        # Save image
        photo_count = count_existing_photos(student.id) + 1
        filename = f'user.{student.id}.{photo_count}.jpg'
        filepath = os.path.join(training_dir, filename)
        cv2.imwrite(filepath, face_gray)

        # Mark photo_samples_taken once we have >= 100 photos
        should_train = False
        if photo_count >= 100 and not student.photo_samples_taken:
            student.photo_samples_taken = True
            db.commit()
            should_train = True

        return schemas.SavePhotoResponse(
            success=True,
            count=photo_count,
            should_train=should_train,
            message=f"Photo {photo_count} saved successfully."
        )

    except Exception as e:
        return schemas.SavePhotoResponse(success=False, error=str(e))



@router.post("/train", response_model=schemas.TrainResponse, summary="Train the face recognition model")
def train_model(db: Session = Depends(get_db)):
    """
    Train the LBPH face recognition model from all saved training photos.
    Requires at least 10 images.
    """
    try:
        training_dir = get_training_dir()

        if not os.path.exists(training_dir):
            return schemas.TrainResponse(
                success=False,
                error="No training data found. Please capture photos first."
            )

        image_files = [f for f in os.listdir(training_dir) if f.endswith('.jpg')]
        if len(image_files) < 10:
            return schemas.TrainResponse(
                success=False,
                error=f"Insufficient training data. Found {len(image_files)} images, need at least 10."
            )

        recognizer = FaceRecognizer()
        success, num_images, num_students, accuracy = recognizer.train_model(training_dir)

        if success:
            record = models.TrainingModel(
                model_file='media/models/classifier.xml',
                num_students=num_students,
                num_images=num_images,
                accuracy=accuracy
            )
            db.add(record)
            db.commit()

            return schemas.TrainResponse(
                success=True,
                message=f"Model trained successfully! {num_images} images from {num_students} students.",
                num_images=num_images,
                num_students=num_students,
                accuracy=accuracy
            )
        else:
            return schemas.TrainResponse(success=False, error="Training failed. Check training data.")

    except Exception as e:
        return schemas.TrainResponse(success=False, error=str(e))


@router.post("/recognize", response_model=schemas.RecognizeResponse, summary="Recognize a face from image upload")
def recognize_face(
    file: UploadFile = File(..., description="Image file to recognize face from"),
    db: Session = Depends(get_db)
):
    """
    Upload an image file (JPEG/PNG).
    Detects and recognizes the face using the trained LBPH model.
    Automatically marks attendance if a student is recognized with >= 70% confidence.
    """
    import cv2
    import numpy as np
    from datetime import date, datetime

    try:
        # Read uploaded image
        image_bytes = file.file.read()
        np_arr = np.frombuffer(image_bytes, np.uint8)
        img_cv = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if img_cv is None:
            return schemas.RecognizeResponse(success=False, error="Could not decode image.")

        # Detect face
        detector = FaceDetector()
        faces = detector.detect_faces(img_cv)

        if len(faces) == 0:
            return schemas.RecognizeResponse(
                success=True,
                recognized=False,
                message="No face detected in the image."
            )

        # Pick the largest face for best match
        faces_sorted = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
        x, y, w, h = faces_sorted[0]
        face_roi = detector.crop_face(img_cv, (x, y, w, h))
        face_gray = detector.preprocess_face(face_roi)

        # Recognize
        recognizer = FaceRecognizer()
        if not recognizer.load_model():
            return schemas.RecognizeResponse(
                success=False,
                error="No trained model found. Please train the model first."
            )

        student_db_id, confidence = recognizer.recognize_face(face_gray)

        if confidence < 70:
            return schemas.RecognizeResponse(
                success=True,
                recognized=False,
                confidence=round(confidence, 1),
                message=f"Face not recognized with sufficient confidence ({confidence:.1f}%)."
            )

        # Lookup student
        student = db.query(models.Student).filter(models.Student.id == student_db_id).first()
        if not student:
            return schemas.RecognizeResponse(
                success=True,
                recognized=False,
                message=f"No student found with DB id {student_db_id}."
            )

        # Mark attendance
        today_str = date.today().isoformat()
        now_str = datetime.now().strftime("%H:%M:%S")

        existing = db.query(models.Attendance).filter(
            models.Attendance.student_id == student.id,
            models.Attendance.date == today_str
        ).first()

        if existing:
            return schemas.RecognizeResponse(
                success=True,
                recognized=True,
                student_id=student.student_id,
                student_name=student.name,
                department=student.get_department_display(),
                confidence=round(confidence, 1),
                attendance_marked=False,
                already_marked=True,
                marked_at=existing.time,
                message=f"Attendance already marked for {student.name} today at {existing.time}."
            )

        # Create attendance record
        attendance = models.Attendance(
            student_id=student.id,
            date=today_str,
            time=now_str,
            status='P',
            confidence=round(confidence, 1)
        )
        db.add(attendance)
        db.commit()

        return schemas.RecognizeResponse(
            success=True,
            recognized=True,
            student_id=student.student_id,
            student_name=student.name,
            department=student.get_department_display(),
            confidence=round(confidence, 1),
            attendance_marked=True,
            already_marked=False,
            message=f"Attendance marked for {student.name} ({student.student_id})."
        )

    except Exception as e:
        return schemas.RecognizeResponse(success=False, error=str(e))
