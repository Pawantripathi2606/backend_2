"""
Recognition router — fully DB-backed.
Training photos and model XML are stored in PostgreSQL (no disk required).
"""
from fastapi import APIRouter, Depends, UploadFile, File
from sqlalchemy.orm import Session
from datetime import date, datetime
import base64
import numpy as np
import cv2

import models
import schemas
from database import get_db
from utils.face_utils import FaceDetector, FaceRecognizer

router = APIRouter(prefix="/recognition", tags=["Recognition"])


# ─── Helpers ──────────────────────────────────────────────────────────────────

def decode_base64_image(image_data: str):
    """Decode a base64 image (with or without data URI prefix) to a numpy array."""
    from PIL import Image
    from io import BytesIO

    if ',' in image_data:
        image_data = image_data.split(',')[1]

    img_bytes = base64.b64decode(image_data)
    pil_img = Image.open(BytesIO(img_bytes))
    img_cv = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    return img_cv


def encode_image_to_bytes(gray_img) -> bytes:
    """Encode a grayscale numpy image to JPEG bytes for storage in DB."""
    success, buffer = cv2.imencode('.jpg', gray_img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not success:
        raise ValueError("Failed to encode image to JPEG")
    return buffer.tobytes()


def count_photos_in_db(student_db_id: int, db: Session) -> int:
    return db.query(models.TrainingPhoto).filter(
        models.TrainingPhoto.student_id == student_db_id
    ).count()


def get_latest_model(db: Session):
    """Return the latest TrainingModel row, or None."""
    return db.query(models.TrainingModel).order_by(
        models.TrainingModel.id.desc()
    ).first()


# ─── Endpoints ────────────────────────────────────────────────────────────────

@router.post("/save-photo", response_model=schemas.SavePhotoResponse,
             summary="Save a captured face photo to the database")
def save_photo(payload: schemas.SavePhotoRequest, db: Session = Depends(get_db)):
    """
    Accepts a base64-encoded image + student_id string.
    Detects the face, crops it, and stores JPEG bytes directly in PostgreSQL.
    No disk write — survives Render redeployments.
    """
    try:
        # Look up student by their string ID
        student = db.query(models.Student).filter(
            models.Student.student_id == payload.student_id
        ).first()
        if not student:
            return schemas.SavePhotoResponse(
                success=False,
                error=f"Student '{payload.student_id}' not found. Register the student first."
            )

        # Decode base64 → numpy image
        img_cv = decode_base64_image(payload.image)

        # Detect face (fast params for bulk capture)
        detector = FaceDetector()
        faces = detector.detect_faces(img_cv, fast=True)

        if len(faces) == 0:
            return schemas.SavePhotoResponse(
                success=False,
                error="No face detected. Ensure your face is well-lit and centred."
            )

        # Crop & preprocess face → 200×200 grayscale
        x, y, w, h = faces[0]
        face_roi = detector.crop_face(img_cv, (x, y, w, h))
        face_gray = detector.preprocess_face(face_roi)

        # Encode to JPEG bytes
        img_bytes = encode_image_to_bytes(face_gray)

        # Count existing photos for this student (DB query)
        existing_count = count_photos_in_db(student.id, db)
        photo_num = existing_count + 1

        # Save to training_photos table
        photo_row = models.TrainingPhoto(
            student_id=student.id,
            photo_num=photo_num,
            image_data=img_bytes,
        )
        db.add(photo_row)

        # Mark photo_samples_taken after 100 photos
        should_train = False
        if photo_num >= 100 and not student.photo_samples_taken:
            student.photo_samples_taken = True
            should_train = True

        db.commit()

        return schemas.SavePhotoResponse(
            success=True,
            count=photo_num,
            should_train=should_train,
            message=f"Photo {photo_num} saved to database."
        )

    except Exception as e:
        db.rollback()
        return schemas.SavePhotoResponse(success=False, error=str(e))


@router.post("/train", response_model=schemas.TrainResponse,
             summary="Train the face recognition model from DB photos")
def train_model(db: Session = Depends(get_db)):
    """
    Loads all TrainingPhoto rows from PostgreSQL, trains an LBPH model,
    and stores the model XML bytes back in PostgreSQL.
    """
    try:
        # Load all training photos from DB
        all_photos = db.query(
            models.TrainingPhoto.student_id,
            models.TrainingPhoto.image_data
        ).all()

        if len(all_photos) == 0:
            return schemas.TrainResponse(
                success=False,
                error="No training photos found. Capture photos for at least one student first."
            )

        if len(all_photos) < 10:
            return schemas.TrainResponse(
                success=False,
                error=f"Only {len(all_photos)} photos found. Need at least 10 to train."
            )

        # Train LBPH model
        recognizer = FaceRecognizer()
        photo_rows = [(row.student_id, bytes(row.image_data)) for row in all_photos]
        success, model_bytes, num_images, num_students, accuracy = \
            recognizer.train_from_db_rows(photo_rows)

        if not success or not model_bytes:
            return schemas.TrainResponse(
                success=False,
                error="Training failed. Check that photos contain detectable faces."
            )

        # Store model in DB (replace previous record)
        db.query(models.TrainingModel).delete()  # Keep only latest
        tm = models.TrainingModel(
            model_file='db',
            model_data=model_bytes,
            num_students=num_students,
            num_images=num_images,
            accuracy=accuracy,
        )
        db.add(tm)
        db.commit()

        return schemas.TrainResponse(
            success=True,
            num_images=num_images,
            num_students=num_students,
            accuracy=accuracy,
            message=f"Model trained on {num_images} photos from {num_students} students."
        )

    except Exception as e:
        db.rollback()
        return schemas.TrainResponse(success=False, error=str(e))


@router.post("/recognize", response_model=schemas.RecognizeResponse,
             summary="Recognize a face from an uploaded image")
def recognize_face(file: UploadFile = File(...), db: Session = Depends(get_db)):
    """
    Accepts an uploaded image, detects the face, runs LBPH recognition
    using the model loaded from PostgreSQL, and marks attendance.
    """
    try:
        # Read uploaded file
        img_bytes = file.file.read()
        np_arr = np.frombuffer(img_bytes, dtype=np.uint8)
        img_cv = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if img_cv is None:
            return schemas.RecognizeResponse(success=False, error="Could not decode image.")

        img_h, img_w = img_cv.shape[:2]

        # Detect face (accurate params for recognition)
        detector = FaceDetector()
        faces = detector.detect_faces(img_cv)

        if len(faces) == 0:
            return schemas.RecognizeResponse(
                success=True, recognized=False,
                img_w=img_w, img_h=img_h,
                message="No face detected in the image."
            )

        # Pick largest face
        faces_sorted = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
        x, y, w, h = faces_sorted[0]
        bbox = {"x": int(x), "y": int(y), "w": int(w), "h": int(h)}

        face_roi = detector.crop_face(img_cv, (x, y, w, h))
        face_gray = detector.preprocess_face(face_roi)

        # Load latest model from DB
        tm = get_latest_model(db)
        if not tm or not tm.model_data:
            return schemas.RecognizeResponse(
                success=False,
                error="No trained model found. Please train the model first."
            )

        recognizer = FaceRecognizer()
        if not recognizer.load_model_from_bytes(bytes(tm.model_data)):
            return schemas.RecognizeResponse(
                success=False,
                error="Failed to load recognition model from database."
            )

        student_db_id, confidence = recognizer.recognize_face(face_gray)

        if confidence < 55:
            return schemas.RecognizeResponse(
                success=True, recognized=False,
                confidence=round(confidence, 1),
                bbox=bbox, img_w=img_w, img_h=img_h,
                message=f"Unknown ({confidence:.1f}%)"
            )

        # Look up student
        student = db.query(models.Student).filter(
            models.Student.id == student_db_id
        ).first()
        if not student:
            return schemas.RecognizeResponse(
                success=True, recognized=False,
                bbox=bbox, img_w=img_w, img_h=img_h,
                message="Unknown"
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
                success=True, recognized=True,
                student_id=student.student_id,
                student_name=student.name,
                department=student.get_department_display(),
                confidence=round(confidence, 1),
                attendance_marked=False,
                already_marked=True,
                marked_at=existing.time,
                bbox=bbox, img_w=img_w, img_h=img_h,
                message=f"Already marked today at {existing.time}."
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
            success=True, recognized=True,
            student_id=student.student_id,
            student_name=student.name,
            department=student.get_department_display(),
            confidence=round(confidence, 1),
            attendance_marked=True,
            already_marked=False,
            bbox=bbox, img_w=img_w, img_h=img_h,
            message=f"Attendance marked for {student.name} ({student.student_id})."
        )

    except Exception as e:
        return schemas.RecognizeResponse(success=False, error=str(e))


@router.get("/photos/count", summary="Get photo count per student from DB")
def photo_counts(db: Session = Depends(get_db)):
    """Returns photo count for each student stored in the training_photos table."""
    from sqlalchemy import func as sqlfunc
    rows = db.query(
        models.TrainingPhoto.student_id,
        sqlfunc.count(models.TrainingPhoto.id).label('count')
    ).group_by(models.TrainingPhoto.student_id).all()

    result = {}
    for student_db_id, count in rows:
        student = db.query(models.Student).filter(
            models.Student.id == student_db_id
        ).first()
        if student:
            result[student.student_id] = count
    return result


@router.delete("/photos/{student_id}", summary="Delete all training photos for a student")
def delete_photos(student_id: str, db: Session = Depends(get_db)):
    """Deletes all training photos for a student from the DB."""
    student = db.query(models.Student).filter(
        models.Student.student_id == student_id
    ).first()
    if not student:
        return {"success": False, "error": f"Student '{student_id}' not found."}

    deleted = db.query(models.TrainingPhoto).filter(
        models.TrainingPhoto.student_id == student.id
    ).delete()
    student.photo_samples_taken = False
    db.commit()
    return {"success": True, "deleted": deleted}
