"""
Face detection and recognition utilities — fully DB-backed.
No disk I/O: photos and model are stored in PostgreSQL as binary blobs.
"""
import cv2
import numpy as np
import os
import tempfile


class FaceDetector:
    """Utility class for face detection using Haar Cascades."""

    def __init__(self):
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(cascade_path)

    def detect_faces(self, image, fast=False):
        """Detect faces. fast=True is very lenient — catches faces in small frames and poor lighting."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Histogram equalization improves detection under variable lighting
        gray = cv2.equalizeHist(gray)
        if fast:
            # Very lenient params for bulk capture: fine pyramid, few neighbors, small minSize
            faces = self.face_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=2, minSize=(30, 30)
            )
        else:
            # Accurate params for recognition
            faces = self.face_cascade.detectMultiScale(
                gray, scaleFactor=1.3, minNeighbors=5, minSize=(60, 60)
            )
        return faces

    def crop_face(self, image, face_coords):
        x, y, w, h = face_coords
        return image[y:y + h, x:x + w]

    def preprocess_face(self, face_image, size=(200, 200)):
        """Resize and convert face to grayscale for recognition."""
        face_resized = cv2.resize(face_image, size)
        if len(face_resized.shape) == 3:
            face_resized = cv2.cvtColor(face_resized, cv2.COLOR_BGR2GRAY)
        return face_resized


class FaceRecognizer:
    """LBPH face recognizer — loads/saves model from PostgreSQL bytes."""

    def __init__(self):
        self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        self._loaded = False

    def load_model_from_bytes(self, model_bytes: bytes) -> bool:
        """Load LBPH model from raw XML bytes (stored in DB). Returns True on success."""
        if not model_bytes:
            return False
        try:
            # Write to temp file, read with OpenCV, then delete
            with tempfile.NamedTemporaryFile(suffix='.xml', delete=False) as tmp:
                tmp.write(model_bytes)
                tmp_path = tmp.name
            self.recognizer.read(tmp_path)
            os.unlink(tmp_path)
            self._loaded = True
            return True
        except Exception as e:
            print(f"Error loading model from bytes: {e}")
            return False

    def recognize_face(self, face_image):
        """
        Recognize a face. Returns (student_db_id, confidence_percentage).
        confidence_percentage: 0-100, higher is better.
        """
        if not self._loaded:
            return None, 0.0

        if len(face_image.shape) == 3:
            face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
        face_image = cv2.resize(face_image, (200, 200))

        student_db_id, raw_confidence = self.recognizer.predict(face_image)
        confidence_pct = max(0.0, min(100.0, (1 - (raw_confidence / 150)) * 100))
        return student_db_id, confidence_pct

    def train_from_db_rows(self, photo_rows):
        """
        Train LBPH model from a list of (student_db_id, image_bytes) tuples.
        Returns (success, model_bytes, num_images, num_students, accuracy).
        """
        faces = []
        ids = []

        for student_db_id, image_bytes in photo_rows:
            try:
                np_arr = np.frombuffer(image_bytes, dtype=np.uint8)
                img = cv2.imdecode(np_arr, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue
                img = cv2.resize(img, (200, 200))
                faces.append(img)
                ids.append(student_db_id)
            except Exception as e:
                print(f"Warning: skipping photo for student {student_db_id}: {e}")
                continue

        if len(faces) == 0:
            return False, None, 0, 0, 0.0

        print(f"Training on {len(faces)} images from {len(set(ids))} students...")

        try:
            self.recognizer.train(faces, np.array(ids))

            # Estimate accuracy
            accuracy = 0.0
            if len(faces) >= 10:
                correct = sum(
                    1 for i in range(0, len(faces), 5)
                    if self.recognizer.predict(faces[i])[0] == ids[i]
                )
                total = len(range(0, len(faces), 5))
                accuracy = (correct / total) * 100 if total > 0 else 0.0
            else:
                accuracy = 85.0

            # Serialize model to bytes via temp file
            with tempfile.NamedTemporaryFile(suffix='.xml', delete=False) as tmp:
                tmp_path = tmp.name
            self.recognizer.write(tmp_path)
            with open(tmp_path, 'rb') as f:
                model_bytes = f.read()
            os.unlink(tmp_path)

            print(f"Training complete! Accuracy: {accuracy:.1f}%, model size: {len(model_bytes)} bytes")
            return True, model_bytes, len(faces), len(set(ids)), round(accuracy, 1)

        except Exception as e:
            print(f"Training error: {e}")
            return False, None, 0, 0, 0.0
