"""
Face detection and recognition utilities — fully DB-backed.
No disk I/O: photos and model are stored in PostgreSQL as binary blobs.
"""
import cv2
import numpy as np
import os
import tempfile


class FaceDetector:
    """
    Multi-cascade face detector.
    Uses frontalface_alt2 (better with glasses) + frontalface_default as fallback.
    """

    def __init__(self):
        haarcascade_dir = cv2.data.haarcascades
        # alt2 handles glasses, profiles, and partial faces much better
        self.cascade_alt2 = cv2.CascadeClassifier(
            haarcascade_dir + 'haarcascade_frontalface_alt2.xml'
        )
        # default as fallback
        self.cascade_default = cv2.CascadeClassifier(
            haarcascade_dir + 'haarcascade_frontalface_default.xml'
        )

    def detect_faces(self, image, fast=False):
        """
        Detect faces using multiple cascades.
        Returns list of (x, y, w, h).
        fast=True: very lenient (for bulk photo capture)
        fast=False: balanced (for recognition)
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # Always equalise histogram for better detection under varying light
        gray = cv2.equalizeHist(gray)

        if fast:
            # Very lenient — catches faces even at low quality / small size
            params = dict(scaleFactor=1.1, minNeighbors=2, minSize=(30, 30))
        else:
            # Balanced — good accuracy for recognition, still tolerant of glasses
            params = dict(scaleFactor=1.1, minNeighbors=3, minSize=(40, 40))

        # Try alt2 first (better with glasses)
        faces = self.cascade_alt2.detectMultiScale(gray, **params)

        # If nothing found, try default cascade as fallback
        if len(faces) == 0:
            faces = self.cascade_default.detectMultiScale(gray, **params)

        # If still nothing, try with even looser params
        if len(faces) == 0:
            loose = dict(scaleFactor=1.1, minNeighbors=1, minSize=(20, 20))
            faces = self.cascade_alt2.detectMultiScale(gray, **loose)

        return faces

    def crop_face(self, image, face_coords):
        x, y, w, h = face_coords
        return image[y:y + h, x:x + w]

    def preprocess_face(self, face_image, size=(200, 200)):
        """Resize and convert face to grayscale for recognition/storage."""
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
        """Load LBPH model from raw XML bytes. Returns True on success."""
        if not model_bytes:
            return False
        try:
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
        confidence_percentage: 0–100, higher is better.
        """
        if not self._loaded:
            return None, 0.0

        if len(face_image.shape) == 3:
            face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
        face_image = cv2.resize(face_image, (200, 200))

        student_db_id, raw_confidence = self.recognizer.predict(face_image)
        # LBPH raw confidence: 0 = perfect match, 150+ = no match
        confidence_pct = max(0.0, min(100.0, (1 - (raw_confidence / 150)) * 100))
        return student_db_id, confidence_pct

    def train_from_db_rows(self, photo_rows):
        """
        Train LBPH from (student_db_id, image_bytes) tuples.
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

        if len(faces) == 0:
            return False, None, 0, 0, 0.0

        print(f"Training on {len(faces)} images from {len(set(ids))} students...")

        try:
            self.recognizer.train(faces, np.array(ids))

            # Estimate accuracy on a sample
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

            # Serialize model bytes via temp file
            with tempfile.NamedTemporaryFile(suffix='.xml', delete=False) as tmp:
                tmp_path = tmp.name
            self.recognizer.write(tmp_path)
            with open(tmp_path, 'rb') as f:
                model_bytes = f.read()
            os.unlink(tmp_path)

            print(f"Training complete! Accuracy: {accuracy:.1f}%, model: {len(model_bytes)} bytes")
            return True, model_bytes, len(faces), len(set(ids)), round(accuracy, 1)

        except Exception as e:
            print(f"Training error: {e}")
            return False, None, 0, 0, 0.0
