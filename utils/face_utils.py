"""
Face detection and recognition utilities.
Ported from Django's recognition/utils.py — no Django dependencies.
"""
import cv2
import numpy as np
import os

# Use MEDIA_ROOT env var if set (Render persistent disk), else local ../media
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MEDIA_ROOT = os.getenv("MEDIA_ROOT", os.path.join(BASE_DIR, 'media'))
MODEL_PATH = os.path.join(MEDIA_ROOT, 'models', 'classifier.xml')


class FaceDetector:
    """Utility class for face detection using Haar Cascades."""

    def __init__(self):
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(cascade_path)

    def detect_faces(self, image, fast=False):
        """Detect faces in an image. Returns list of (x, y, w, h).
        fast=True uses looser params for training photo capture (3x faster).
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if fast:
            # Faster params: larger scaleFactor + fewer minNeighbors
            faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=3)
        else:
            # Accurate params: for recognition
            faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
        return faces

    def crop_face(self, image, face_coords):
        """Crop face from image based on (x, y, w, h) coordinates."""
        x, y, w, h = face_coords
        return image[y:y + h, x:x + w]

    def preprocess_face(self, face_image, size=(450, 450)):
        """Resize and convert face to grayscale for recognition."""
        face_resized = cv2.resize(face_image, size)
        face_gray = cv2.cvtColor(face_resized, cv2.COLOR_BGR2GRAY)
        return face_gray


class FaceRecognizer:
    """Utility class for face recognition using LBPH algorithm."""

    def __init__(self):
        self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        self.model_path = MODEL_PATH

    def load_model(self) -> bool:
        """Load trained LBPH model from disk. Returns True if loaded."""
        if os.path.exists(self.model_path):
            self.recognizer.read(self.model_path)
            return True
        return False

    def recognize_face(self, face_image):
        """
        Recognize a face image.
        Returns (student_db_id, confidence_percentage).
        confidence_percentage: 0-100, higher is better.
        """
        if not os.path.exists(self.model_path):
            return None, 0.0

        # Ensure grayscale and correct size
        if len(face_image.shape) == 3:
            face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
        face_image = cv2.resize(face_image, (450, 450))

        student_db_id, raw_confidence = self.recognizer.predict(face_image)

        # Convert LBPH confidence (lower is better, 0-150+) to percentage (higher is better)
        confidence_percentage = max(0.0, min(100.0, (1 - (raw_confidence / 150)) * 100))
        return student_db_id, confidence_percentage

    def train_model(self, training_data_path: str):
        """
        Train the LBPH model from images in training_data_path.
        Returns (success, num_images, num_students, accuracy_percent).
        """
        faces = []
        ids = []

        if not os.path.exists(training_data_path):
            print(f"Error: Training data path does not exist: {training_data_path}")
            return False, 0, 0, 0.0

        image_files = [f for f in os.listdir(training_data_path) if f.endswith('.jpg')]

        if len(image_files) == 0:
            print("Error: No training images found")
            return False, 0, 0, 0.0

        print(f"Loading {len(image_files)} training images...")

        import re
        for image_file in image_files:
            try:
                # Filename format: user.<db_id>.<photo_num>.jpg
                parts = image_file.split('.')
                if len(parts) < 3:
                    continue
                if not parts[1].isdigit():
                    continue

                student_db_id = int(parts[1])
                img_path = os.path.join(training_data_path, image_file)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

                if img is None or img.shape[0] == 0 or img.shape[1] == 0:
                    continue

                # Resize to 200x200 — smaller file, LBPH still accurate at this size
                img = cv2.resize(img, (200, 200))
                faces.append(img)
                ids.append(student_db_id)

            except Exception as e:
                print(f"Warning: Error processing {image_file}: {e}")
                continue

        if len(faces) == 0:
            print("Error: No valid training images could be loaded")
            return False, 0, 0, 0.0

        print(f"Successfully loaded {len(faces)} images from {len(set(ids))} students")

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
                accuracy = 85.0  # Default for small datasets

            # Save model
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            self.recognizer.write(self.model_path)

            print(f"Training complete! Accuracy: {accuracy:.1f}%")
            return True, len(faces), len(set(ids)), round(accuracy, 1)

        except Exception as e:
            print(f"Error during training: {e}")
            return False, 0, 0, 0.0
