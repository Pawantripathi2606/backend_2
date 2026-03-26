# Face Recognition Attendance System — FastAPI.

A RESTful API backend for automated student attendance marking using facial recognition (LBPH algorithm).

## 🚀 Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the server
uvicorn main:app --reload
```

API runs at **http://127.0.0.1:8000**  
Swagger docs at **http://127.0.0.1:8000/docs**

---

## 📦 Tech Stack
- **FastAPI** — REST API framework
- **SQLAlchemy** — ORM with SQLite
- **OpenCV** — Face detection (Haar Cascade) + recognition (LBPH)
- **Pandas / OpenPyXL** — Excel report generation
- **ReportLab** — PDF report generation

---

## 📡 API Endpoints

### Students (`/students`)
| Method | Path | Description |
|--------|------|-------------|
| GET | `/students/` | List all students |
| POST | `/students/` | Create student |
| GET | `/students/{id}` | Get by DB id |
| GET | `/students/by-student-id/{sid}` | Get by student string ID |
| PUT | `/students/{id}` | Update student |
| DELETE | `/students/{id}` | Delete student |

### Recognition (`/recognition`)
| Method | Path | Description |
|--------|------|-------------|
| POST | `/recognition/save-photo` | Save base64 face photo |
| POST | `/recognition/train` | Train LBPH model |
| POST | `/recognition/recognize` | Recognize face from image upload |

### Attendance (`/attendance`)
| Method | Path | Description |
|--------|------|-------------|
| GET | `/attendance/` | List records (filterable) |
| GET | `/attendance/stats` | Dashboard statistics |
| DELETE | `/attendance/{id}` | Delete record |
| GET | `/attendance/export/excel` | Download Excel report |
| GET | `/attendance/export/pdf` | Download PDF report |
| GET | `/attendance/export/student-pdf/{id}` | Per-student PDF |
| POST | `/attendance/{id}/send-email` | Send email notification |
| POST | `/attendance/send-all-emails` | Send emails to all records |

---

## 📁 Project Structure
```
fastapi_app/
├── main.py              # App entry, CORS, DB init
├── database.py          # SQLAlchemy engine + session
├── models.py            # ORM models
├── schemas.py           # Pydantic schemas
├── requirements.txt
├── routers/
│   ├── students.py
│   ├── recognition.py
│   └── attendance.py
└── utils/
    └── face_utils.py    # FaceDetector + FaceRecognizer
```

## ⚙️ Environment Variables (optional `.env`)
```
EMAIL_HOST_USER=your_gmail@gmail.com
EMAIL_HOST_PASSWORD=your_app_password
```
