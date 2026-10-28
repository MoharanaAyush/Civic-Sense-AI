# CivicSense — Intelligent Civic Issue Detection & Predictive Resolution System

A runnable educational MVP using FastAPI, scikit-learn, SQLite, and a browser frontend.

## Features
- Citizen complaint form with location, area type, severity, and optional photo.
- Naïve Bayes text classification for civic issue category.
- Decision Tree priority prediction.
- Random Forest Regression resolution-hours estimate.
- SQLite complaint storage and tracking.
- Admin dashboard with status updates and summary counts.
- Optional K-Means demo for grouping locations by complaint counts.
- FastAPI interactive API docs at `/docs`.

## Important data limitation
The initial model training phrases and priority/time training rows are synthetic, illustrative demo data. The resolution-time estimates are not validated against real municipal records and must not be treated as real predictions, emergency guidance, or official service-level commitments. Replace the demo training data with documented, labeled historical records and evaluate on a held-out test set before claiming model performance.

## Run on Windows
1. Install Python 3.11 or 3.12 from python.org and enable "Add Python to PATH" during installation.
2. Extract the ZIP to a folder, for example `C:\CivicSense`.
3. Open Command Prompt or the VS Code terminal in that folder.
4. Create a virtual environment:
   ```bat
   py -m venv .venv
   .venv\Scripts\activate
   ```
5. Install requirements:
   ```bat
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```
6. Start the server:
   ```bat
   uvicorn main:app --reload
   ```
7. Open `http://127.0.0.1:8000` in your browser. API documentation: `http://127.0.0.1:8000/docs`.

The database and trained demo models are created automatically at first startup.

## API endpoints
- `GET /api/health`
- `POST /api/complaints` (multipart form)
- `GET /api/complaints`
- `GET /api/complaints/{id}`
- `PATCH /api/complaints/{id}/status?status=In%20Progress`
- `GET /api/stats`
- `GET /api/clusters`

## Next steps for a production-grade system
- Add administrator authentication and role-based access control before exposing status updates publicly.
- Use HTTPS, secure upload validation, malware scanning, rate limiting, and robust privacy controls.
- Add location coordinates and geographic clustering with a suitable distance metric.
- Replace synthetic data with reliable labeled civic data; evaluate classification and regression models on held-out data.
- Add duplicate-report detection and notifications only after configuration and testing.
- Add a true image classifier only with a suitable labeled image dataset.
