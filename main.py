import os
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

BASE = Path(__file__).resolve().parent
STATIC = BASE / "static"
UPLOADS = BASE / "uploads"
DATA = BASE / "civic.db"
MODEL_DIR = BASE / "models"
UPLOADS.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)

app = FastAPI(title="Intelligent Civic Issue Detection & Predictive Resolution System", version="1.0.0")
app.mount("/static", StaticFiles(directory=STATIC), name="static")
app.mount("/uploads", StaticFiles(directory=UPLOADS), name="uploads")

CATEGORIES = ["Road", "Garbage", "Water", "Drainage", "Electricity", "Other"]
PRIORITIES = ["Low", "Medium", "High", "Critical"]
STATUSES = ["Submitted", "Assigned", "In Progress", "Resolved", "Rejected"]

def db():
    con = sqlite3.connect(DATA)
    con.row_factory = sqlite3.Row
    return con

def init_db():
    with db() as con:
        con.execute("""
        CREATE TABLE IF NOT EXISTS complaints (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            description TEXT NOT NULL,
            location TEXT NOT NULL,
            area_type TEXT NOT NULL,
            severity TEXT NOT NULL,
            category TEXT NOT NULL,
            priority TEXT NOT NULL,
            estimated_hours REAL NOT NULL,
            status TEXT NOT NULL DEFAULT 'Submitted',
            department TEXT NOT NULL,
            photo TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
        """)

def train_models():
    model_dir = MODEL_DIR
    text_path = model_dir / "category_model.joblib"
    priority_path = model_dir / "priority_model.joblib"
    time_path = model_dir / "time_model.joblib"
    if all(p.exists() for p in [text_path, priority_path, time_path]):
        return

    examples = {
        "Road": [
            "deep pothole on the road", "damaged road surface", "broken pavement near school",
            "road has large cracks", "open manhole on street", "damaged footpath"
        ],
        "Garbage": [
            "garbage bin is overflowing", "waste has not been collected", "trash dumped on street",
            "bad smell from garbage", "litter is everywhere", "municipal garbage pickup missed"
        ],
        "Water": [
            "water pipe is leaking", "no water supply in my area", "drinking water is dirty",
            "pipeline burst on road", "water is flowing continuously", "low water pressure"
        ],
        "Drainage": [
            "drain is blocked", "sewage overflowing in street", "waterlogging after rain",
            "blocked storm drain", "dirty water overflowing from sewer", "drain cover is broken"
        ],
        "Electricity": [
            "streetlight is not working", "electric pole wire is hanging", "street lights are off",
            "exposed electrical wire in public area", "damaged street lamp", "sparking from electric pole"
        ],
        "Other": [
            "public park equipment is broken", "traffic sign is damaged", "public bench is broken",
            "noise from public construction", "damaged bus stop shelter", "tree branch blocking walkway"
        ],
    }
    texts, labels = [], []
    for cat, lines in examples.items():
        for line in lines:
            texts.append(line)
            labels.append(cat)
    # Repeat small variations so the baseline demo model can fit common terms.
    category_model = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), strip_accents="unicode")),
        ("model", __import__("sklearn.naive_bayes", fromlist=["MultinomialNB"]).MultinomialNB())
    ])
    category_model.fit(texts, labels)
    joblib.dump(category_model, text_path)

    rows = []
    for cat in CATEGORIES:
        for sev in ["Low", "Medium", "High"]:
            for area in ["Residential", "Commercial", "School/Hospital", "Roadside"]:
                for count in [0, 1, 3, 7, 12]:
                    if sev == "High" and (cat in ["Road", "Water", "Drainage", "Electricity"]):
                        pri = "Critical" if area == "School/Hospital" or count >= 7 else "High"
                    elif sev == "High":
                        pri = "High"
                    elif sev == "Medium" and count >= 7:
                        pri = "High"
                    elif sev == "Low" and count < 3:
                        pri = "Low"
                    else:
                        pri = "Medium"
                    rows.append([cat, sev, area, count, pri])
    df = pd.DataFrame(rows, columns=["category", "severity", "area_type", "similar_count", "priority"])
    pre = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), ["category", "severity", "area_type"]),
        ("num", "passthrough", ["similar_count"])
    ])
    pm = Pipeline([("pre", pre), ("model", DecisionTreeClassifier(max_depth=6, random_state=42))])
    pm.fit(df[["category", "severity", "area_type", "similar_count"]], df["priority"])
    joblib.dump(pm, priority_path)

    # Synthetic demonstration target only; replace with real resolved-case history for meaningful estimates.
    time_rows = []
    base_hours = {"Road": 40, "Garbage": 8, "Water": 18, "Drainage": 24, "Electricity": 16, "Other": 20}
    sev_add = {"Low": 0, "Medium": 8, "High": 24}
    for cat in CATEGORIES:
        for sev in ["Low", "Medium", "High"]:
            for area in ["Residential", "Commercial", "School/Hospital", "Roadside"]:
                for count in [0, 1, 3, 7, 12]:
                    hours = base_hours[cat] + sev_add[sev] + (10 if area == "School/Hospital" else 0) + min(count, 12) * 0.8
                    time_rows.append([cat, sev, area, count, hours])
    td = pd.DataFrame(time_rows, columns=["category", "severity", "area_type", "similar_count", "hours"])
    tr = Pipeline([
        ("pre", ColumnTransformer([
            ("cat", OneHotEncoder(handle_unknown="ignore"), ["category", "severity", "area_type"]),
            ("num", "passthrough", ["similar_count"])
        ])),
        ("model", RandomForestRegressor(n_estimators=80, random_state=42))
    ])
    tr.fit(td[["category", "severity", "area_type", "similar_count"]], td["hours"])
    joblib.dump(tr, time_path)

@app.on_event("startup")
def startup():
    init_db()
    train_models()

@app.get("/")
def home():
    return FileResponse(STATIC / "index.html")

@app.get("/api/health")
def health():
    return {"status": "ok", "app": app.title, "models": ["text category classifier", "priority classifier", "resolution-time regressor"]}

def model(name):
    path = MODEL_DIR / name
    if not path.exists():
        train_models()
    return joblib.load(path)

def infer_category(description):
    pred = model("category_model.joblib").predict([description])[0]
    return pred if pred in CATEGORIES else "Other"

def predict_priority(category, severity, area_type, similar_count):
    frame = pd.DataFrame([{
        "category": category, "severity": severity, "area_type": area_type,
        "similar_count": similar_count
    }])
    return str(model("priority_model.joblib").predict(frame)[0])

def predict_hours(category, severity, area_type, similar_count):
    frame = pd.DataFrame([{
        "category": category, "severity": severity, "area_type": area_type,
        "similar_count": similar_count
    }])
    return max(1, round(float(model("time_model.joblib").predict(frame)[0]), 1))

def department_for(category):
    return {
        "Road": "Roads & Public Works", "Garbage": "Sanitation",
        "Water": "Water Supply", "Drainage": "Drainage & Sewerage",
        "Electricity": "Street Lighting / Electrical", "Other": "General Civic Services"
    }.get(category, "General Civic Services")

@app.post("/api/complaints")
async def create_complaint(
    name: str = Form(...),
    description: str = Form(...),
    location: str = Form(...),
    area_type: str = Form(...),
    severity: str = Form(...),
    similar_count: int = Form(0),
    photo: UploadFile | None = File(None)
):
    name, description, location = name.strip(), description.strip(), location.strip()
    if not name or not description or not location:
        raise HTTPException(400, "Name, description, and location are required.")
    if len(description) > 2000 or len(location) > 300 or len(name) > 100:
        raise HTTPException(400, "One or more fields are too long.")
    if area_type not in ["Residential", "Commercial", "School/Hospital", "Roadside"]:
        raise HTTPException(400, "Invalid area type.")
    if severity not in ["Low", "Medium", "High"]:
        raise HTTPException(400, "Invalid severity.")
    similar_count = max(0, min(int(similar_count), 1000))
    photo_url = None
    if photo and photo.filename:
        ext = Path(photo.filename).suffix.lower()
        if ext not in [".jpg", ".jpeg", ".png", ".webp"]:
            raise HTTPException(400, "Photo must be JPG, PNG, or WEBP.")
        content = await photo.read()
        if len(content) > 5 * 1024 * 1024:
            raise HTTPException(400, "Photo must be 5 MB or smaller.")
        safe_name = f"{uuid.uuid4().hex}{ext}"
        (UPLOADS / safe_name).write_bytes(content)
        photo_url = f"/uploads/{safe_name}"

    category = infer_category(description)
    priority = predict_priority(category, severity, area_type, similar_count)
    hours = predict_hours(category, severity, area_type, similar_count)
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with db() as con:
        cur = con.execute("""
            INSERT INTO complaints
            (name, description, location, area_type, severity, category, priority,
             estimated_hours, status, department, photo, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'Submitted', ?, ?, ?, ?)
        """, (name, description, location, area_type, severity, category, priority,
              hours, department_for(category), photo_url, now, now))
        cid = cur.lastrowid
        row = con.execute("SELECT * FROM complaints WHERE id=?", (cid,)).fetchone()
    return dict(row)

@app.get("/api/complaints")
def get_complaints(status: str | None = None):
    with db() as con:
        if status and status in STATUSES:
            rows = con.execute("SELECT * FROM complaints WHERE status=? ORDER BY id DESC", (status,)).fetchall()
        else:
            rows = con.execute("SELECT * FROM complaints ORDER BY id DESC").fetchall()
    return [dict(r) for r in rows]

@app.get("/api/complaints/{cid}")
def get_complaint(cid: int):
    with db() as con:
        row = con.execute("SELECT * FROM complaints WHERE id=?", (cid,)).fetchone()
    if not row:
        raise HTTPException(404, "Complaint not found.")
    return dict(row)

@app.patch("/api/complaints/{cid}/status")
def update_status(cid: int, status: str):
    if status not in STATUSES:
        raise HTTPException(400, f"Status must be one of: {', '.join(STATUSES)}")
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with db() as con:
        cur = con.execute("UPDATE complaints SET status=?, updated_at=? WHERE id=?", (status, now, cid))
        if cur.rowcount == 0:
            raise HTTPException(404, "Complaint not found.")
        row = con.execute("SELECT * FROM complaints WHERE id=?", (cid,)).fetchone()
    return dict(row)

@app.get("/api/stats")
def stats():
    with db() as con:
        total = con.execute("SELECT COUNT(*) n FROM complaints").fetchone()["n"]
        pending = con.execute("SELECT COUNT(*) n FROM complaints WHERE status IN ('Submitted','Assigned','In Progress')").fetchone()["n"]
        resolved = con.execute("SELECT COUNT(*) n FROM complaints WHERE status='Resolved'").fetchone()["n"]
        critical = con.execute("SELECT COUNT(*) n FROM complaints WHERE priority='Critical' AND status!='Resolved'").fetchone()["n"]
        cats = con.execute("SELECT category, COUNT(*) count FROM complaints GROUP BY category ORDER BY count DESC").fetchall()
        statuses = con.execute("SELECT status, COUNT(*) count FROM complaints GROUP BY status").fetchall()
    return {"total": total, "pending": pending, "resolved": resolved, "critical_open": critical,
            "by_category": [dict(r) for r in cats], "by_status": [dict(r) for r in statuses]}

@app.get("/api/clusters")
def clusters():
    # Optional unsupervised-learning demo: clusters complaint locations by their observed issue counts.
    with db() as con:
        rows = con.execute("SELECT location, COUNT(*) count FROM complaints GROUP BY location").fetchall()
    if len(rows) < 3:
        return {"message": "Submit complaints from at least three different locations to see area clusters.", "clusters": []}
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    df = pd.DataFrame([dict(r) for r in rows])
    X = df[["count"]].values
    k = min(3, len(df))
    df["cluster"] = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(StandardScaler().fit_transform(X))
    return {"clusters": df.to_dict(orient="records"), "note": "Demo clusters use complaint counts by entered location; standardize location names for better grouping."}
