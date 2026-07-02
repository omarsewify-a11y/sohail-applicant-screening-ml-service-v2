import joblib
import pandas as pd
import csv
import os
from datetime import datetime
from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from typing import Literal
import io

# Load Khaled's v2 Pipeline — never retrain
model = joblib.load("model_v2.joblib")

LOG_FILE = "../predictions_log.csv"
LOG_COLUMNS = ["timestamp", "gpa", "skills_count", "prior_projects", "track",
               "experience_score", "high_gpa", "prediction", "confidence"]

# Create log file with headers if it doesn't exist
if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=LOG_COLUMNS)
        writer.writeheader()

app = FastAPI(
    title="Sohail Applicant Screening API",
    description="v2 — Batch prediction, logging, and results dashboard for internship applicant screening.",
    version="2.0.0"
)


# Input schema — base features; engineered features computed internally
class ApplicantInput(BaseModel):
    gpa: float = Field(..., ge=0.0, le=4.0, description="Applicant GPA (0.0 - 4.0)")
    skills_count: int = Field(..., ge=0, description="Number of technical skills")
    prior_projects: int = Field(..., ge=0, description="Number of completed projects")
    track: Literal["AI", "Data", "Web"] = Field(..., description="Internship track: AI, Data, or Web")


class PredictionOutput(BaseModel):
    prediction: str
    confidence: float


def engineer_features(gpa: float, skills_count: int, prior_projects: int) -> dict:
    """Compute the two engineered features Khaled added in v2."""
    return {
        "experience_score": skills_count + prior_projects,
        "high_gpa": 1 if gpa >= 3.5 else 0
    }


def run_prediction(gpa, skills_count, prior_projects, track):
    """Run prediction through Khaled's v2 Pipeline and return label + confidence."""
    eng = engineer_features(gpa, skills_count, prior_projects)
    features = pd.DataFrame([{
        "gpa": gpa,
        "skills_count": skills_count,
        "prior_projects": prior_projects,
        "experience_score": eng["experience_score"],
        "high_gpa": eng["high_gpa"],
        "track": track
    }])
    prediction_int = model.predict(features)[0]
    confidence = float(max(model.predict_proba(features)[0]))
    label = "Shortlisted" if prediction_int == 1 else "Review Later"
    return label, round(confidence, 4), eng


def log_prediction(gpa, skills_count, prior_projects, track, eng, label, confidence):
    """Append one prediction row to the shared predictions_log.csv."""
    row = {
        "timestamp": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
        "gpa": gpa,
        "skills_count": skills_count,
        "prior_projects": prior_projects,
        "track": track,
        "experience_score": eng["experience_score"],
        "high_gpa": eng["high_gpa"],
        "prediction": label,
        "confidence": confidence
    }
    with open(LOG_FILE, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=LOG_COLUMNS)
        writer.writerow(row)


# ── Endpoints ──────────────────────────────────────────────────────────────────

@app.get("/health")
def health_check():
    return {"status": "ok", "message": "Sohail Applicant Screening API v2 is running."}


@app.post("/predict", response_model=PredictionOutput)
def predict(applicant: ApplicantInput):
    """Single applicant prediction."""
    label, confidence, eng = run_prediction(
        applicant.gpa, applicant.skills_count, applicant.prior_projects, applicant.track
    )
    log_prediction(applicant.gpa, applicant.skills_count, applicant.prior_projects,
                   applicant.track, eng, label, confidence)
    return PredictionOutput(prediction=label, confidence=confidence)


@app.post("/predict-batch")
async def predict_batch(file: UploadFile = File(...)):
    """
    Batch prediction from a CSV file.
    CSV must have columns: gpa, skills_count, prior_projects, track
    Returns predictions for all rows.
    """
    contents = await file.read()
    df = pd.read_csv(io.BytesIO(contents))

    required_cols = {"gpa", "skills_count", "prior_projects", "track"}
    missing = required_cols - set(df.columns)
    if missing:
        return {"error": f"Missing columns in CSV: {missing}"}

    results = []
    for _, row in df.iterrows():
        label, confidence, eng = run_prediction(
            row["gpa"], int(row["skills_count"]), int(row["prior_projects"]), row["track"]
        )
        log_prediction(row["gpa"], int(row["skills_count"]), int(row["prior_projects"]),
                       row["track"], eng, label, confidence)
        results.append({
            "gpa": row["gpa"],
            "skills_count": int(row["skills_count"]),
            "prior_projects": int(row["prior_projects"]),
            "track": row["track"],
            "experience_score": eng["experience_score"],
            "high_gpa": eng["high_gpa"],
            "prediction": label,
            "confidence": confidence
        })

    shortlisted = sum(1 for r in results if r["prediction"] == "Shortlisted")
    review_later = len(results) - shortlisted

    return {
        "total": len(results),
        "shortlisted": shortlisted,
        "review_later": review_later,
        "results": results
    }


@app.get("/results", response_class=HTMLResponse)
def results_dashboard():
    """Simple HTML dashboard showing the latest predictions from the log."""
    if not os.path.exists(LOG_FILE):
        rows = []
    else:
        df = pd.read_csv(LOG_FILE)
        rows = df.tail(50).to_dict(orient="records")

    total = len(rows)
    shortlisted = sum(1 for r in rows if r.get("prediction") == "Shortlisted")
    review_later = total - shortlisted
    avg_conf = round(sum(r.get("confidence", 0) for r in rows) / total, 4) if total > 0 else 0

    rows_html = ""
    for r in reversed(rows):
        badge_color = "#22c55e" if r.get("prediction") == "Shortlisted" else "#f59e0b"
        rows_html += f"""
        <tr>
            <td>{r.get('timestamp', '')}</td>
            <td>{r.get('gpa', '')}</td>
            <td>{r.get('skills_count', '')}</td>
            <td>{r.get('prior_projects', '')}</td>
            <td>{r.get('track', '')}</td>
            <td>{r.get('experience_score', '')}</td>
            <td>{r.get('high_gpa', '')}</td>
            <td><span style="background:{badge_color};color:#fff;padding:2px 10px;border-radius:12px;font-size:0.85em">{r.get('prediction', '')}</span></td>
            <td>{r.get('confidence', '')}</td>
        </tr>"""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Sohail Applicant Screening — Results Dashboard</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: 'Segoe UI', sans-serif; background: #f1f5f9; color: #1e293b; }}
  header {{ background: #1e3a5f; color: white; padding: 24px 32px; }}
  header h1 {{ font-size: 1.5rem; }}
  header p {{ font-size: 0.9rem; opacity: 0.75; margin-top: 4px; }}
  .stats {{ display: flex; gap: 16px; padding: 24px 32px; flex-wrap: wrap; }}
  .card {{ background: white; border-radius: 10px; padding: 20px 28px; flex: 1; min-width: 160px;
           box-shadow: 0 1px 4px rgba(0,0,0,0.08); }}
  .card .label {{ font-size: 0.8rem; color: #64748b; text-transform: uppercase; letter-spacing: 0.05em; }}
  .card .value {{ font-size: 2rem; font-weight: 700; margin-top: 4px; }}
  .card.green .value {{ color: #16a34a; }}
  .card.amber .value {{ color: #d97706; }}
  .card.blue .value {{ color: #2563eb; }}
  .card.slate .value {{ color: #475569; }}
  .table-wrap {{ padding: 0 32px 32px; overflow-x: auto; }}
  h2 {{ font-size: 1rem; color: #475569; margin-bottom: 12px; }}
  table {{ width: 100%; border-collapse: collapse; background: white;
           border-radius: 10px; overflow: hidden;
           box-shadow: 0 1px 4px rgba(0,0,0,0.08); }}
  th {{ background: #1e3a5f; color: white; padding: 12px 14px; text-align: left;
        font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.04em; }}
  td {{ padding: 11px 14px; font-size: 0.88rem; border-bottom: 1px solid #f1f5f9; }}
  tr:last-child td {{ border-bottom: none; }}
  tr:hover td {{ background: #f8fafc; }}
</style>
</head>
<body>
<header>
  <h1>Sohail Applicant Screening — Results Dashboard</h1>
  <p>v2 Production API &nbsp;|&nbsp; Showing latest {min(total, 50)} predictions</p>
</header>
<div class="stats">
  <div class="card slate"><div class="label">Total Predictions</div><div class="value">{total}</div></div>
  <div class="card green"><div class="label">Shortlisted</div><div class="value">{shortlisted}</div></div>
  <div class="card amber"><div class="label">Review Later</div><div class="value">{review_later}</div></div>
  <div class="card blue"><div class="label">Avg Confidence</div><div class="value">{avg_conf}</div></div>
</div>
<div class="table-wrap">
  <h2>Prediction Log</h2>
  <table>
    <thead>
      <tr>
        <th>Timestamp</th><th>GPA</th><th>Skills</th><th>Projects</th>
        <th>Track</th><th>Exp Score</th><th>High GPA</th><th>Prediction</th><th>Confidence</th>
      </tr>
    </thead>
    <tbody>{rows_html if rows_html else '<tr><td colspan="9" style="text-align:center;color:#94a3b8;padding:32px">No predictions yet. Run /predict or /predict-batch to get started.</td></tr>'}</tbody>
  </table>
</div>
</body>
</html>"""
    return HTMLResponse(content=html)
