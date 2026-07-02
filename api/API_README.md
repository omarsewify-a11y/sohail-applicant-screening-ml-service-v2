# API_README — Sohail Applicant Screening API (v2)

## Overview

This FastAPI application serves Khaled's v2 scikit-learn Pipeline as a production-grade prediction service.
It handles single and batch predictions, logs every prediction for Omar's monitoring layer, and exposes a live results dashboard.

---

## What Changed from v1 to v2

| Feature | v1 | v2 |
|---|---|---|
| Model format | `model.joblib` + separate `encoder.joblib` | `model_v2.joblib` (full sklearn Pipeline) |
| Preprocessing | Manual LabelEncoder in API code | Handled inside the Pipeline (OneHotEncoder) |
| Engineered features | None | `experience_score`, `high_gpa` computed in API |
| Endpoints | `/health`, `/predict` | `/health`, `/predict`, `/predict-batch`, `/results` |
| Batch input | Not supported | CSV upload via `POST /predict-batch` |
| Prediction logging | None | Every prediction logged to `predictions_log.csv` |
| Results view | None | Live dashboard at `/results` |

---

## Project Structure

```
api/
├── main.py              # FastAPI v2 application
├── model_v2.joblib      # Khaled's v2 Pipeline (do NOT retrain)
├── sample_batch.csv     # Sample CSV for batch testing
└── API_README.md        # This file

predictions_log.csv      # Shared log — written by API, read by Omar's monitoring
```

---

## Requirements

```bash
pip install fastapi uvicorn pydantic scikit-learn==1.6.1 joblib pandas python-multipart
```

> `scikit-learn==1.6.1` must match the version Khaled used to train `model_v2.joblib`.

---

## How to Run

```bash
cd api
uvicorn main:app --reload
```

API available at: `http://127.0.0.1:8000`

Swagger docs: `http://127.0.0.1:8000/docs`

Results dashboard: `http://127.0.0.1:8000/results`

---

## Endpoints

### GET /health

Check if the API is running.

**Request:**
```bash
curl http://127.0.0.1:8000/health
```

**Response:**
```json
{
  "status": "ok",
  "message": "Sohail Applicant Screening API v2 is running."
}
```

---

### POST /predict

Single applicant prediction. Engineered features (`experience_score`, `high_gpa`) are computed automatically.

**Input fields (JSON body):**

| Field | Type | Required | Description |
|---|---|---|---|
| gpa | float | Yes | Applicant GPA (0.0 – 4.0) |
| skills_count | int | Yes | Number of technical skills |
| prior_projects | int | Yes | Number of completed projects |
| track | string | Yes | Internship track: "AI", "Data", or "Web" |

**Response fields:**

| Field | Type | Description |
|---|---|---|
| prediction | string | "Shortlisted" or "Review Later" |
| confidence | float | Model confidence score (0.0 – 1.0) |

**Example:**
```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"gpa": 3.8, "skills_count": 8, "prior_projects": 5, "track": "AI"}'
```

**Response:**
```json
{
  "prediction": "Shortlisted",
  "confidence": 0.8126
}
```

---

### POST /predict-batch

Batch prediction from a CSV file upload. Every row is predicted and logged.

**CSV format** (columns required):
```
gpa,skills_count,prior_projects,track
3.8,8,5,AI
2.5,2,0,Web
```

**Request:**
```bash
curl -X POST http://127.0.0.1:8000/predict-batch \
  -F "file=@sample_batch.csv"
```

**Response:**
```json
{
  "total": 10,
  "shortlisted": 6,
  "review_later": 4,
  "results": [
    {
      "gpa": 3.8,
      "skills_count": 8,
      "prior_projects": 5,
      "track": "AI",
      "experience_score": 13,
      "high_gpa": 1,
      "prediction": "Shortlisted",
      "confidence": 0.8126
    },
    ...
  ]
}
```

---

### GET /results

Live HTML dashboard showing the latest 50 predictions with summary stats.

Open in browser: `http://127.0.0.1:8000/results`

Displays:
- Total predictions, shortlisted count, review later count, average confidence
- Full prediction log table with all fields

---

## Prediction Log

Every prediction (single and batch) is automatically appended to `predictions_log.csv` at the project root.

**Log columns:**

| Column | Type | Description |
|---|---|---|
| timestamp | string | UTC datetime of prediction |
| gpa | float | Input GPA |
| skills_count | int | Input skills count |
| prior_projects | int | Input projects count |
| track | string | Input track |
| experience_score | int | Engineered: skills_count + prior_projects |
| high_gpa | int | Engineered: 1 if gpa >= 3.5, else 0 |
| prediction | string | "Shortlisted" or "Review Later" |
| confidence | float | Model confidence (0.0 – 1.0) |

Omar's monitoring script reads this file from `../predictions_log.csv`.

---

## Engineered Features

The API computes these automatically — you do not need to include them in input:

- `experience_score = skills_count + prior_projects`
- `high_gpa = 1 if gpa >= 3.5 else 0`

These match exactly what Khaled used during training.

---

## Notes

- `model_v2.joblib` is a full scikit-learn Pipeline — no manual preprocessing needed.
- The API never retrains the model — it only loads and serves it.
- `random_state=42` was used in training (Khaled's `train_model_v2.py`).
- Input validation handled by Pydantic — invalid inputs return `422 Unprocessable Entity`.

---

## Internship Project

AI Foundation Internship Program — Sohail Smart Solutions

Easa — API & Serving Layer (v2)
