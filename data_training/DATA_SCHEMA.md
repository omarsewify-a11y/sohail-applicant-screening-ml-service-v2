# DATA_SCHEMA (Version 2)

## Dataset

Applicant Screening Dataset Version 2

## Features

| Column | Type | Description |
|---------|------|-------------|
| gpa | Float | Applicant GPA (0.0–4.0) |
| skills_count | Integer | Number of technical skills |
| prior_projects | Integer | Number of completed projects |
| track | String | Internship track (AI, Data, Web) |
| experience_score | Integer | skills_count + prior_projects |
| high_gpa | Integer | 1 if GPA ≥ 3.5, otherwise 0 |

## Target

| Column | Type |
|---------|------|
| shortlisted | Integer (0 or 1) |

- 1 = Shortlisted
- 0 = Review Later

## Preprocessing

- OneHotEncoder is applied to the `track` column.
- Numeric features pass through unchanged.

## Model

- Logistic Regression
- Scikit-learn Pipeline

## Reproducibility

All train/test splits use:

```python
random_state = 42
```

## Model File

```python
model_v2.joblib
```
