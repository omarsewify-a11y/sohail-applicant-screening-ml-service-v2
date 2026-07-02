# Version 2 Improvements

Compared to Version 1, the following improvements were made:

## Dataset

- Expanded the applicant dataset to 50 records.

## Feature Engineering

Added:

- experience_score
- high_gpa

These engineered features provide additional information for the model.

## Pipeline

Version 2 replaces manual preprocessing with a Scikit-learn Pipeline that combines preprocessing and model training into one reusable object.

## Cross Validation

Added 5-fold cross-validation to evaluate model stability and report the mean accuracy.

## Model Versioning

The trained model is now saved as:

- model_v2.joblib

The original Version 1 model remains unchanged.

## Documentation

Updated DATA_SCHEMA.md to reflect the new engineered features and pipeline workflow.
