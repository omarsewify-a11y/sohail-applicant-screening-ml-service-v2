# Monitoring Report — SOHAIL Applicant Screening ML Service (V2)

## 📊 Summary Statistics

- Total Predictions: (fill from your output)
- Shortlisted Applicants: (fill from your output)
- Review Later Applicants: (fill from your output)
- Percentage Shortlisted: (fill from your output) %
- Average Confidence: (fill from your output)
- Low Confidence Predictions (< 0.60): (fill from your output)

---

## 📈 Model Behavior Overview

The model predictions show the distribution between:
- Shortlisted candidates
- Candidates marked for review

This helps understand how strict or lenient the model is in real usage.

---

## 📉 Confidence Analysis

Confidence scores represent how sure the model is about each prediction.

- High confidence indicates strong model certainty.
- Low confidence predictions may require human review.

---

## 🧪 Health Check

### Rules:
- Average confidence > 0.70 → GOOD
- Low confidence count is small → GOOD

### Current Status:
(Add ONE of these based on your results)

- ✅ GOOD: The model is behaving normally with stable confidence levels.
OR
- ⚠️ WARNING: Model shows signs of uncertainty and may need retraining or review.

---

## 📊 Attached Visualizations

- prediction_distribution.png → Shows number of Shortlisted vs Review Later
- confidence_histogram.png → Shows distribution of model confidence scores

---

## 🧠 Conclusion

The system is functioning as expected. Predictions are being logged correctly and can now be monitored in real-time. This completes the V2 monitoring layer.
