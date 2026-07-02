import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("../predictions_log.csv")

# Stats
total_predictions = len(df)
shortlisted = (df["prediction"] == "Shortlisted").sum()
review_later = (df["prediction"] == "Review Later").sum()

percentage_shortlisted = (shortlisted / total_predictions) * 100
average_confidence = df["confidence"].mean()
low_confidence = (df["confidence"] < 0.60).sum()

# Chart 1
df["prediction"].value_counts().plot(kind="bar")
plt.title("Prediction Distribution")
plt.savefig("prediction_distribution.png")
plt.close()

# Chart 2
plt.hist(df["confidence"], bins=10)
plt.title("Confidence Distribution")
plt.savefig("confidence_histogram.png")
plt.close()

# Report
report = f"""
# Monitoring Report

Total Predictions: {total_predictions}
Shortlisted: {shortlisted}
Review Later: {review_later}
Percentage Shortlisted: {round(percentage_shortlisted,2)}%
Average Confidence: {round(average_confidence,2)}
Low Confidence: {low_confidence}

Health Check:
{"GOOD" if average_confidence > 0.7 else "WARNING"}
"""

with open("MONITORING_REPORT.md", "w") as f:
    f.write(report)

print("Monitoring complete!")
