
import os
import numpy as np
import joblib

from sklearn.ensemble import IsolationForest

FEATURES_PATH = "models/ood_reference_features.npz"
OUTPUT_PATH = "models/ood_detector.pkl"

print("Loading extracted features...")

data = np.load(FEATURES_PATH)

train_features = data["train_features"]
validation_features = data["validation_features"]

print("Training feature shape:", train_features.shape)

print("\nTraining Isolation Forest...")

detector = IsolationForest(
    n_estimators=200,
    contamination="auto",
    random_state=42,
    n_jobs=-1
)

detector.fit(train_features)
joblib.dump(detector, OUTPUT_PATH)

print("\nOOD detector training completed.")
print("Saved to:", OUTPUT_PATH)

print("\nEvaluating validation images...")

scores = detector.decision_function(validation_features)

print("Validation images:", len(scores))
print("Minimum score:", round(float(np.min(scores)), 4))
print("Maximum score:", round(float(np.max(scores)), 4))
print("Mean score:", round(float(np.mean(scores)), 4))
print("1st percentile:", round(float(np.percentile(scores, 1)), 4))
print("5th percentile:", round(float(np.percentile(scores, 5)), 4))

print("\nValidation rejection rates:")

for threshold in [0.00, -0.02, -0.05, -0.10]:
    flagged = np.sum(scores < threshold)
    percentage = flagged / len(scores) * 100

    print(
        f"Threshold {threshold}: "
        f"{flagged}/{len(scores)} "
        f"({percentage:.2f}%) flagged"
    )
