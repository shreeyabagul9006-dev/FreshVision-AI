
import os
import json
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


IMAGE_SIZE = (160, 160)
BATCH_SIZE = 32

# Load trained model
model = tf.keras.models.load_model(
    "models/freshvision_model.keras"
)

# Load class names
with open("models/class_names.json", "r") as file:
    class_names = json.load(file)

# Load test dataset
test_ds = tf.keras.utils.image_dataset_from_directory(
    "data/dataset/Test",
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_names=class_names,
    label_mode="int",
    shuffle=False
)

# Collect actual labels
y_true = np.concatenate([
    labels.numpy() for images, labels in test_ds
])

# Generate predictions
predictions = model.predict(test_ds, verbose=1)
y_pred = np.argmax(predictions, axis=1)

# Calculate accuracy
accuracy = accuracy_score(y_true, y_pred)

# Generate classification report
report = classification_report(
    y_true,
    y_pred,
    labels=range(len(class_names)),
    target_names=class_names,
    zero_division=0
)

# Generate confusion matrix
matrix = confusion_matrix(
    y_true,
    y_pred,
    labels=range(len(class_names))
)

# Create reports folder
os.makedirs("reports", exist_ok=True)

# Save classification report
with open("reports/classification_report.txt", "w") as file:
    file.write(f"Model Accuracy: {accuracy * 100:.2f}%\n\n")
    file.write(report)

# Plot confusion matrix
plt.figure(figsize=(13, 11))

plt.imshow(matrix, interpolation="nearest")
plt.title("FreshVision AI - Confusion Matrix")
plt.colorbar()

plt.xticks(
    range(len(class_names)),
    class_names,
    rotation=90
)

plt.yticks(
    range(len(class_names)),
    class_names
)

plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.tight_layout()

plt.savefig(
    "reports/confusion_matrix.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# Display results
print("\nMODEL EVALUATION COMPLETE")
print(f"Accuracy: {accuracy * 100:.2f}%")
print("\nClassification Report:\n")
print(report)

print("\nSaved files:")
print("reports/classification_report.txt")
print("reports/confusion_matrix.png")
