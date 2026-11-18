
import os
import json
import numpy as np
import tensorflow as tf
import cv2

from pathlib import Path
from PIL import Image
from utils.gradcam import generate_gradcam


# Load trained model
model = tf.keras.models.load_model(
    "models/freshvision_model.keras"
)

# Load class names
with open("models/class_names.json", "r") as file:
    class_names = json.load(file)

# Find a test image
test_folder = Path("data/dataset/Test")

image_extensions = ["*.jpg", "*.jpeg", "*.png", "*.webp"]

image_files = []

for extension in image_extensions:
    image_files.extend(test_folder.rglob(extension))

if not image_files:
    raise FileNotFoundError("No test images found.")

image_path = image_files[0]

# Preprocess image
image = Image.open(image_path).convert("RGB")
image = image.resize((160, 160))

image_array = np.array(
    image,
    dtype=np.float32
)

image_array = np.expand_dims(image_array, axis=0)

# Predict food class
predictions = model.predict(image_array, verbose=0)[0]

predicted_index = int(np.argmax(predictions))

print("Image:", image_path)
print("Predicted class:", class_names[predicted_index])
print("Confidence:", round(float(predictions[predicted_index]) * 100, 2), "%")

# Generate Grad-CAM heatmap
heatmap = generate_gradcam(
    model,
    image_array,
    predicted_index
)

# Save heatmap
os.makedirs("reports", exist_ok=True)

output_path = "reports/gradcam_heatmap.jpg"

from PIL import Image as PILImage

PILImage.fromarray(heatmap).save(output_path)

# Create Grad-CAM overlay
original_image = np.array(image)

overlay = cv2.addWeighted(
    original_image,
    0.6,
    heatmap,
    0.4,
    0
)

overlay_path = "reports/gradcam_overlay.jpg"

PILImage.fromarray(overlay).save(overlay_path)

print("Grad-CAM heatmap saved:", output_path)
print("Grad-CAM overlay saved:", overlay_path)