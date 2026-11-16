
import os
import json
import uuid
import numpy as np
import tensorflow as tf

from flask import Flask, render_template, request, send_from_directory
from werkzeug.utils import secure_filename
from PIL import Image

from database.db import initialize_database, get_connection
from utils.gradcam import generate_gradcam


app = Flask(__name__)

app.config["UPLOAD_FOLDER"] = "uploads"
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024

ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "webp"}

initialize_database()
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

model = tf.keras.models.load_model(
    "models/freshvision_model.keras"
)

with open("models/class_names.json", "r") as file:
    class_names = json.load(file)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    if "food_image" not in request.files:
        return render_template("index.html", error="Please upload an image.")

    file = request.files["food_image"]

    if file.filename == "":
        return render_template("index.html", error="Please select an image.")

    filename = secure_filename(file.filename)

    if "." not in filename:
        return render_template("index.html", error="Invalid image filename.")

    extension = filename.rsplit(".", 1)[1].lower()

    if extension not in ALLOWED_EXTENSIONS:
        return render_template("index.html", error="Unsupported image format.")

    unique_filename = f"{uuid.uuid4().hex}_{filename}"
    overlay_filename = f"{uuid.uuid4().hex}_gradcam.jpg"

    filepath = os.path.join(
        app.config["UPLOAD_FOLDER"],
        unique_filename
    )

    overlay_path = os.path.join(
        app.config["UPLOAD_FOLDER"],
        overlay_filename
    )

    try:
        file.save(filepath)

        with Image.open(filepath) as uploaded_image:
            original_image = uploaded_image.convert("RGB")

        model_image = original_image.resize((160, 160))

        image_array = np.array(
            model_image,
            dtype=np.float32
        )

        image_array = np.expand_dims(image_array, axis=0)

        predictions = model.predict(image_array, verbose=0)[0]

        predicted_index = int(np.argmax(predictions))
        predicted_class = class_names[predicted_index]

        freshness = (
            "Fresh" if predicted_class.startswith("fresh") else "Rotten"
        )

        food_name = predicted_class.replace(
            "fresh", ""
        ).replace("rotten", "").capitalize()

        confidence = float(predictions[predicted_index]) * 100

        # Generate Grad-CAM heatmap
        heatmap = generate_gradcam(
            model,
            image_array,
            predicted_index
        )

        # Resize heatmap to original image dimensions
        heatmap_image = Image.fromarray(heatmap).resize(
            original_image.size
        )

        # Create original image and heatmap overlay
        overlay = Image.blend(
            original_image,
            heatmap_image,
            alpha=0.4
        )

        overlay.save(overlay_path, format="JPEG", quality=95)

        connection = get_connection()

        connection.execute(
            """
            INSERT INTO predictions
            (food_name, freshness, confidence, image_filename)
            VALUES (?, ?, ?, ?)
            """,
            (food_name, freshness, confidence, unique_filename)
        )

        connection.commit()
        connection.close()

        return render_template(
            "result.html",
            food_name=food_name,
            freshness=freshness,
            confidence=round(confidence, 2),
            image_filename=unique_filename,
            overlay_filename=overlay_filename
        )

    except Exception:
        app.logger.exception("Prediction failed")

        for path in [filepath, overlay_path]:
            if os.path.exists(path):
                os.remove(path)

        return render_template(
            "index.html",
            error="Unable to analyse this image. Please try another."
        )


@app.route("/dashboard")
def dashboard():

    connection = get_connection()

    total = connection.execute(
        "SELECT COUNT(*) FROM predictions"
    ).fetchone()[0]

    fresh = connection.execute(
        "SELECT COUNT(*) FROM predictions WHERE freshness = 'Fresh'"
    ).fetchone()[0]

    rotten = connection.execute(
        "SELECT COUNT(*) FROM predictions WHERE freshness = 'Rotten'"
    ).fetchone()[0]

    average_confidence = connection.execute(
        "SELECT AVG(confidence) FROM predictions"
    ).fetchone()[0]

    history = connection.execute(
        """
        SELECT * FROM predictions
        ORDER BY created_at DESC
        LIMIT 10
        """
    ).fetchall()

    connection.close()

    return render_template(
        "dashboard.html",
        total=total,
        fresh=fresh,
        rotten=rotten,
        average_confidence=round(average_confidence or 0, 2),
        history=history
    )


@app.route("/uploads/<filename>")
def uploaded_file(filename):
    return send_from_directory(
        app.config["UPLOAD_FOLDER"],
        filename
    )
@app.route("/reports/<filename>")
def report_file(filename):
    return send_from_directory(
        "reports",
        filename
    )

if __name__ == "__main__":
    app.run(debug=True)
