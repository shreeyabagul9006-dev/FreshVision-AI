# FreshVision AI

An AI-powered food freshness classification web application built using Flask, TensorFlow, MobileNetV2 and CLIP.

## Overview

FreshVision AI analyses uploaded food images and predicts whether the food appears fresh or rotten. It also uses CLIP to screen images that do not appear to contain food.

## Features

- Food freshness classification across 14 categories.
- MobileNetV2 image classification.
- CLIP-based food and non-food image screening.
- Grad-CAM visualisation.
- Prediction confidence display.
- SQLite prediction history.
- Analytics dashboard with charts.
- Image upload validation.

## Technology Stack

- Python and Flask
- TensorFlow and Keras
- MobileNetV2
- PyTorch
- Hugging Face Transformers and CLIP
- OpenCV and NumPy
- SQLite
- HTML, CSS and JavaScript
- Chart.js

## Model Evaluation

The classifier achieved 98.84% accuracy on the reported evaluation set of 6,738 images across 14 classes.

These results represent dataset evaluation performance and do not guarantee real-world food safety.

## Setup

1. Clone the repository and open its folder.
2. Create a virtual environment:

   `python3 -m venv .venv`

3. Activate it:

   `source .venv/bin/activate`

4. Install dependencies:

   `pip install -r requirements.txt`

5. Start the application:

   `python app.py`

6. Open `http://127.0.0.1:5000` in your browser.

The trained model files must be present in the `models/` directory. CLIP may download its model files on first use.
