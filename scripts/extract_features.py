
import json
import os

import numpy as np
import tensorflow as tf


IMAGE_SIZE = (160, 160)
BATCH_SIZE = 32

TRAIN_DIR = "data/splits/Train"
VALIDATION_DIR = "data/splits/Validation"

MODEL_PATH = "models/freshvision_model.keras"
CLASS_NAMES_PATH = "models/class_names.json"
OUTPUT_PATH = "models/ood_reference_features.npz"


with open(CLASS_NAMES_PATH, "r") as file:
    class_names = json.load(file)


model = tf.keras.models.load_model(MODEL_PATH)

# Build the feature extractor using the model's first three layers.
# This preserves the original rescaling and MobileNetV2 preprocessing.

inputs = tf.keras.Input(shape=(160, 160, 3))

x = inputs

for layer in model.layers[:3]:
    x = layer(x, training=False)

feature_extractor = tf.keras.Model(
    inputs=inputs,
    outputs=x
)


def extract_dataset_features(directory):

    dataset = tf.keras.utils.image_dataset_from_directory(
        directory,
        labels="inferred",
        label_mode="int",
        class_names=class_names,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    features = []
    labels = []

    for images, batch_labels in dataset:

        batch_features = feature_extractor.predict(
            images,
            verbose=0
        )

        features.append(batch_features)
        labels.append(batch_labels.numpy())

    return (
        np.concatenate(features),
        np.concatenate(labels)
    )


print("Extracting training features...")

train_features, train_labels = extract_dataset_features(
    TRAIN_DIR
)

print("Extracting validation features...")

validation_features, validation_labels = extract_dataset_features(
    VALIDATION_DIR
)


os.makedirs("models", exist_ok=True)

np.savez_compressed(
    OUTPUT_PATH,
    train_features=train_features,
    train_labels=train_labels,
    validation_features=validation_features,
    validation_labels=validation_labels,
    class_names=np.array(class_names)
)


print("\nFeature extraction completed.")
print("Training features:", train_features.shape)
print("Validation features:", validation_features.shape)
print("Saved to:", OUTPUT_PATH)
