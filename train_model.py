
import json
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2

IMAGE_SIZE = (160, 160)
BATCH_SIZE = 32
EPOCHS = 5

CLASS_NAMES = [
    "freshapples", "freshbanana", "freshcucumber",
    "freshokra", "freshoranges", "freshpotato",
    "freshtomato", "rottenapples", "rottenbanana",
    "rottencucumber", "rottenokra", "rottenoranges",
    "rottenpotato", "rottentomato"
]

train_ds = tf.keras.utils.image_dataset_from_directory(
    "data/processed/Train",
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_names=CLASS_NAMES,
    label_mode="int",
    seed=42
)

test_ds = tf.keras.utils.image_dataset_from_directory(
    "data/processed/Test",
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_names=CLASS_NAMES,
    label_mode="int",
    shuffle=False
)

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
test_ds = test_ds.prefetch(buffer_size=AUTOTUNE)

base_model = MobileNetV2(
    input_shape=(160, 160, 3),
    include_top=False,
    weights="imagenet"
)

base_model.trainable = False

model = models.Sequential([
    layers.Input(shape=(160, 160, 3)),
    layers.Rescaling(1.0 / 127.5, offset=-1),
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.3),
    layers.Dense(len(CLASS_NAMES), activation="softmax")
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=2,
    restore_best_weights=True
)

history = model.fit(
    train_ds,
    validation_data=test_ds,
    epochs=EPOCHS,
    callbacks=[early_stopping]
)

loss, accuracy = model.evaluate(test_ds)

model.save("models/freshvision_model.keras")

with open("models/class_names.json", "w") as file:
    json.dump(CLASS_NAMES, file)

print(f"\nTest Accuracy: {accuracy * 100:.2f}%")
print("Model saved successfully!")
