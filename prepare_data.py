
import os
from pathlib import Path
from sklearn.model_selection import train_test_split

SOURCE_DIR = Path("data/dataset/Train")
OUTPUT_DIR = Path("data/splits")

CLASS_NAMES = [
    "freshapples",
    "freshbanana",
    "freshcucumber",
    "freshokra",
    "freshoranges",
    "freshpotato",
    "freshtomato",
    "rottenapples",
    "rottenbanana",
    "rottencucumber",
    "rottenokra",
    "rottenoranges",
    "rottenpotato",
    "rottentomato",
]

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

if OUTPUT_DIR.exists():
    raise FileExistsError(
        "data/splits already exists. It was not modified. "
        "Inspect it before deciding whether to recreate it."
    )

for split in ["Train", "Validation"]:
    for class_name in CLASS_NAMES:
        (OUTPUT_DIR / split / class_name).mkdir(
            parents=True,
            exist_ok=True
        )

total_train = 0
total_validation = 0

for class_name in CLASS_NAMES:
    class_dir = SOURCE_DIR / class_name

    if not class_dir.is_dir():
        raise FileNotFoundError(class_dir)

    images = sorted(
        path for path in class_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )

    if len(images) < 2:
        raise ValueError(f"Not enough images in {class_name}")

    train_images, validation_images = train_test_split(
        images,
        test_size=0.20,
        random_state=42
    )

    for image_path in train_images:
        target = OUTPUT_DIR / "Train" / class_name / image_path.name
        target.symlink_to(image_path.resolve())
    
    for image_path in validation_images:
        target = OUTPUT_DIR / "Validation" / class_name / image_path.name
        target.symlink_to(image_path.resolve())

    total_train += len(train_images)
    total_validation += len(validation_images)

    print(
        f"{class_name}: "
        f"{len(train_images)} training, "
        f"{len(validation_images)} validation"
    )

print("\nDATASET SPLIT COMPLETE")
print(f"Training images: {total_train}")
print(f"Validation images: {total_validation}")
print("Original Test dataset remains untouched.")
