
import os
import shutil

SOURCE_TRAIN = "data/dataset/Train"
SOURCE_TEST = "data/dataset/Test"

TARGET_TRAIN = "data/processed/Train"
TARGET_TEST = "data/processed/Test"

CLASSES = [
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

TEST_FOLDER_MAP = {
    "freshpotato": "freshpotato",
    "freshtomato": "freshtomato",
    "rottenpotato": "rottenpotato",
    "rottentomato": "rottentomato",
}

for folder in [TARGET_TRAIN, TARGET_TEST]:
    os.makedirs(folder, exist_ok=True)

for class_name in CLASSES:
    train_source = os.path.join(SOURCE_TRAIN, class_name)
    train_target = os.path.join(TARGET_TRAIN, class_name)

    if not os.path.isdir(train_source):
        raise FileNotFoundError(train_source)

    if not os.path.exists(train_target):
        os.symlink(os.path.abspath(train_source), train_target)

    test_source = os.path.join(SOURCE_TEST, TEST_FOLDER_MAP.get(class_name, class_name))
    test_target = os.path.join(TARGET_TEST, class_name)

    if not os.path.isdir(test_source):
        raise FileNotFoundError(test_source)

    if not os.path.exists(test_target):
        os.symlink(os.path.abspath(test_source), test_target)

print("Dataset preparation completed!")
print("Training classes:", len(os.listdir(TARGET_TRAIN)))
print("Testing classes:", len(os.listdir(TARGET_TEST)))
