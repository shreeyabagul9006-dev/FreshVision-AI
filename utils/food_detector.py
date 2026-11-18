import torch
from transformers import CLIPModel, CLIPProcessor

MODEL_NAME = "openai/clip-vit-base-patch32"

FOOD_PROMPTS = [
    "a photo of fresh fruit",
    "a photo of fresh vegetables",
    "a photo of a banana",
    "a photo of apples",
    "a photo of oranges",
    "a photo of tomatoes",
    "a photo of rotten fruit",
    "a photo of spoiled vegetables",
    "a close-up photograph of food",
    "a photograph of food ingredients",
    "a photograph of okra vegetables",
    "a photograph of lady finger vegetable",
    "a photograph of potatoes",
    "a photograph of cucumbers",
]

NONFOOD_PROMPTS = [
    "a photo of a car",
    "a photo of furniture",
    "a photo of electronic devices",
    "a photo of household objects",
    "a photo of animals",
    "a photo of clothing",
    "a photo of tools",
    "a photo of buildings",
    "a photo of stationery",
    "a photograph of non-food objects",
]

processor = CLIPProcessor.from_pretrained(MODEL_NAME)
model = CLIPModel.from_pretrained(MODEL_NAME)
model.eval()

PROMPTS = FOOD_PROMPTS + NONFOOD_PROMPTS


def detect_food(image):
    inputs = processor(
        text=PROMPTS,
        images=image,
        return_tensors="pt",
        padding=True
    )

    with torch.no_grad():
        scores = model(**inputs).logits_per_image[0]

    food_score = scores[:len(FOOD_PROMPTS)].max()
    nonfood_score = scores[len(FOOD_PROMPTS):].max()

    comparison = torch.softmax(
        torch.stack([food_score, nonfood_score]),
        dim=0
    )

    return float(comparison[0].item() * 100)
