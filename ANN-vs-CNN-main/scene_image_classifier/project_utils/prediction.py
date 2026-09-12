"""Single-image inference for the trained scene classifier."""

from pathlib import Path

import torch
from PIL import Image


def predict_image(image_path, model, transform, class_names, device):
    """Return the predicted scene, confidence, and all class probabilities."""
    image_path = Path(image_path)
    if not image_path.is_file():
        raise FileNotFoundError(f"Image file not found: {image_path}")

    image = Image.open(image_path).convert("RGB")
    inputs = transform(image).unsqueeze(0).to(device)
    model.eval()
    with torch.inference_mode():
        probabilities = torch.softmax(model(inputs), dim=1)[0].cpu()

    predicted_index = int(probabilities.argmax())
    return {
        "class_name": class_names[predicted_index],
        "confidence": float(probabilities[predicted_index]),
        "probabilities": {
            name: float(probabilities[index])
            for index, name in enumerate(class_names)
        },
    }
