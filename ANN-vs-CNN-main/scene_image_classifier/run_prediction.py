"""Predict one of the six Intel scene classes for a single image."""

import argparse
import json
from pathlib import Path

import torch

from model_architectures.cnn_classifier import CNNClassifier
from project_utils.data import get_transforms
from project_utils.prediction import predict_image
from project_utils.training import get_device_info

PROJECT_ROOT = Path(__file__).parent
MODEL_PATH = PROJECT_ROOT / "results" / "models" / "best_cnn_model.pth"
CLASS_NAMES_PATH = PROJECT_ROOT / "results" / "models" / "class_names.json"


def main():
    parser = argparse.ArgumentParser(description="Classify one scene image")
    parser.add_argument("--image", required=True, help="Path to an image file")
    args = parser.parse_args()

    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Trained CNN checkpoint not found: {MODEL_PATH}. "
            "Run the notebook first."
        )

    device, device_info = get_device_info()
    class_names = json.loads(CLASS_NAMES_PATH.read_text()) if CLASS_NAMES_PATH.is_file() else [
        "buildings", "forest", "glacier", "mountain", "sea", "street"
    ]
    model = CNNClassifier(num_classes=len(class_names), image_size=150)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)

    result = predict_image(
        args.image,
        model,
        get_transforms(image_size=150),
        class_names,
        device,
    )
    print(device_info)
    print(f"Predicted class: {result['class_name'].upper()}")
    print(f"Confidence: {result['confidence']:.2%}")
    print("Class probabilities:")
    for name, probability in result["probabilities"].items():
        print(f"  {name.title():<10} {probability:.2%}")
    print("Note: this classifier supports only the six trained scene classes.")


if __name__ == "__main__":
    main()
