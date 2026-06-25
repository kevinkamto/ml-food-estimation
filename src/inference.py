import argparse
import os
import sys

import torch
from loguru import logger
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset import get_transforms
from model import DualStreamEfficientNet


def predict(before_path, after_path, checkpoint_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    norm_params = checkpoint.get("normalization_params", {})
    max_weight = norm_params.get("max_weight", 1.0)
    label_classes = checkpoint.get("label_encoder_classes", [])

    model = DualStreamEfficientNet(num_classes=len(label_classes) or 34, pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    transform = get_transforms("val")
    before = transform(Image.open(before_path).convert("RGB")).unsqueeze(0).to(device)
    after = transform(Image.open(after_path).convert("RGB")).unsqueeze(0).to(device)

    with torch.no_grad():
        pred_leftover, pred_category = model(before, after)

    leftover_norm = float(pred_leftover.item())
    leftover_g = leftover_norm * max_weight
    cat_idx = int(pred_category.argmax(dim=1).item())
    cat_name = label_classes[cat_idx] if label_classes else str(cat_idx)
    confidence = float(torch.softmax(pred_category, dim=1).max().item())

    return {
        "leftover_normalized": round(leftover_norm, 4),
        "leftover_grams": round(leftover_g, 2),
        "food_category": cat_name,
        "confidence": round(confidence, 4),
    }


def main():
    parser = argparse.ArgumentParser(description="Run inference on a before/after image pair")
    parser.add_argument("--before", required=True, help="Path to segmented before image")
    parser.add_argument("--after", required=True, help="Path to segmented after image")
    parser.add_argument("--checkpoint", required=True, help="Path to .pth checkpoint file")
    args = parser.parse_args()

    result = predict(args.before, args.after, args.checkpoint)
    for k, v in result.items():
        logger.info(f"{k}: {v}")


if __name__ == "__main__":
    main()
