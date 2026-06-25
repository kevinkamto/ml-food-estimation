import argparse
import os
import sys
import tempfile

import numpy as np
import torch
from loguru import logger
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset import _pixel_area, get_transforms
from model import DualStreamEfficientNet
from segmentation import segment_image


def _compute_area_ratio(before_path: str, after_path: str) -> float:
    before_area = _pixel_area(before_path)
    after_area = _pixel_area(after_path)
    ratio = float(after_area / before_area) if before_area > 0 else 0.0
    return min(ratio, 1.0)


def _run_model(
    checkpoint_path: str,
    before_t: torch.Tensor,
    after_t: torch.Tensor,
    area_ratio_t: torch.Tensor,
    device: torch.device,
) -> float:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = DualStreamEfficientNet(pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()
    with torch.no_grad():
        r_hat = float(model(before_t, after_t, area_ratio_t).item())
    return r_hat


def predict(
    before_path: str,
    after_path: str,
    checkpoint_path: str,
    weight_before_g: float | None = None,
) -> dict:
    """
    Predict consumption ratio and optionally denormalize to grams.

    Args:
        before_path: path to segmented before image
        after_path: path to segmented after image
        checkpoint_path: .pth checkpoint
        weight_before_g: known serving weight in grams; if provided, computes w_after_hat

    Returns dict with consumption_ratio, and optionally weight_after_grams.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = get_transforms("val")
    before_t = transform(Image.open(before_path).convert("RGB")).unsqueeze(0).to(device)
    after_t = transform(Image.open(after_path).convert("RGB")).unsqueeze(0).to(device)
    area_ratio = _compute_area_ratio(before_path, after_path)
    area_ratio_t = torch.tensor([[area_ratio]], dtype=torch.float32).to(device)

    r_hat = _run_model(checkpoint_path, before_t, after_t, area_ratio_t, device)

    result: dict = {
        "consumption_ratio": round(r_hat, 4),
        "area_ratio": round(area_ratio, 4),
    }
    if weight_before_g is not None:
        result["weight_after_grams"] = round(r_hat * weight_before_g, 2)
        result["leftover_grams"] = round((1 - r_hat) * weight_before_g, 2)
        result["weight_before_grams"] = weight_before_g

    return result


def ensemble_predict(
    before_path: str,
    after_path: str,
    checkpoint_paths: list[str],
    weight_before_g: float | None = None,
) -> dict:
    """Average predictions across all checkpoint paths."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = get_transforms("val")

    # Compute images and area_ratio once -- they are identical across all fold models
    before_t = transform(Image.open(before_path).convert("RGB")).unsqueeze(0).to(device)
    after_t = transform(Image.open(after_path).convert("RGB")).unsqueeze(0).to(device)
    area_ratio = _compute_area_ratio(before_path, after_path)
    area_ratio_t = torch.tensor([[area_ratio]], dtype=torch.float32).to(device)

    ratios = [_run_model(cp, before_t, after_t, area_ratio_t, device) for cp in checkpoint_paths]
    r_mean = float(np.mean(ratios))
    r_std = float(np.std(ratios))
    result: dict = {
        "consumption_ratio_mean": round(r_mean, 4),
        "consumption_ratio_std": round(r_std, 4),
        "n_models": len(ratios),
    }
    if weight_before_g is not None:
        result["weight_after_grams"] = round(r_mean * weight_before_g, 2)
        result["leftover_grams"] = round((1 - r_mean) * weight_before_g, 2)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Run inference on a before/after image pair")
    parser.add_argument("--before", required=True, help="Path to before image")
    parser.add_argument("--after", required=True, help="Path to after image")
    parser.add_argument("--checkpoint", required=True, help="Path to .pth checkpoint file")
    parser.add_argument(
        "--weight_before",
        type=float,
        default=None,
        help="Known serving weight in grams (optional; enables gram output)",
    )
    parser.add_argument(
        "--raw",
        action="store_true",
        help="Treat --before/--after as raw (unsegmented) images and auto-segment first",
    )
    parser.add_argument(
        "--output_seg",
        default=None,
        metavar="DIR",
        help="If --raw, save the auto-segmented images to this directory for inspection",
    )
    args = parser.parse_args()

    before_path = args.before
    after_path = args.after

    if args.raw:
        tmp_ctx = tempfile.TemporaryDirectory()
        tmp_dir = tmp_ctx.name if args.output_seg is None else args.output_seg
        if args.output_seg:
            os.makedirs(args.output_seg, exist_ok=True)
            tmp_ctx = None  # type: ignore[assignment]

        before_name = os.path.splitext(os.path.basename(args.before))[0] + "_seg.jpg"
        after_name = os.path.splitext(os.path.basename(args.after))[0] + "_seg.jpg"
        before_path = os.path.join(tmp_dir, before_name)
        after_path = os.path.join(tmp_dir, after_name)

        logger.info("Segmenting before image ...")
        segment_image(args.before, before_path)
        logger.info("Segmenting after image ...")
        segment_image(args.after, after_path)
        if args.output_seg:
            logger.info(f"Segmented images saved to {args.output_seg}")

    try:
        result = predict(before_path, after_path, args.checkpoint, args.weight_before)
    finally:
        if args.raw and tmp_ctx is not None:
            tmp_ctx.cleanup()

    for k, v in result.items():
        logger.info(f"{k}: {v}")


if __name__ == "__main__":
    main()
