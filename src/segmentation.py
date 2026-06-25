"""
Automatic segmentation of raw food plate images into the ground-truth format:
  black background | pure white plate area | food at original colors

Two-stage algorithm:
  Stage 1 -- GrabCut + convex hull: detect the plate boundary
  Stage 2 -- HSV threshold within hull: plate surface -> white, food -> original colors
"""

import argparse
import os

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

_OUT_SIZE = 800
_GRABCUT_RECT_INSET = 0.10
_CLOSE_KERNEL = 30
_PLATE_SAT_MAX = 40
_PLATE_VAL_MIN = 150


def segment_image(input_path: str, output_path: str | None = None) -> Image.Image:
    """
    Segment a raw food plate image into the standard 3-region format.

    Returns an 800x800 RGB PIL Image:
      - background (outside plate hull) -> black (0, 0, 0)
      - plate surface (inside hull, low saturation / high brightness) -> white (255, 255, 255)
      - food (inside hull, colored or textured) -> original pixel colors

    Args:
        input_path: Path to the raw input image (JPG/PNG).
        output_path: If provided, saves the result to this path.
    """
    bgr = cv2.imread(input_path)
    if bgr is None:
        raise FileNotFoundError(f"Cannot read image: {input_path}")

    bgr_sq = _pad_to_square(bgr)
    bgr_800 = cv2.resize(bgr_sq, (_OUT_SIZE, _OUT_SIZE), interpolation=cv2.INTER_LINEAR)

    plate_mask = _detect_plate_hull(bgr_800)
    result_bgr = _normalize_plate(bgr_800, plate_mask)

    rgb = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB)
    out = Image.fromarray(rgb)
    if output_path is not None:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        out.save(output_path)
    return out


def _pad_to_square(bgr: np.ndarray) -> np.ndarray:
    """Pad the shorter dimension with black so the image becomes square."""
    h, w = bgr.shape[:2]
    if h == w:
        return bgr
    side = max(h, w)
    canvas = np.zeros((side, side, 3), dtype=bgr.dtype)
    top = (side - h) // 2
    left = (side - w) // 2
    canvas[top : top + h, left : left + w] = bgr
    return canvas


def _detect_plate_hull(bgr: np.ndarray) -> np.ndarray:
    """
    Stage 1: GrabCut with center-biased rect, then convex hull of the largest contour.
    Returns a binary uint8 mask (255 inside hull, 0 outside), same size as bgr.
    """
    h, w = bgr.shape[:2]
    inset = int(min(h, w) * _GRABCUT_RECT_INSET)
    rect = (inset, inset, w - 2 * inset, h - 2 * inset)

    gc_mask = np.zeros((h, w), dtype=np.uint8)
    bgd_model = np.zeros((1, 65), dtype=np.float64)
    fgd_model = np.zeros((1, 65), dtype=np.float64)
    cv2.grabCut(bgr, gc_mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)

    fg_mask = np.where(
        (gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD),
        np.uint8(255),
        np.uint8(0),
    )

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (_CLOSE_KERNEL, _CLOSE_KERNEL)
    )
    closed = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return closed

    largest = max(contours, key=cv2.contourArea)
    hull = cv2.convexHull(largest)

    hull_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(hull_mask, [hull], -1, 255, thickness=cv2.FILLED)
    return hull_mask


def _normalize_plate(bgr: np.ndarray, plate_mask: np.ndarray) -> np.ndarray:
    """
    Stage 2: within the plate hull, classify pixels by HSV and normalize plate surface to white.

    Plate surface: S < _PLATE_SAT_MAX and V > _PLATE_VAL_MIN -> (255, 255, 255)
    Food:          everything else inside the hull -> keep original BGR
    Outside hull:  (0, 0, 0)
    """
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]

    inside = plate_mask == 255
    plate_surface = inside & (s < _PLATE_SAT_MAX) & (v > _PLATE_VAL_MIN)

    result = np.zeros_like(bgr)
    result[inside] = bgr[inside]
    result[plate_surface] = (255, 255, 255)
    return result


def _batch_segment(input_dir: str, output_dir: str) -> None:
    """Walk input_dir recursively and segment every JPG/PNG into output_dir (flat)."""
    exts = {".jpg", ".jpeg", ".png"}
    paths = []
    for root, _, files in os.walk(input_dir):
        for f in files:
            if os.path.splitext(f)[1].lower() in exts:
                paths.append(os.path.join(root, f))

    os.makedirs(output_dir, exist_ok=True)
    for src in tqdm(paths, desc="Segmenting"):
        dst = os.path.join(output_dir, os.path.basename(src))
        try:
            segment_image(src, dst)
        except Exception as exc:
            print(f"WARN: skipping {src} -- {exc}")


def _cli() -> None:
    parser = argparse.ArgumentParser(
        description="Segment raw food plate images to the standard 3-region format."
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--input", help="Single input image path")
    mode.add_argument("--input_dir", help="Batch mode: directory of raw images (walked recursively)")
    parser.add_argument("--output", help="Single output image path (required with --input)")
    parser.add_argument(
        "--output_dir",
        help="Batch mode: directory to write segmented images (flat)",
    )
    args = parser.parse_args()

    if args.input:
        if not args.output:
            parser.error("--output is required when using --input")
        img = segment_image(args.input, args.output)
        print(f"Saved {args.output}  ({img.size[0]}x{img.size[1]})")
    else:
        if not args.output_dir:
            parser.error("--output_dir is required when using --input_dir")
        _batch_segment(args.input_dir, args.output_dir)


if __name__ == "__main__":
    _cli()
