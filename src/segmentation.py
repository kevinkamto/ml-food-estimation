"""
Automatic segmentation of raw food plate images into the ground-truth format:
  black background | pure white plate area | food at original colors

Two-stage algorithm:
  Stage 1 -- SAM ViT-B: center-point + corner-background prompts identify the plate region
  Stage 2 -- HSV + texture within mask: plate surface -> white, food -> original colors

Debug mode writes numbered intermediate images and masks to `debug_dir`:
  1.x: padded and resized input
  3.x: SAM masks, prompt overlay, closed mask, shrunk mask
  4.x: HSV/texture diagnostics and plate surface cleanup
  5.x: final result
"""

import argparse
import os
import urllib.request
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from segment_anything import SamPredictor, sam_model_registry
from tqdm import tqdm

from utils import save_ndarray_image

_OUT_SIZE = 800
_MASK_CLOSE_KERNEL = 21  # morphological closing after SAM mask to fill small gaps
_MASK_ERODE_KERNEL = 21  # morphological erosion to trim plate border from SAM mask
_NOISE_REMOVAL_KERNEL = 21  # morphological opening to remove small plate-surface dots inside food region
_PLATE_SAT_MAX = 15
_PLATE_VAL_MIN = 150
_PLATE_TEX_MAX = 12

_SAM_MODEL_TYPE = "vit_b"
_SAM_CHECKPOINT_URL = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
_SAM_CHECKPOINT_PATH = Path.home() / ".cache" / "segment_anything" / "sam_vit_b_01ec64.pth"

_sam_predictor: SamPredictor | None = None


def _save_debug_image(debug_dir: str, filename: str, array: np.ndarray, *, as_bgr: bool = False) -> None:
    save_ndarray_image(array, os.path.join(debug_dir, filename), as_bgr=as_bgr)


def _debug_filename(image_id: str, main: int, sub: int, meaning: str, ext: str) -> str:
    safe_desc = meaning.replace(" ", "_")
    return f"{image_id}_{main:02d}.{sub:02d}_{safe_desc}{ext}"


def _save_debug_step(
    debug_dir: str,
    image_id: str,
    main: int,
    sub: int,
    meaning: str,
    array: np.ndarray,
    *,
    as_bgr: bool = False,
    ext: str | None = None,
) -> None:
    if ext is None:
        ext = ".png"
    filename = _debug_filename(image_id, main, sub, meaning, ext)
    _save_debug_image(debug_dir, filename, array, as_bgr=as_bgr)


def _overlay_mask(image: np.ndarray, mask: np.ndarray, alpha: float = 0.35) -> np.ndarray:
    overlay = image.copy()
    mask_rgb = np.zeros_like(image)
    mask_rgb[mask == 255] = (0, 0, 255)
    cv2.addWeighted(overlay, 1.0 - alpha, mask_rgb, alpha, 0, overlay)
    return overlay


def _remove_plate_noise(plate_surface: np.ndarray, kernel_size: int = _NOISE_REMOVAL_KERNEL) -> np.ndarray:
    """Remove tiny plate-surface dots inside the food region using morphology."""
    mask = (plate_surface.astype(np.uint8) * 255) if plate_surface.dtype == bool else plate_surface.astype(np.uint8)
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return (cleaned > 0)


def _erode_plate_mask(mask: np.ndarray, kernel_size: int = _MASK_ERODE_KERNEL) -> np.ndarray:
    """Erode the plate mask to trim the outer border."""
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    return cv2.erode(mask, kernel, iterations=1)


def _close_mask_border(mask: np.ndarray, kernel_size: int = _MASK_CLOSE_KERNEL) -> np.ndarray:
    """Close the plate mask using black padding to avoid edge morphing artifacts."""
    if kernel_size % 2 == 0:
        kernel_size += 1
    pad = kernel_size // 2
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    padded = cv2.copyMakeBorder(mask, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)
    closed = cv2.morphologyEx(padded, cv2.MORPH_CLOSE, kernel)
    return closed[pad:-pad, pad:-pad] if pad > 0 else closed


def _get_sam_predictor() -> SamPredictor:
    """Lazy-load SAM ViT-B predictor, downloading the checkpoint on first use (~375 MB)."""
    global _sam_predictor
    if _sam_predictor is None:
        if not _SAM_CHECKPOINT_PATH.exists():
            _SAM_CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
            print(f"Downloading SAM checkpoint (~375 MB) to {_SAM_CHECKPOINT_PATH} ...")
            urllib.request.urlretrieve(_SAM_CHECKPOINT_URL, str(_SAM_CHECKPOINT_PATH))
        device = "cuda" if torch.cuda.is_available() else "cpu"
        sam = sam_model_registry[_SAM_MODEL_TYPE](checkpoint=str(_SAM_CHECKPOINT_PATH))
        sam.to(device)
        sam.eval()
        _sam_predictor = SamPredictor(sam)
    return _sam_predictor


def segment_image(
    input_path: str,
    output_path: str | None = None,
    debug_dir: str | None = None,
) -> Image.Image:
    """
    Segment a raw food plate image into the standard 3-region format.

    Returns an 800x800 RGB PIL Image:
      - background (outside plate) -> black (0, 0, 0)
      - plate surface (inside mask, low saturation / high brightness) -> white (255, 255, 255)
      - food (inside mask, colored or textured) -> original pixel colors

    If `debug_dir` is provided, this function saves intermediate artifacts with a
    numbered naming scheme. The saved images document key steps:
      - padded input, resized input
      - SAM candidate masks and prompt overlay
      - closed plate mask and shrunk plate mask
      - HSV, texture, and plate surface classification maps
      - final segmented output

    Args:
        input_path: Path to the raw input image (JPG/PNG).
        output_path: If provided, saves the result to this path.
        debug_dir: If provided, writes intermediate debug images and masks here.
    """
    image_bgr = cv2.imread(input_path)
    if image_bgr is None:
        raise FileNotFoundError(f"Cannot read image: {input_path}")

    if debug_dir is not None:
        os.makedirs(debug_dir, exist_ok=True)

    image_id = Path(input_path).stem
    h0, w0 = image_bgr.shape[:2]
    padded_bgr, pad_top, pad_left = _pad_to_square(image_bgr)
    if debug_dir is not None:
        _save_debug_step(debug_dir, image_id, 1, 0, "padded", padded_bgr, as_bgr=True)

    side = max(h0, w0)
    resized_bgr = cv2.resize(padded_bgr, (_OUT_SIZE, _OUT_SIZE), interpolation=cv2.INTER_LINEAR)
    if debug_dir is not None:
        _save_debug_step(debug_dir, image_id, 2, 0, "resized", resized_bgr, as_bgr=True)

    scale = _OUT_SIZE / float(side)
    prompt_coords = _build_sam_prompt_points(w0, h0, pad_left, pad_top, scale)
    plate_mask = _detect_plate_mask(
        resized_bgr,
        prompt_coords=prompt_coords,
        debug_dir=debug_dir,
        image_id=image_id,
    )
    if debug_dir is not None:
        _save_debug_step(debug_dir, image_id, 3, 5, "plate_mask", plate_mask)
        _save_debug_step(
            debug_dir,
            image_id,
            3,
            6,
            "plate_mask_overlay",
            _overlay_mask(resized_bgr, plate_mask),
            as_bgr=True,
            ext=".jpg",
        )

    food_hint = _build_food_hint_mask(resized_bgr, plate_mask)
    food_mask = _refine_food_mask_with_sam(
        resized_bgr,
        plate_mask,
        food_hint,
        debug_dir=debug_dir,
        image_id=image_id,
    )

    result_bgr = _normalize_plate(
        resized_bgr,
        plate_mask,
        food_mask,
        debug_dir=debug_dir,
        image_id=image_id,
    )

    if debug_dir is not None:
        _save_debug_step(debug_dir, image_id, 5, 0, "result", result_bgr, as_bgr=True, ext=".jpg")

    rgb = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB)
    out = Image.fromarray(rgb)
    if output_path is not None:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        out.save(output_path)
    return out


def _pad_to_square(bgr: np.ndarray) -> tuple[np.ndarray, int, int]:
    """Pad the shorter dimension with black so the image becomes square.

    Returns the padded image and the top/left offsets of the original image inside it.
    """
    h, w = bgr.shape[:2]
    if h == w:
        return bgr, 0, 0
    side = max(h, w)
    canvas = np.zeros((side, side, 3), dtype=bgr.dtype)
    top = (side - h) // 2
    left = (side - w) // 2
    canvas[top : top + h, left : left + w] = bgr
    return canvas, top, left


def _build_sam_prompt_points(w0: int, h0: int, pad_left: int, pad_top: int, scale: float) -> np.ndarray:
    """Build SAM prompt coordinates in padded + resized image space."""
    def map_point(x: float, y: float) -> tuple[float, float]:
        px = (pad_left + x) * scale
        py = (pad_top + y) * scale
        return px, py

    foreground = [
        (w0 / 2.0, h0 / 2.0),
        (w0 / 4.0, h0 / 4.0),
        (3.0 * w0 / 4.0, h0 / 4.0),
        (3.0 * w0 / 4.0, 3.0 * h0 / 4.0),
        (w0 / 4.0, 3.0 * h0 / 4.0),
    ]
    background = [
        (0.0, 0.0),
        (float(w0 - 1), 0.0),
        (0.0, float(h0 - 1)),
        (float(w0 - 1), float(h0 - 1)),
    ]
    points = [map_point(x, y) for (x, y) in foreground + background]
    return np.array(points, dtype=np.float32)


def _draw_prompt_points(
    image: np.ndarray,
    prompt_coords: np.ndarray,
    point_labels: np.ndarray | None = None,
    point_categories: list[str] | None = None,
    radius: int = 10,
) -> np.ndarray:
    overlay = image.copy()
    for idx, (x, y) in enumerate(prompt_coords.tolist()):
        point = (int(round(x)), int(round(y)))
        if point_categories is not None:
            category = point_categories[idx]
            if category == "food":
                color = (0, 0, 255)
            elif category == "plate":
                color = (0, 255, 0)
            else:
                color = (255, 0, 0)
        elif point_labels is not None:
            color = (0, 0, 255) if point_labels[idx] == 1 else (255, 0, 0)
        else:
            color = (0, 0, 255)

        cv2.circle(overlay, point, radius, color, thickness=-1)
        cv2.putText(
            overlay,
            str(idx),
            (point[0] + 5, point[1] - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            thickness=1,
            lineType=cv2.LINE_AA,
        )
    return overlay


def _build_food_hint_mask(
    bgr: np.ndarray,
    plate_mask: np.ndarray,
    debug_dir: str | None = None,
    image_id: str | None = None,
) -> np.ndarray:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    mean = cv2.blur(gray, (7, 7))
    mean_sq = cv2.blur(gray**2, (7, 7))
    texture_std = np.sqrt(np.clip(mean_sq - mean**2, 0, None))

    inside = plate_mask == 255
    plate_surface = (
        inside & (s < _PLATE_SAT_MAX) & (v > _PLATE_VAL_MIN) & (texture_std < _PLATE_TEX_MAX)
    )
    food_hint = inside & ~plate_surface

    if debug_dir is not None:
        image_id = image_id or "image"
        _save_debug_step(debug_dir, image_id, 4, 1, "food_hint", food_hint.astype(np.uint8) * 255)
        _save_debug_step(debug_dir, image_id, 4, 2, "plate_surface_hint", plate_surface.astype(np.uint8) * 255)

    return (food_hint.astype(np.uint8) * 255)


def _get_plate_edge_points(mask: np.ndarray) -> np.ndarray:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return np.zeros((0, 2), dtype=np.float32)

    contour = max(contours, key=cv2.contourArea).reshape(-1, 2)
    top = contour[np.argmin(contour[:, 1])]
    bottom = contour[np.argmax(contour[:, 1])]
    left = contour[np.argmin(contour[:, 0])]
    right = contour[np.argmax(contour[:, 0])]
    points = np.vstack([top, right, bottom, left]).astype(np.float32)
    return points


def _build_food_prompt_points(
    bgr: np.ndarray,
    plate_mask: np.ndarray,
    food_hint: np.ndarray,
    max_food_points: int = 8,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    h, w = bgr.shape[:2]
    corners = np.array(
        [[0.0, 0.0], [float(w - 1), 0.0], [0.0, float(h - 1)], [float(w - 1), float(h - 1)]],
        dtype=np.float32,
    )
    plate_points = _get_plate_edge_points(plate_mask)
    if plate_points.shape[0] < 4:
        plate_points = np.array([corners[0], corners[1], corners[2], corners[3]], dtype=np.float32)

    food_locations = np.argwhere(food_hint == 255)
    if food_locations.size == 0:
        return np.zeros((0, 2), dtype=np.float32), np.zeros((0,), dtype=np.int32), []

    if len(food_locations) <= max_food_points:
        selected = food_locations
    else:
        step = len(food_locations) / float(max_food_points)
        selected = food_locations[(np.arange(max_food_points) * step).astype(int)]

    food_points = selected[:, ::-1].astype(np.float32)

    prompt_coords = np.vstack([food_points, plate_points, corners])
    point_labels = np.concatenate(
        [np.ones(len(food_points), dtype=np.int32), np.zeros(len(plate_points) + len(corners), dtype=np.int32)]
    )
    categories = ["food"] * len(food_points) + ["plate"] * len(plate_points) + ["bg"] * len(corners)
    return prompt_coords, point_labels, categories


def _choose_best_food_mask(masks: np.ndarray, food_hint: np.ndarray) -> np.ndarray:
    if masks.shape[0] == 0:
        return np.zeros_like(food_hint, dtype=np.uint8)

    best_iou = -1.0
    best_mask = np.zeros_like(food_hint, dtype=np.uint8)
    hint_bool = food_hint.astype(bool)
    for mask in masks:
        mask_bool = mask.astype(bool)
        intersection = np.logical_and(mask_bool, hint_bool).sum()
        union = np.logical_or(mask_bool, hint_bool).sum()
        iou = float(intersection) / float(union) if union > 0 else 0.0
        if iou > best_iou:
            best_iou = iou
            best_mask = mask.astype(np.uint8) * 255
    return best_mask


def _refine_food_mask_with_sam(
    bgr: np.ndarray,
    plate_mask: np.ndarray,
    food_hint: np.ndarray,
    debug_dir: str | None = None,
    image_id: str | None = None,
) -> np.ndarray:
    if np.count_nonzero(food_hint) == 0:
        return np.zeros_like(food_hint, dtype=np.uint8)

    prompt_coords, point_labels, categories = _build_food_prompt_points(bgr, plate_mask, food_hint)
    if prompt_coords.shape[0] == 0:
        return np.zeros_like(food_hint, dtype=np.uint8)

    if debug_dir is not None:
        image_id = image_id or "image"
        _save_debug_step(
            debug_dir,
            image_id,
            3,
            51,
            "food_prompt_overlay",
            _draw_prompt_points(bgr, prompt_coords, point_labels=point_labels, point_categories=categories),
            as_bgr=True,
            ext=".jpg",
        )

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    predictor = _get_sam_predictor()
    predictor.set_image(rgb)

    mask_input = cv2.resize(plate_mask, (256, 256), interpolation=cv2.INTER_NEAREST)
    mask_input = (mask_input > 128).astype(np.float32)[None, :, :]

    masks, scores, _ = predictor.predict(
        point_coords=prompt_coords,
        point_labels=point_labels,
        mask_input=mask_input,
        multimask_output=True,
    )

    if debug_dir is not None:
        image_id = image_id or "image"
        for i, mask in enumerate(masks):
            mask_img = (mask.astype(np.uint8) * 255)
            _save_debug_step(debug_dir, image_id, 3, 60 + i * 3 + 1, f"sam_food_mask_{i:02d}", mask_img)
            _save_debug_step(
                debug_dir,
                image_id,
                3,
                60 + i * 3 + 2,
                f"sam_food_mask_{i:02d}_overlay",
                _overlay_mask(bgr, mask_img),
                as_bgr=True,
                ext=".jpg",
            )

    best_mask = _choose_best_food_mask(masks, food_hint)
    if debug_dir is not None:
        _save_debug_step(debug_dir, image_id, 3, 99, "sam_food_mask_best", best_mask)
        _save_debug_step(
            debug_dir,
            image_id,
            3,
            100,
            "sam_food_mask_best_overlay",
            _overlay_mask(bgr, best_mask),
            as_bgr=True,
            ext=".jpg",
        )

    return best_mask


def _detect_plate_mask(
    bgr: np.ndarray,
    prompt_coords: np.ndarray | None = None,
    debug_dir: str | None = None,
    image_id: str | None = None,
) -> np.ndarray:
    """
    Stage 1: SAM ViT-B with center-point + corner-background prompts.

    Foreground prompt: center of frame (plate is always centered, fixed camera).
    Background prompts: four corners (actual image corners, not padded border).
    Returns a binary uint8 mask (255 = plate+food, 0 = background), same HxW as bgr.

    When debug mode is enabled, this function saves:
      - each SAM candidate mask and overlay
      - a prompt overlay image
      - the closed mask after morphology
      - the shrunk mask after erosion
    """
    h, w = bgr.shape[:2]
    if prompt_coords is None:
        margin = min(h, w) // 10
        prompt_coords = np.array(
            [
                [w // 2, h // 2],  # center = foreground (plate)
                [margin, margin],  # top-left = background
                [w - margin, margin],  # top-right = background
                [margin, h - margin],  # bottom-left = background
                [w - margin, h - margin],  # bottom-right = background
            ]
        )
        point_labels = np.array([1, 0, 0, 0, 0], dtype=np.int32)
    else:
        # first values are foreground points, last values are background points
        point_labels = np.array([1] * 5 + [0] * 4, dtype=np.int32)

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    predictor = _get_sam_predictor()
    predictor.set_image(rgb)

    masks, scores, _ = predictor.predict(
        point_coords=prompt_coords,
        point_labels=point_labels,
        multimask_output=True,
    )

    if debug_dir is not None:
        image_id = image_id or "image"
        for i, (mask, score) in enumerate(zip(masks, scores)):
            mask_img = (mask.astype(np.uint8) * 255)
            _save_debug_step(debug_dir, image_id, 3, i * 3 + 1, f"sam_mask_{i:02d}", mask_img)
            _save_debug_step(
                debug_dir,
                image_id,
                3,
                i * 3 + 2,
                f"sam_mask_{i:02d}_overlay",
                _overlay_mask(bgr, mask_img),
                as_bgr=True,
                ext=".jpg",
            )
            score_path = os.path.join(debug_dir, _debug_filename(image_id, 3, i * 3 + 3, f"sam_score_{i:02d}", ".txt"))
            with open(score_path, "w", encoding="utf-8") as score_file:
                score_file.write(f"{score:.6f}\n")
        if prompt_coords is not None:
            _save_debug_step(
                debug_dir,
                image_id,
                3,
                50,
                "prompt_overlay",
                _draw_prompt_points(bgr, prompt_coords),
                as_bgr=True,
                ext=".jpg",
            )

    best = masks[int(np.argmax(scores))].astype(np.uint8) * 255
    closed = _close_mask_border(best)
    shrunk = _erode_plate_mask(closed)
    if debug_dir is not None and image_id is not None:
        _save_debug_step(debug_dir, image_id, 3, 55, "plate_mask_closed", closed)
        _save_debug_step(debug_dir, image_id, 3, 56, "plate_mask_shrunk", shrunk)
    return np.asarray(shrunk, dtype=np.uint8)


def _normalize_plate(
    bgr: np.ndarray,
    plate_mask: np.ndarray,
    food_mask: np.ndarray,
    debug_dir: str | None = None,
    image_id: str | None = None,
) -> np.ndarray:
    """
    Convert the plate area into white plate surface and original food using a refined food mask.

    Food is kept as original BGR inside the plate mask when the SAM-refined food mask is present.
    Plate surface pixels inside the plate mask but outside the food mask become white.
    Outside mask becomes black.
    """
    inside = plate_mask == 255
    food = (food_mask == 255) & inside
    plate_surface = inside & ~food

    if debug_dir is not None:
        image_id = image_id or "image"
        _save_debug_step(debug_dir, image_id, 4, 1, "food_mask", food.astype(np.uint8) * 255)
        _save_debug_step(debug_dir, image_id, 4, 2, "food_mask_overlay", _overlay_mask(bgr, food.astype(np.uint8) * 255), as_bgr=True, ext=".jpg")

    result = np.zeros_like(bgr)
    result[inside] = bgr[inside]
    result[plate_surface] = (255, 255, 255)
    return result


def _batch_segment(input_dir: str, output_dir: str, debug_dir: str | None = None) -> None:
    """Walk input_dir recursively and segment every JPG/PNG into output_dir (flat)."""
    exts = {".jpg", ".jpeg", ".png"}
    paths = []
    for root, _, files in os.walk(input_dir):
        for f in files:
            if os.path.splitext(f)[1].lower() in exts:
                paths.append(os.path.join(root, f))

    os.makedirs(output_dir, exist_ok=True)
    if debug_dir is not None:
        os.makedirs(debug_dir, exist_ok=True)

    for src in tqdm(paths, desc="Segmenting"):
        dst = os.path.join(output_dir, os.path.basename(src))
        file_debug_dir = None
        if debug_dir is not None:
            file_debug_dir = os.path.join(debug_dir, os.path.splitext(os.path.basename(src))[0])
        try:
            segment_image(src, dst, debug_dir=file_debug_dir)
        except Exception as exc:
            print(f"WARN: skipping {src} -- {exc}")


def _cli() -> None:
    parser = argparse.ArgumentParser(
        description="Segment raw food plate images to the standard 3-region format."
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--input", help="Single input image path")
    mode.add_argument(
        "--input_dir", help="Batch mode: directory of raw images (walked recursively)"
    )
    parser.add_argument("--output", help="Single output image path (required with --input)")
    parser.add_argument(
        "--output_dir",
        help="Batch mode: directory to write segmented images (flat)",
    )
    parser.add_argument(
        "--debug_dir",
        help="Directory to write numbered segmentation debug images and masks",
    )
    args = parser.parse_args()

    if args.input:
        if not args.output:
            parser.error("--output is required when using --input")
        img = segment_image(args.input, args.output, debug_dir=args.debug_dir)
        print(f"Saved {args.output}  ({img.size[0]}x{img.size[1]})")
    else:
        if not args.output_dir:
            parser.error("--output_dir is required when using --input_dir")
        _batch_segment(args.input_dir, args.output_dir, debug_dir=args.debug_dir)


if __name__ == "__main__":
    _cli()
