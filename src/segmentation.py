"""
Automatic segmentation of raw food plate images into the ground-truth format:
  black background | pure white plate area | food at original colors

Two-stage algorithm:
  Stage 1 -- SAM ViT-B: center-point + corner-background prompts identify the plate region
  Stage 2 -- HSV + texture within mask: plate surface -> white, food -> original colors
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
_CLOSE_KERNEL = 20  # morphological closing after SAM mask to fill small gaps
_PLATE_SAT_MAX = 40
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

    Args:
        input_path: Path to the raw input image (JPG/PNG).
        output_path: If provided, saves the result to this path.
        debug_dir: If provided, writes intermediate debug images and masks here.
    """
    bgr = cv2.imread(input_path)
    if bgr is None:
        raise FileNotFoundError(f"Cannot read image: {input_path}")

    if debug_dir is not None:
        os.makedirs(debug_dir, exist_ok=True)

    image_id = Path(input_path).stem
    h0, w0 = bgr.shape[:2]
    bgr_sq, pad_top, pad_left = _pad_to_square(bgr)
    if debug_dir is not None:
        _save_debug_step(debug_dir, image_id, 1, 0, "padded", bgr_sq, as_bgr=True)

    side = max(h0, w0)
    bgr_800 = cv2.resize(bgr_sq, (_OUT_SIZE, _OUT_SIZE), interpolation=cv2.INTER_LINEAR)
    if debug_dir is not None:
        _save_debug_step(debug_dir, image_id, 2, 0, "resized", bgr_800, as_bgr=True)

    scale = _OUT_SIZE / float(side)
    prompt_coords = _build_sam_prompt_coords(w0, h0, pad_left, pad_top, scale)
    plate_mask = _detect_plate_sam(
        bgr_800,
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
            _overlay_mask(bgr_800, plate_mask),
            as_bgr=True,
            ext=".jpg",
        )

    result_bgr = _normalize_plate(bgr_800, plate_mask, debug_dir=debug_dir, image_id=image_id)

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


def _build_sam_prompt_coords(w0: int, h0: int, pad_left: int, pad_top: int, scale: float) -> np.ndarray:
    """Build SAM prompt coordinates in padded+resized image space."""
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


def _overlay_prompt_points(image: np.ndarray, prompt_coords: np.ndarray, radius: int = 10) -> np.ndarray:
    overlay = image.copy()
    for idx, (x, y) in enumerate(prompt_coords.tolist()):
        point = (int(round(x)), int(round(y)))
        color = (0, 0, 255) if idx < 5 else (255, 0, 0)
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


def _detect_plate_sam(
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
                _overlay_prompt_points(bgr, prompt_coords),
                as_bgr=True,
                ext=".jpg",
            )

    best = masks[int(np.argmax(scores))].astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (_CLOSE_KERNEL, _CLOSE_KERNEL))
    return np.asarray(cv2.morphologyEx(best, cv2.MORPH_CLOSE, kernel), dtype=np.uint8)


def _normalize_plate(
    bgr: np.ndarray,
    plate_mask: np.ndarray,
    debug_dir: str | None = None,
    image_id: str | None = None,
) -> np.ndarray:
    """
    Stage 2: within the plate mask, classify pixels by HSV + local texture.

    Plate surface: S < _PLATE_SAT_MAX AND V > _PLATE_VAL_MIN AND texture_std < _PLATE_TEX_MAX
    Food:          everything else inside the mask -> keep original BGR
    Outside mask:  (0, 0, 0)

    Texture check is required to preserve white/cream foods (rice, porridge -- ~29% of dataset).
    Rice grains have local std ~10-20; smooth plate surface has local std ~3-8.
    """
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
    plate_condition = (s < _PLATE_SAT_MAX) & (v > _PLATE_VAL_MIN) & (texture_std < _PLATE_TEX_MAX)

    if debug_dir is not None:
        image_id = image_id or "image"
        _save_debug_step(debug_dir, image_id, 4, 1, "hsv_s", s)
        _save_debug_step(debug_dir, image_id, 4, 2, "hsv_v", v)
        _save_debug_step(debug_dir, image_id, 4, 3, "gray", gray)
        _save_debug_step(debug_dir, image_id, 4, 4, "texture_std", texture_std)
        _save_debug_step(debug_dir, image_id, 4, 5, "inside_mask", inside.astype(np.uint8) * 255)
        _save_debug_step(debug_dir, image_id, 4, 6, "plate_condition", plate_condition.astype(np.uint8) * 255)
        _save_debug_step(debug_dir, image_id, 4, 7, "plate_surface_mask", plate_surface.astype(np.uint8) * 255)
        _save_debug_step(
            debug_dir,
            image_id,
            4,
            8,
            "plate_surface_overlay",
            _overlay_mask(bgr, plate_surface.astype(np.uint8) * 255),
            as_bgr=True,
            ext=".jpg",
        )
        _save_debug_step(
            debug_dir,
            image_id,
            4,
            9,
            "inside_overlay",
            _overlay_mask(bgr, inside.astype(np.uint8) * 255),
            as_bgr=True,
            ext=".jpg",
        )

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
        help="Directory to write intermediate debug images and masks",
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
