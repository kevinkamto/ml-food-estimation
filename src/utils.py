import os
import random

import numpy as np
from PIL import Image
import torch


def save_ndarray_image(array: np.ndarray, output_path: str, *, as_bgr: bool = False) -> None:
    """
    Save a NumPy image array to disk.

    Supports:
      - grayscale arrays (H, W)
      - RGB arrays (H, W, 3)
      - BGR arrays when as_bgr=True
      - boolean arrays and float arrays in [0, 1]
    """
    if array.dtype == bool:
        array = (array.astype(np.uint8) * 255)
    elif np.issubdtype(array.dtype, np.floating):
        array = np.clip(array, 0.0, 1.0)
        array = (array * 255).astype(np.uint8)
    elif array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)

    if array.ndim == 2:
        mode = "L"
    elif array.ndim == 3 and array.shape[2] == 3:
        if as_bgr:
            array = array[..., ::-1]
        mode = "RGB"
    elif array.ndim == 3 and array.shape[2] == 4:
        if as_bgr:
            array = array[..., [2, 1, 0, 3]]
        mode = "RGBA"
    else:
        raise ValueError(f"Cannot save ndarray with shape {array.shape} as an image")

    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
    Image.fromarray(array, mode=mode).save(output_path)


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_mae(preds, targets):
    return float(torch.mean(torch.abs(preds - targets)))


def compute_rmse(preds, targets):
    return float(torch.sqrt(torch.mean((preds - targets) ** 2)))
