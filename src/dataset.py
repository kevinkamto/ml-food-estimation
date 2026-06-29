import json
import os
import random as _random

import numpy as np
import pandas as pd
import torch
from loguru import logger
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


def load_metadata(xlsx_path: str, before_dir: str, after_dir: str, save_dir: str = ".") -> tuple:
    """
    Load and filter metadata. Returns (df, norm_params).

    Target: consumption ratio r = Weight_After / Weight_Before in [0, 1].
    Denormalize at inference: w_after_hat = r_hat * w_before.
    """
    df = pd.read_excel(xlsx_path)

    # SAFETY: Ensure before-weight is always positive (clamp to avoid division by zero)
    df["Weight Before Eaten (g)"] = df["Weight Before Eaten (g)"].clip(lower=0.1)

    # SAFETY: After-weight cannot be negative
    df["Weight After Eaten (g)"] = df["Weight After Eaten (g)"].clip(lower=0.0)

    # SAFETY: Ensure after-weight never exceeds before-weight (clamp invalid data)
    invalid_mask = df["Weight After Eaten (g)"] > df["Weight Before Eaten (g)"]
    if invalid_mask.any():
        logger.warning(
            f"Found {invalid_mask.sum()} samples where after-weight > before-weight. Capping to before-weight."
        )
        df.loc[invalid_mask, "Weight After Eaten (g)"] = df.loc[invalid_mask, "Weight Before Eaten (g)"]

    # Compute leftovers (always >= 0 by construction)
    df["Weight Leftover (g)"] = df["Weight Before Eaten (g)"] - df["Weight After Eaten (g)"]

    # Final safety clamp to ensure no numerical errors
    df["Weight Leftover (g)"] = df["Weight Leftover (g)"].clip(lower=0.0)
    if (df["Weight Leftover (g)"] < 0).any():
        logger.error("Critical: Negative leftovers detected even after clipping. Check source data.")

    # Filter to rows where both segmented images exist on disk
    available_bef = {f for _, _, files in os.walk(before_dir) for f in files}
    available_aft = {f for _, _, files in os.walk(after_dir) for f in files}
    mask = df["Image Before Eaten"].apply(_seg_filename).isin(available_bef) & df[
        "Image After Eaten"
    ].apply(_seg_filename).isin(available_aft)
    n_dropped = (~mask).sum()
    if n_dropped > 0:
        logger.info(
            f"Skipped {n_dropped} samples with missing segmented images ({mask.sum()} usable)."
        )
    df = df[mask].reset_index(drop=True)

    # Consumption ratio: fraction of serving weight remaining after eating
    df["consumption_ratio"] = df["Weight After Eaten (g)"] / df["Weight Before Eaten (g)"]
    df["consumption_ratio"] = df["consumption_ratio"].clip(0.0, 1.0)

    # Group label for GroupKFold -- group by food category to prevent leakage
    df["group"] = df["Name of the food"]

    os.makedirs(save_dir, exist_ok=True)
    norm_params = {
        "target": "consumption_ratio",
        "description": "r = w_after / w_before; denormalize: w_after_hat = r_hat * w_before",
    }
    with open(os.path.join(save_dir, "normalization_params.json"), "w") as f:
        json.dump(norm_params, f, indent=2)

    return df, norm_params


def compute_class_weights(df: pd.DataFrame, n_bins: int = 10) -> torch.Tensor:
    """
    Inverse-frequency weights for WeightedRandomSampler.
    Bins the continuous consumption_ratio into n_bins buckets and returns
    per-sample weights proportional to 1 / bin_frequency.
    """
    ratios = df["consumption_ratio"].values
    bin_indices = np.digitize(ratios, np.linspace(0, 1, n_bins + 1)[1:-1])
    bin_counts = np.bincount(bin_indices, minlength=n_bins)
    bin_counts = np.where(bin_counts == 0, 1, bin_counts)
    weights = 1.0 / bin_counts[bin_indices]
    weights = weights / weights.sum() * len(weights)
    return torch.tensor(weights, dtype=torch.float32)


def _seg_filename(raw_filename: str) -> str:
    # Segmented files are named {category}_{raw_filename}, e.g.
    # raw: 001_001_DSC_0059_bef.JPG -> segmented: 001_001_001_DSC_0059_bef.JPG
    cat = raw_filename[:3]
    return f"{cat}_{raw_filename}"


def find_image(root_dir: str, filename: str) -> str:
    for dirpath, _, files in os.walk(root_dir):
        if filename in files:
            return os.path.join(dirpath, filename)
    raise FileNotFoundError(f"Image '{filename}' not found under {root_dir}")


def get_transforms(mode: str = "train") -> transforms.Compose:
    if mode == "train":
        return transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(p=1 / 7),
                transforms.RandomVerticalFlip(p=1 / 7),
                transforms.RandomApply([transforms.RandomRotation(degrees=15)], p=1 / 7),
                transforms.RandomApply(
                    [transforms.Compose([transforms.Pad(20), transforms.Resize((224, 224))])],
                    p=1 / 7,
                ),
                transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=1 / 7),
                transforms.RandomAdjustSharpness(sharpness_factor=2, p=1 / 7),
                transforms.RandomApply([transforms.ColorJitter(contrast=0.5)], p=1 / 7),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
    return transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def _apply_pair_transform(
    transform: transforms.Compose, img1: Image.Image, img2: Image.Image
) -> tuple:
    seed = torch.randint(0, 2**32, (1,)).item()
    _random.seed(seed)
    torch.manual_seed(seed)
    t1 = transform(img1)
    _random.seed(seed)
    torch.manual_seed(seed)
    t2 = transform(img2)
    return t1, t2


class FoodWasteDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        before_dir: str,
        after_dir: str,
        transform: transforms.Compose | None = None,
    ) -> None:
        self.df = df.reset_index(drop=True)
        self.before_dir = before_dir
        self.after_dir = after_dir
        self.transform = transform
        # Resolve paths once at init to avoid repeated os.walk on every __getitem__
        self._before_paths = [
            find_image(before_dir, _seg_filename(row["Image Before Eaten"]))
            for _, row in self.df.iterrows()
        ]
        self._after_paths = [
            find_image(after_dir, _seg_filename(row["Image After Eaten"]))
            for _, row in self.df.iterrows()
        ]

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]

        before_path = self._before_paths[idx]
        after_path = self._after_paths[idx]

        before_img = Image.open(before_path).convert("RGB")
        after_img = Image.open(after_path).convert("RGB")

        # Area ratio computed from already-loaded arrays to avoid a second disk read
        before_arr = np.array(before_img)
        after_arr = np.array(after_img)
        before_area = float(np.any(before_arr > 0, axis=2).sum())
        after_area = float(np.any(after_arr > 0, axis=2).sum())
        area_ratio = float(after_area / before_area) if before_area > 0 else 0.0
        area_ratio = min(area_ratio, 1.0)

        if self.transform:
            before_img, after_img = _apply_pair_transform(self.transform, before_img, after_img)

        return {
            "before": before_img,
            "after": after_img,
            "area_ratio": torch.tensor(area_ratio, dtype=torch.float32),
            "consumption_ratio": torch.tensor(row["consumption_ratio"], dtype=torch.float32),
            "weight_before": float(row["Weight Before Eaten (g)"]),
            "weight_after": float(row["Weight After Eaten (g)"]),
            "food_name": str(row["Name of the food"]),
        }
