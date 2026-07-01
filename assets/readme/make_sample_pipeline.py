"""One-off script to render assets/readme/sample_pipeline.png from real dataset samples.
Not part of the training/inference pipeline; kept for reproducing the README asset.
"""
from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]

PAIRS = [
    ("Raw -- Before", ROOT / "data/raw/data_before/001/001_031_DSC_0343_bef.JPG"),
    ("Segmented -- Before", ROOT / "data/segmented/data_before/001_001_031_DSC_0343_bef.JPG"),
    ("Raw -- After", ROOT / "data/raw/data_after/001/001_031_DSC_0393_aft.JPG"),
    ("Segmented -- After", ROOT / "data/segmented/data_after/001_001_031_DSC_0393_aft.JPG"),
]

fig, axes = plt.subplots(2, 2, figsize=(9, 7))
for ax, (title, path) in zip(axes.flat, PAIRS):
    img = Image.open(path).convert("RGB")
    ax.imshow(img)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.axis("off")

fig.suptitle(
    "Segmentation removes the background and isolates the food on each plate",
    fontsize=12,
    y=0.99,
)
fig.tight_layout(rect=[0, 0, 1, 0.96])
out_path = Path(__file__).parent / "sample_pipeline.png"
fig.savefig(out_path, dpi=150)
print(f"saved {out_path}")
