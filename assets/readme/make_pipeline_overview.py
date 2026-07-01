"""One-off script to render assets/readme/pipeline_overview.png, a top-level
banner showing raw photos -> segmentation -> model -> output. Not part of the
training/inference pipeline; kept for reproducing the README asset.
"""
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
OUT_PATH = Path(__file__).parent / "pipeline_overview.png"

BEFORE_RAW = ROOT / "data/raw/data_before/001/001_031_DSC_0343_bef.JPG"
AFTER_RAW = ROOT / "data/raw/data_after/001/001_031_DSC_0393_aft.JPG"
BEFORE_SEG = ROOT / "data/segmented/data_before/001_001_031_DSC_0343_bef.JPG"
AFTER_SEG = ROOT / "data/segmented/data_after/001_001_031_DSC_0393_aft.JPG"

fig = plt.figure(figsize=(13, 4.2))
gs = fig.add_gridspec(2, 6, width_ratios=[1.1, 0.55, 1.1, 0.7, 1.3, 1.0],
                       height_ratios=[1, 1], hspace=0.15, wspace=0.15)

ax_raw_b = fig.add_subplot(gs[0, 0])
ax_raw_a = fig.add_subplot(gs[1, 0])
ax_seg_b = fig.add_subplot(gs[0, 2])
ax_seg_a = fig.add_subplot(gs[1, 2])

for ax, path, label in [
    (ax_raw_b, BEFORE_RAW, "Raw before"),
    (ax_raw_a, AFTER_RAW, "Raw after"),
    (ax_seg_b, BEFORE_SEG, "Segmented before"),
    (ax_seg_a, AFTER_SEG, "Segmented after"),
]:
    ax.imshow(Image.open(path).convert("RGB"))
    ax.set_xlabel(label, fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

# Arrow between raw and segmented columns
ax_arrow1 = fig.add_subplot(gs[:, 1])
ax_arrow1.axis("off")
ax_arrow1.set_xlim(0, 1)
ax_arrow1.set_ylim(0, 1)
ax_arrow1.add_patch(FancyArrowPatch((0.05, 0.5), (0.95, 0.5),
                                     arrowstyle="-|>", mutation_scale=18,
                                     linewidth=1.6, color="#5A5A5A"))
ax_arrow1.text(0.5, 0.62, "SAM\nsegmentation", ha="center", fontsize=8.5,
               color="#5A5A5A", style="italic")

# Model box
ax_model = fig.add_subplot(gs[:, 4])
ax_model.axis("off")
ax_model.set_xlim(0, 1)
ax_model.set_ylim(0, 1)
box = FancyBboxPatch((0.05, 0.2), 0.9, 0.6, boxstyle="round,pad=0.03,rounding_size=0.06",
                      linewidth=1.2, edgecolor="#C77B2C", facecolor="#C77B2C", alpha=0.92)
ax_model.add_patch(box)
ax_model.text(0.5, 0.5, "Dual-Stream\nEfficientNet-B0", ha="center", va="center",
              fontsize=10.5, color="white", fontweight="bold")

ax_arrow2 = fig.add_subplot(gs[:, 3])
ax_arrow2.axis("off")
ax_arrow2.set_xlim(0, 1)
ax_arrow2.set_ylim(0, 1)
ax_arrow2.add_patch(FancyArrowPatch((0.05, 0.5), (0.95, 0.5),
                                     arrowstyle="-|>", mutation_scale=18,
                                     linewidth=1.6, color="#5A5A5A"))

# Output box
ax_out = fig.add_subplot(gs[:, 5])
ax_out.axis("off")
ax_out.set_xlim(0, 1)
ax_out.set_ylim(0, 1)
ax_out.add_patch(FancyArrowPatch((-0.15, 0.5), (0.1, 0.5),
                                  arrowstyle="-|>", mutation_scale=18,
                                  linewidth=1.6, color="#5A5A5A", clip_on=False))
out_box = FancyBboxPatch((0.12, 0.2), 0.85, 0.6, boxstyle="round,pad=0.03,rounding_size=0.06",
                          linewidth=1.2, edgecolor="#1F1F1F", facecolor="#1F1F1F", alpha=0.92)
ax_out.add_patch(out_box)
ax_out.text(0.545, 0.56, "r = 0.98", ha="center", va="center",
            fontsize=11, color="white", fontweight="bold")
ax_out.text(0.545, 0.34, "consumption ratio", ha="center", va="center",
            fontsize=7.5, color="#DDDDDD")

fig.suptitle("From raw meal photos to a consumption ratio prediction", fontsize=13, fontweight="bold", y=1.0)
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(OUT_PATH, dpi=150)
print(f"saved {OUT_PATH}")
