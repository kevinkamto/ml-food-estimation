# Food Waste Estimation -- Dual-Stream CNN with EfficientNet-B0

Single-task deep learning system that estimates food consumption ratio from before/after meal image pairs, then denormalizes to grams. Reproduces and extends the methodology from:

> **Automated Food Leftover Estimation Using Deep Learning**
> https://doi.org/10.1371/journal.pone.0320426

Dataset: **LeFoodSet** -- 524 usable samples across 34 food categories (678 rows in Excel; 154 lack segmented images and are skipped automatically).

---

## Overview

Given a pair of segmented images (before and after a meal), the model predicts:

- **Consumption ratio** `r = w_after / w_before` in [0, 1] (primary output, regression)
- **Weight after eating** in grams (optional, requires known serving weight)

Target: beat the human visual observer baseline of **MAE = 0.0926** on the consumption ratio scale.

---

## Architecture

Single-task dual-stream EfficientNet-B0 with enhanced fusion:

```
Before image (3,224,224) -> EfficientNet-B0 -> feat_before (1280,)
After  image (3,224,224) -> EfficientNet-B0 -> feat_after  (1280,)
                                    |
            |feat_before - feat_after| -> diff (1280,)
                                    |
  concat([feat_before, feat_after, diff, area_ratio]) -> (3841,)
                                    |
              FC(3841->1024) + ReLU + Dropout(0.3)
              FC(1024->512)  + ReLU + Dropout(0.2)
                                    |
                      FC(512->1) + Sigmoid
                                    |
                    consumption_ratio r  in [0, 1]
```

Both streams share EfficientNet-B0 weights (Siamese-style). Backbone is pretrained on ImageNet, frozen for the first 5 epochs, then fully unfrozen (LR reset to initial value on unfreeze).

**area_ratio**: non-black pixel count of after image / before image -- gives the model an explicit visual coverage signal.

**Loss**: `HuberLoss(delta=0.1)`
**Optimizer**: Adam, lr=0.0001
**Denormalize**: `w_after_hat = r_hat * w_before`

---

## Project Structure

```
ml-food-waste-estimation/
├── CLAUDE.md                       # Agent instructions and project rules
├── SPEC.md                         # Detailed technical specification
├── data/
│   ├── data_original.xlsx          # Metadata: filenames, weights, visual scores
│   ├── raw/
│   │   ├── data_before/            # Raw before-eating images (by food category)
│   │   └── data_after/             # Raw after-eating images (by food category)
│   └── segmented/
│       ├── data_before/            # Segmented before images (black background)
│       └── data_after/             # Segmented after images (black background)
├── notebooks/
│   ├── LeFoodSet_Leftovers_EDA.ipynb
│   ├── LeFoodSet_Leftovers_Training.ipynb    # Full training pipeline (local + Colab)
│   └── LeFoodSet_Leftovers_Inference.ipynb   # Demo: load image pair and predict
├── src/
│   ├── dataset.py                  # FoodWasteDataset, area_ratio, transforms
│   ├── model.py                    # DualStreamEfficientNet (single-task)
│   ├── train.py                    # 5-fold GroupKFold training loop
│   ├── inference.py                # CLI inference script
│   └── utils.py                    # Metrics, seed fixing
├── checkpoints/                    # Best model per fold
└── results/                        # Metrics, logs, training curves
```

---

## Dataset

| Property | Value |
|---|---|
| Samples | 524 usable (678 in Excel, 154 skipped -- no segmented images) |
| Categories | 34 Indonesian foods |
| Input | Segmented images (black background, `data/segmented/`) |
| Metadata | `data/data_original.xlsx` |
| Label | `consumption_ratio = Weight_After / Weight_Before`, clipped to [0, 1] |
| Resolution | ~500x400 or ~700x520, resized to 224x224 |

**Important**: Always use segmented images as input. Never use raw images.

The visual score column (1-7) in the metadata is a human observer rating and is **not** the training target.

---

## Setup

### Local (uv)

[uv](https://docs.astral.sh/uv/) is the package manager for local development.

```bash
# Install all dependencies into a managed virtual environment
uv sync

# Run scripts inside the environment
uv run python src/train.py --folds 5 --epochs 100 --lr 0.0001 --batch_size 16
```

### Google Colab

The entire project folder is stored on Google Drive. Before running any code, mount Drive, set the working directory, and install dependencies with pip:

```python
from google.colab import drive
drive.mount('/content/drive')

import os
os.chdir('/content/drive/MyDrive/ml-food-waste-estimation')  # adjust to your folder name

!pip install -r requirements.txt
```

After that, all relative paths (`data/`, `checkpoints/`, `results/`, `src/`) resolve correctly, and checkpoints are automatically persisted to Drive.

---

## Training

Local (uv):
```bash
uv run python src/train.py --folds 5 --epochs 100 --lr 0.0001 --batch_size 16
```

Colab:
```bash
python src/train.py --folds 5 --epochs 100 --lr 0.0001 --batch_size 16
```

Training details:
- 5-fold GroupKFold (grouped by food category to prevent leakage)
- WeightedRandomSampler with inverse-frequency bin weights (bimodal distribution)
- ReduceLROnPlateau(factor=0.5, patience=5); LR reset when backbone unfreezes at epoch 6
- Early stopping: patience=20 epochs on val MAE
- Random seeds fixed at 42 for Python, NumPy, PyTorch, and CUDA

Checkpoints are saved to `checkpoints/fold_{n}_best.pth` relative to the project root.

### Augmentation

Both streams receive the **same** random augmentation each sample:

| Transform | Probability |
|---|---|
| Random horizontal flip | 1/7 |
| Random vertical flip | 1/7 |
| Random rotation (+-15 deg) | 1/7 |
| Random padding | 1/7 |
| Gaussian blur | 1/7 |
| Random sharpness | 1/7 |
| Random contrast | 1/7 |

---

## Inference

Local (uv):
```bash
uv run python src/inference.py \
  --before path/to/before.jpg \
  --after  path/to/after.jpg \
  --checkpoint checkpoints/fold_1_best.pth
```

With optional serving weight for gram output:
```bash
uv run python src/inference.py \
  --before path/to/before.jpg \
  --after  path/to/after.jpg \
  --checkpoint checkpoints/fold_1_best.pth \
  --weight_before 250
```

Returns:
```json
{
  "consumption_ratio": 0.62,
  "area_ratio": 0.58,
  "weight_after_grams": 155.0,
  "leftover_grams": 95.0,
  "weight_before_grams": 250.0
}
```

Ensemble inference (all 5 folds, averaged) is available via `notebooks/LeFoodSet_Leftovers_Inference.ipynb`.

---

## Evaluation

| Metric | Target | Baseline |
|---|---|---|
| MAE (consumption ratio) | **< 0.0926** | Human observer: 0.0926 |
| RMSE (consumption ratio) | minimize | N/A |

Results are aggregated in `results/summary.json` after all folds complete.

---

## Known Limitations

- **Rice / rice porridge**: white food on white plate degrades segmentation quality and confuses the model
- **Oily / saucy dishes**: residual oil is misclassified as food waste
- **Class imbalance**: Nasi ~78 samples, Tim ~76, most others 20-28

---

## Out of Scope

- Training a segmentation model (ground-truth segmented images are provided)
- Food category classification (single-task design)
- Web or mobile deployment
- Real-time video inference
- Foods outside the 34 LeFoodSet categories
