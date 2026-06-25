# Food Waste Estimation -- Dual-Stream CNN with EfficientNet-B0

Multi-task deep learning system that estimates food waste in grams from before/after meal image pairs. Reproduces and extends the methodology from:

> **Automated Food Leftover Estimation Using Deep Learning**  
> https://doi.org/10.1371/journal.pone.0320426

Dataset: **LeFoodSet** -- 678 samples, 34 Indonesian hospital cafeteria food categories.

---

## Overview

Given a pair of segmented images (before and after a meal), the model predicts:

- **Leftover weight in grams** (primary task, regression)
- **Food category** (auxiliary task, 34-class classification)

Target: beat the human visual observer baseline of **MAE = 0.0926** on the normalized leftover scale.

---

## Architecture

Dual-stream CNN with late fusion:

```
Before image (3,224,224) -> EfficientNet-B0 -> feature_before (1280,)
After image  (3,224,224) -> EfficientNet-B0 -> feature_after  (1280,)
                                    |
                          Concatenate -> (2560,)
                                    |
                      FC (2560 -> 1024) + ReLU + Dropout(0.3)
                                    |
              +---------------------+---------------------+
              |                                           |
      Regression head                       Classification head
  FC(1024->512) -> FC(512->1)          FC(1024->512) -> FC(512->34)
  Sigmoid -> leftover_norm (0-1)       Softmax -> food_category
```

Both streams share EfficientNet-B0 weights (Siamese-style). Backbone is pretrained on ImageNet, frozen for the first 5 epochs, then fully unfrozen.

**Loss**: `0.9 x MSE(regression) + 0.1 x CrossEntropy(classification)`  
**Optimizer**: Adam, lr=0.0001

---

## Project Structure

```
food-waste-estimation/
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
│   ├── EDA_LeFoodSet_Leftovers.ipynb
│   ├── training.ipynb              # Full training pipeline (local + Colab)
│   └── inference.ipynb             # Demo: load image pair and predict
├── src/
│   ├── dataset.py                  # FoodWasteDataset + transforms
│   ├── model.py                    # DualStreamEfficientNet
│   ├── train.py                    # 10-fold CV training loop
│   └── utils.py                    # Metrics, seed fixing, logging
├── checkpoints/                    # Best model per fold
└── results/                        # Metrics, logs, training curves
```

---

## Dataset

| Property | Value |
|---|---|
| Samples | 678 |
| Categories | 34 Indonesian foods |
| Input | Segmented images (black background, `data/segmented/`) |
| Metadata | `data/data_original.xlsx` |
| Label | `Weight Before Eaten (g) - Weight After Eaten (g)`, normalized to 0-1 |
| Resolution | ~500x400 or ~700x520, resized to 224x224 |

**Important**: Always use segmented images as input. Never use raw images.

The visual score column (1-7) in the metadata is a human observer rating and is **not** the training target.

---

## Setup

The code runs identically in both environments once dependencies are installed.

### Local (uv)

[uv](https://docs.astral.sh/uv/) is the package manager for local development.

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install all dependencies into a managed virtual environment
uv sync

# Run scripts inside the environment
uv run python src/train.py --folds 10 --epochs 100 --lr 0.0001 --batch_size 16
```

### Google Colab

The entire project folder is stored on Google Drive. Before running any code, mount Drive, set the working directory, and install dependencies with pip:

```python
from google.colab import drive
drive.mount('/content/drive')

import os
os.chdir('/content/drive/MyDrive/food-waste-estimation')  # adjust to your folder name

!pip install -r requirements.txt
```

After that, all relative paths (`data/`, `checkpoints/`, `results/`, `src/`) resolve correctly, and checkpoints are automatically persisted to Drive without any extra configuration.

---

## Training

Local (uv):
```bash
uv run python src/train.py --folds 10 --epochs 100 --lr 0.0001 --batch_size 16
```

Colab:
```bash
python src/train.py --folds 10 --epochs 100 --lr 0.0001 --batch_size 16
```

Training details:
- 10-fold cross-validation (StratifiedKFold on food category)
- Split per fold: 70% train / 20% val / 10% test
- Early stopping: patience = 20 epochs on val MAE
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
  --checkpoint checkpoints/best_fold_1.pth
```

Colab:
```bash
python src/inference.py \
  --before path/to/before.jpg \
  --after  path/to/after.jpg \
  --checkpoint checkpoints/best_fold_1.pth
```

Returns:
```json
{
  "leftover_normalized": 0.34,
  "leftover_grams": 87.2,
  "food_category": "Nasi Putih",
  "confidence": 0.91
}
```

Ensemble inference (all 10 folds, averaged) is available via `notebooks/inference.ipynb`.

---

## Evaluation

| Metric | Target | Baseline |
|---|---|---|
| MAE (normalized) | **< 0.0926** | Human observer: 0.0926 |
| Food classification accuracy | > 90% | N/A |
| RMSE (normalized) | minimize | N/A |

Results are aggregated in `results/summary.json` after all folds complete.

---

## Known Limitations

- **Rice / rice porridge**: white food on white plate degrades segmentation quality and confuses the model
- **Oily / saucy dishes**: residual oil is misclassified as food waste
- **Class imbalance**: Nasi ~78 samples, Tim ~76, most others 20-28

---

## Out of Scope

- Training a segmentation model (ground-truth segmented images are provided)
- Web or mobile deployment
- Real-time video inference
- Foods outside the 34 LeFoodSet categories
