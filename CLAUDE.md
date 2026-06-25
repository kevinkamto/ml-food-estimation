# Food Waste Estimation -- Dual-Stream CNN with EfficientNet-B0

Multi-task deep learning system that estimates food waste in grams from before/after meal image pairs. Based on the LeFoodSet dataset (Indonesian hospital cafeteria food). Reproduces and adapts the methodology from: https://doi.org/10.1371/journal.pone.0320426

---

## Project Structure

```
food-waste-estimation/
├── CLAUDE.md
├── SPEC.md
├── data/
│   ├── data_original.xlsx          # Metadata: food names, image filenames, weights, visual scores
│   ├── raw/
│   │   ├── data_before/            # Raw before-eating images (subfolders by food category)
│   │   └── data_after/             # Raw after-eating images (subfolders by food category)
│   └── segmented/
│       ├── data_before/            # Ground truth segmented before images (black background)
│       └── data_after/             # Ground truth segmented after images (black background)
├── notebooks/
│   ├── EDA_LeFoodSet_Leftovers.ipynb   # Exploratory data analysis (existing)
│   ├── training.ipynb                  # Training pipeline (to be built)
│   └── inference.ipynb                 # Inference demo (to be built)
├── src/
│   ├── dataset.py                  # PyTorch Dataset class
│   ├── model.py                    # Dual-stream EfficientNet-B0 model
│   ├── train.py                    # Training loop and k-fold CV
│   └── utils.py                    # Helpers: metrics, transforms, logging
├── checkpoints/                    # Saved model weights per fold
└── results/                        # Logs, metrics, plots
```

---

## Dataset

- **678 samples**, 34 Indonesian food categories
- Each sample: before image + after image (both raw and segmented versions)
- Metadata in `data_original.xlsx` with columns: ID, Name of the food, Image Before Eaten, Weight Before Eaten (g), Image After Eaten, Weight After Eaten (g), Visual Estimation by Observer (1-7)
- **Target label**: `Weight Leftover (g) = Weight Before Eaten (g) - Weight After Eaten (g)`, normalized to 0.0-1.0
- **Visual score**: 1 = not consumed at all, 7 = zero remaining (fully eaten), inverse of waste
- Images have two resolution groups: ~500x400px and ~700x520px, always resize to 224x224

---

## Model Architecture

Dual-stream CNN with late fusion (per paper methodology):

- **Stream 1**: Segmented before image -> EfficientNet-B0 -> feature vector
- **Stream 2**: Segmented after image -> EfficientNet-B0 (shared or separate weights) -> feature vector
- **Fusion**: Concatenate both feature vectors
- **Multi-task heads**:
  - Regression head -> normalized leftover value (0.0-1.0), primary task
  - Classification head -> food category (34 classes), auxiliary task
- **Loss**: Combined loss = 0.9 x regression_loss + 0.1 x classification_loss (per paper)
- **Optimizer**: Adam, lr=0.0001
- **Input**: Segmented images only (NOT raw images), background already removed

---

## Training Setup

- **Framework**: PyTorch
- **Environments**: Local machine (CPU or GPU) and Google Colab Pro (T4 GPU), code must run in both
- **Cross-validation**: 10-fold (per paper), save indices for reproducibility
- **Data split per fold**: 70% train / 20% validation / 10% test
- **Early stopping**: Stop after 20 consecutive epochs with no improvement
- **Checkpointing**: Save best-by-validation to `checkpoints/` relative to project root. On Colab, the project folder is mounted from Google Drive, so this path is already persisted on Drive.
- **Random seeds**: Fix for Python, NumPy, PyTorch, and CUDA at start of every run

### Data Augmentation (applied identically to both streams)

- Random horizontal flip
- Random vertical flip
- Random rotation
- Random padding
- Random Gaussian blur
- Random sharpness adjustment
- Random contrast (probability 1/7 each)

### Normalization

- Resize to 224x224
- Normalize with ImageNet stats: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
- **Labels**: Normalize leftover weight to 0.0-1.0 using max weight in dataset

---

## Evaluation Metrics

- **Primary**: MAE on normalized leftover value (target: beat human observer MAE of 0.0926)
- **Secondary**: RMSE, percentage error on grams
- **Auxiliary**: Food classification accuracy (target: >90%)
- **Baseline to beat**: Human visual observer MAE = 0.0926 (from paper)

---

## Style Rules

- NEVER use em-dashes in any written output: documentation, comments, or responses

---

## Environment Rules

- Local development uses `uv` as the package manager (`uv sync`, `uv run python ...`)
- Google Colab uses `pip install -r requirements.txt` (uv is not pre-installed in Colab)
- `pyproject.toml` is the source of truth for dependencies; `requirements.txt` mirrors it for Colab

---

## Key Rules

- ALWAYS use segmented images as input, not raw images
- ALWAYS apply the same augmentation transform to both the before and after image in a pair
- ALWAYS normalize labels to 0.0-1.0 before training, never use raw gram values as target
- ALWAYS save checkpoints inside the project `checkpoints/` folder. On Colab, this persists to Drive because the project itself is on Drive.
- ALWAYS fix random seeds before any split or training operation
- NEVER use the visual score (1-7) as the training target, use weight leftover only
- NEVER load the full dataset into memory, use PyTorch DataLoader with num_workers

---

## Commands

```bash
# Install dependencies (local -- uses uv)
uv sync

# Install dependencies (Google Colab -- uses pip)
pip install -r requirements.txt

# Run training (set working directory to project root first)
python src/train.py --folds 10 --epochs 100 --lr 0.0001 --batch_size 16

# Run inference on a single pair
python src/inference.py --before path/to/before.jpg --after path/to/after.jpg --checkpoint checkpoints/best_fold_1.pth
```

---

## Known Limitations (from paper)

- Rice and rice porridge are the hardest cases, white food on white plate confuses the model
- Oily/saucy dishes cause false positives, model detects oil as food waste
- Dataset is imbalanced: Nasi (rice) ~78 samples, Tim ~76 samples, others 20-28 each
