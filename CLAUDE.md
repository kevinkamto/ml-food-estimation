# Food Waste Estimation -- Dual-Stream CNN with EfficientNet-B0

Single-task deep learning system that estimates food consumption ratio from before/after meal image pairs, then denormalizes to grams. Based on the LeFoodSet dataset (Indonesian hospital cafeteria food). Reproduces and adapts the methodology from: https://doi.org/10.1371/journal.pone.0320426

---

## Project Structure

```
ml-food-waste-estimation/
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
│   ├── LeFoodSet_Leftovers_EDA.ipynb   # Exploratory data analysis (existing)
│   ├── LeFoodSet_Leftovers_Training.ipynb                  # Full training pipeline (local + Colab)
│   └── LeFoodSet_Leftovers_Inference.ipynb                 # Inference demo (single model + ensemble)
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

- **524 usable samples** (678 in Excel, 154 lack segmented images and are skipped automatically), 34 food categories with complete image data
- Each sample: before image + after image (both raw and segmented versions)
- Metadata in `data_original.xlsx` with columns: ID, Name of the food, Image Before Eaten, Weight Before Eaten (g), Image After Eaten, Weight After Eaten (g), Visual Estimation by Observer (1-7)
- **Target label**: `consumption_ratio = Weight_After_Eaten (g) / Weight_Before_Eaten (g)`, clipped to [0, 1]. Denormalize: `w_after_hat = r_hat * w_before`
- **Visual score**: 1 = not consumed at all, 7 = zero remaining (fully eaten), inverse of waste
- Images have two resolution groups: ~500x400px and ~700x520px, always resize to 224x224

---

## Model Architecture

Single-task dual-stream EfficientNet-B0 with enhanced fusion:

- **Stream 1**: Segmented before image -> EfficientNet-B0 -> feat_before (1280,)
- **Stream 2**: Segmented after image -> EfficientNet-B0 (shared weights) -> feat_after (1280,)
- **Fusion**: concat([feat_before, feat_after, |feat_before - feat_after|, area_ratio]) -> (3841,)
- **area_ratio**: scalar = non-black pixels in after_seg / non-black pixels in before_seg
- **Regression head**: FC(3841->1024->512->1) + Sigmoid -> consumption ratio r in [0,1]
- **No classification head** (single-task design)
- **Loss**: HuberLoss(delta=0.1)
- **Optimizer**: Adam, lr=0.0001
- **Input**: Segmented images only (NOT raw images), background already removed

---

## Training Setup

- **Framework**: PyTorch
- **Environments**: Local machine (CPU or GPU) and Google Colab Pro (T4 GPU), code must run in both
- **Cross-validation**: 5-fold GroupKFold grouped by food category (prevents leakage)
- **Data split per fold**: train / val only (no separate test split)
- **Early stopping**: Stop after 20 consecutive epochs with no improvement
- **Scheduler**: ReduceLROnPlateau(factor=0.5, patience=5) on val MAE
- **Sample weighting**: WeightedRandomSampler with inverse-frequency bin weights
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

- **Primary**: MAE on consumption_ratio scale (target: beat human observer MAE of 0.0926)
- **Secondary**: RMSE on consumption_ratio scale
- **Gram-level reporting**: MAE and RMSE on weight_after_grams (informational only, not the training target)
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
- ALWAYS use consumption ratio r = Weight_After / Weight_Before as target, never raw grams
- ALWAYS compute area_ratio from segmented mask pixel counts and pass it to the model
- ALWAYS denormalize predictions at reporting time: w_after_hat = r_hat * w_before
- ALWAYS save checkpoints inside the project `checkpoints/` folder. On Colab, this persists to Drive because the project itself is on Drive.
- ALWAYS fix random seeds before any split or training operation
- NEVER use the visual score (1-7) as the training target, use consumption ratio only
- NEVER load the full dataset into memory, use PyTorch DataLoader with num_workers
- NEVER add a classification head; this is a single-task regression model

---

## Commands

```bash
# Install dependencies (local -- uses uv)
uv sync

# Install dependencies (Google Colab -- uses pip)
pip install -r requirements.txt

# Run training (set working directory to project root first)
python src/train.py --folds 5 --epochs 100 --lr 0.0001 --batch_size 16

# Segment a single raw image (produces 800x800: black background, white plate, food as-is)
python src/segmentation.py --input data/raw/data_before/001/001_001_DSC_0059_bef.JPG --output results/seg_test.jpg

# Batch-segment all raw images in a directory
python src/segmentation.py --input_dir data/raw/data_before --output_dir data/segmented/data_before

# Run inference on pre-segmented images
python src/inference.py --before path/to/before_seg.jpg --after path/to/after_seg.jpg --checkpoint checkpoints/fold_1_best.pth

# Run inference directly on raw images (auto-segments before predicting)
python src/inference.py --before path/to/raw_before.jpg --after path/to/raw_after.jpg --checkpoint checkpoints/fold_1_best.pth --raw

# Save auto-segmented images for inspection during --raw inference
python src/inference.py --before raw_before.jpg --after raw_after.jpg --checkpoint checkpoints/fold_1_best.pth --raw --output_seg results/seg_preview/
```

---

## Known Limitations (from paper)

- Rice and rice porridge are the hardest cases, white food on white plate confuses the model
- Oily/saucy dishes cause false positives, model detects oil as food waste
- Dataset is severely imbalanced: Nasi ~78 samples, Tim ~76, down to 1-2 samples for rare categories (Bubur, Telur orak arik, Tahu goreng)
