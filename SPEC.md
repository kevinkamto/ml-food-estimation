# SPEC.md -- Food Waste Estimation System

**Project**: Single-Task Dual-Stream CNN for Automated Food Waste Estimation Using EfficientNet-B0
**Reference paper**: https://doi.org/10.1371/journal.pone.0320426
**Dataset**: LeFoodSet -- 524 usable samples, 34 food categories (678 rows in Excel; 154 skipped automatically due to missing segmented images)
**Environment**: Local (CPU/GPU) and Google Colab Pro (T4 GPU). The full project folder is stored on Google Drive and mounted in Colab, so all relative paths (`data/`, `checkpoints/`, `results/`) work identically in both environments.
**Package manager**: `uv` for local development, `pip` for Colab.

---

## 1. Problem Statement

Estimate the weight of food remaining after eating (in grams) from a pair of segmented images taken before and after a meal. The system replaces subjective visual scoring by hospital dietitians with an objective, reproducible measurement, and must outperform the human visual observer baseline (MAE = 0.0926).

The core difficulty is that weight is not directly visible: two plates can look similar but weigh very differently due to density and height differences. This is solved by predicting a ratio instead of raw grams.

---

## 2. Inputs & Outputs

### Inputs (at inference)

| Input          | Type    | Description                                                  |
| -------------- | ------- | ------------------------------------------------------------ |
| Before image   | JPG/PNG | Segmented food image before eating (black background)        |
| After image    | JPG/PNG | Segmented food image after eating (black background)         |
| weight_before  | float   | Known serving weight in grams (optional; enables gram output)|

### Outputs

| Output                  | Type  | Range   | Description                                      |
| ----------------------- | ----- | ------- | ------------------------------------------------ |
| consumption_ratio (r)   | float | 0.0-1.0 | Fraction of serving weight remaining after eating|
| weight_after_grams      | float | >= 0    | Denormalized: r * weight_before (if provided)    |
| leftover_grams          | float | >= 0    | (1 - r) * weight_before (if provided)            |

**Denormalization formula**: `w_after_hat = r_hat * w_before`

This cancels out portion size so the model judges only what fraction of the plate remains, which is visible from the images. The before image acts as the reference for the after image.

---

## 3. Data Pipeline

### 3.1 Metadata Loading (`src/dataset.py`)

```
Load data_original.xlsx
Compute: Weight Leftover (g) = Weight Before Eaten (g) - Weight After Eaten (g)
Validate: assert no negative leftover weights
Filter: skip rows where segmented images are missing from disk
Compute target: consumption_ratio = Weight_After / Weight_Before, clipped to [0, 1]
Compute group: food category name (used by GroupKFold)
Save: normalization_params.json (records target formula)
```

### 3.2 Image Loading

- Images are in subfolders within `data_before/` and `data_after/`
- Use recursive file search (`os.walk`) to find files by filename
- Load segmented versions (black background), NOT raw images
- Raise FileNotFoundError if segmented image is missing; do not fall back to raw

### 3.3 Area Ratio Feature

The area ratio is a scalar feature computed from the segmentation masks:

```python
area_ratio = count_nonblack_pixels(after_seg) / count_nonblack_pixels(before_seg)
area_ratio = clip(area_ratio, 0.0, 1.0)
```

This gives the model a direct visual measure of how much of the plate is still covered with food, complementing the CNN features.

### 3.4 Transforms

```python
# Applied identically to both before and after images using same random seed
train_transform = Compose([
    Resize((224, 224)),
    RandomHorizontalFlip(p=1/7),
    RandomVerticalFlip(p=1/7),
    RandomRotation(degrees=15, p=1/7),
    RandomPadding(p=1/7),
    GaussianBlur(p=1/7),
    RandomAdjustSharpness(p=1/7),
    RandomAutocontrast(p=1/7),
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transform = Compose([
    Resize((224, 224)),
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
```

### 3.5 Dataset Class (`FoodWasteDataset`)

```python
class FoodWasteDataset(Dataset):
    def __getitem__(self, idx):
        return {
            'before':            tensor,   # (3, 224, 224)
            'after':             tensor,   # (3, 224, 224)
            'area_ratio':        float,    # non-black pixel fraction after/before
            'consumption_ratio': float,    # 0.0-1.0 regression target
            'weight_before':     float,    # serving weight in grams
            'weight_after':      float,    # actual after weight in grams
            'food_name':         str,      # human-readable label
        }
```

---

## 4. Model Architecture (`src/model.py`)

### 4.1 Dual-Stream EfficientNet-B0 (Single-Task)

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
                    consumption_ratio r  (0-1)
```

### 4.2 Weight Sharing

- Both streams share the same EfficientNet-B0 backbone (Siamese-style)
- Pretrained on ImageNet; backbone is frozen for the first 5 epochs, then unfrozen

### 4.3 Loss Function

```python
loss = HuberLoss(delta=0.1)(pred_ratio, true_ratio)
```

Huber loss is less sensitive to outliers than MSE and does not compress gradients near 0 and 1 like MAE. Delta = 0.1 keeps it nearly MAE-shaped across the bimodal ratio distribution.

---

## 5. Training Pipeline (`src/train.py`)

### 5.1 Cross Validation

```
- 5-fold GroupKFold grouped by food category
  -- prevents the same food type appearing in both train and val within a fold
- No separate test split; report val MAE per fold
- Save fold indices to results/fold_indices.json
- Fix seeds: random=42, numpy=42, torch=42, cuda deterministic=True
```

### 5.2 Sample Weighting

The consumption ratio distribution is bimodal (plates near 0 or near 1). To counteract this:

```python
# Bin ratios into n_bins buckets; weight = 1 / bin_frequency
sample_weights = compute_class_weights(train_df, n_bins=10)
sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights))
```

### 5.3 Training Loop Per Fold

```
For each fold:
    1. Initialize fresh model (EfficientNet-B0 pretrained)
    2. Freeze backbone, train fusion + head for 5 epochs
    3. Unfreeze all, train with Adam lr=0.0001
    4. ReduceLROnPlateau: factor=0.5, patience=5, min_lr=1e-6
    5. EarlyStopping: patience=20 epochs on val MAE
    6. Save best checkpoint: checkpoints/fold_{n}_best.pth
    7. Log per epoch: train_loss, val_loss, val_mae, val_rmse, lr
```

### 5.4 Checkpointing

```python
checkpoint = {
    'fold': fold_n,
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'val_mae': best_val_mae,
    'normalization_params': norm_params,   # contains target formula
}
torch.save(checkpoint, f'checkpoints/fold_{fold_n}_best.pth')
```

### 5.5 Results

```
After all folds:
- Compute mean +/- std of MAE, RMSE across folds
- Compare against human baseline MAE = 0.0926
- Save: results/summary.json
- Save: results/fold_{n}_log.csv (per fold)
- Save: results/all_folds_log.csv (combined)
```

---

## 6. Segmentation Pipeline (`src/segmentation.py`)

Converts a raw food photo into the ground-truth segmented format required by the model.

### 6.1 Output Format

```
black background (outside plate) | pure white plate area | food at original colors
```

The model was trained on images in this exact format. Inference on raw images must go through
this pipeline to produce compatible inputs.

### 6.2 Algorithm

**Stage 1 -- Plate boundary detection**
1. Pad raw image (shorter dimension) to square with black pixels
2. Resize to 800x800
3. Run OpenCV GrabCut with center-biased rect (10% inset): plate always fills the central ~80%
4. Morphological closing (30px kernel) to fill gaps in the GrabCut mask
5. Find the largest contour; compute its convex hull
6. Fill the convex hull to produce a binary plate mask

**Stage 2 -- Plate normalization**
7. Within the hull, classify each pixel by HSV:
   - Plate surface: S < 40 AND V > 150 (low saturation, high brightness) -> (255, 255, 255)
   - Food: all other pixels inside the hull -> original RGB preserved
8. Outside the hull -> (0, 0, 0)

### 6.3 Dish Shape Handling

Two dish shapes exist in the dataset:
- **Round circular plate** (most categories): convex hull approximates a circle/ellipse
- **Large rounded-rectangle plate** (cat 007, some cat 033): convex hull approximates a squircle

Convex hull is shape-agnostic and handles both without geometric assumptions.

### 6.4 CLI Usage

```bash
# Single image
python src/segmentation.py --input data/raw/data_before/001/001_001_DSC_0059_bef.JPG \
    --output results/seg_test.jpg

# Batch (walks subdirectories, writes flat into output_dir)
python src/segmentation.py --input_dir data/raw/data_before --output_dir data/segmented/data_before
```

### 6.5 Known Limitations

- Center-rect GrabCut assumes the dish occupies the central ~80% of the frame. Only valid for the fixed hospital cafeteria camera setup; images from a different angle may require rect adjustment.
- White/cream foods (rice, rice porridge, white tofu) have HSV close to the plate surface and may be partially normalized to white. This is the documented hard case from the paper.
- Convex hull slightly over-fills concave corners on rounded-rectangle plates -- acceptable for consumption ratio estimation.

---

## 7. Inference Pipeline (`src/inference.py` + `notebooks/LeFoodSet_Leftovers_Inference.ipynb`)

### 6.1 Single Prediction

```python
def predict(before_path, after_path, checkpoint_path, weight_before_g=None):
    # Compute area_ratio from pixel counts
    # Load model, apply val_transform, forward pass
    # r_hat = model(before, after, area_ratio)
    # Denormalize if weight_before_g provided:
    #   w_after_hat = r_hat * weight_before_g
    #   leftover_g  = (1 - r_hat) * weight_before_g
    return {'consumption_ratio': r_hat, ...}
```

### 6.2 Ensemble Inference

- Load all 5 fold checkpoints
- Average r_hat across folds; report mean +/- std

---

## 7. File Deliverables

| File                                             | Description                            | Status         |
| ------------------------------------------------ | -------------------------------------- | -------------- |
| `notebooks/LeFoodSet_Leftovers_EDA.ipynb`        | Exploratory data analysis              | Exists         |
| `notebooks/LeFoodSet_Leftovers_Training.ipynb`   | Full training pipeline (local + Colab) | Exists         |
| `notebooks/LeFoodSet_Leftovers_Inference.ipynb`  | Demo: load image pair and predict      | Exists         |
| `src/dataset.py`                                 | Dataset, area_ratio, transforms        | Exists         |
| `src/model.py`                                   | Dual-stream model definition           | Exists         |
| `src/train.py`                                   | Training loop with GroupKFold          | Exists         |
| `src/utils.py`                                   | Metrics, helpers, seed fixing          | Exists         |
| `src/inference.py`                               | CLI inference script (+ --raw flag)    | Exists         |
| `src/segmentation.py`                            | Auto-segmentation: raw -> 800x800      | Exists         |
| `checkpoints/`                                   | Saved model weights per fold           | Auto-generated |
| `results/summary.json`                           | Final metrics across all folds         | Auto-generated |
| `results/all_folds_log.csv`                      | Combined per-epoch training log        | Auto-generated |
| `pyproject.toml`                                 | uv project config and dependencies     | Exists         |
| `requirements.txt`                               | pip dependencies for Colab             | Exists         |

---

## 8. Evaluation Criteria

| Metric              | Target   | Baseline               |
| ------------------- | -------- | ---------------------- |
| MAE (ratio scale)   | < 0.0926 | Human observer: 0.0926 |
| RMSE (ratio scale)  | Minimize | N/A                    |

---

## 9. Out of Scope

- Training a separate segmentation model (segmented images already provided)
- Food category classification (single-task design)
- Web or mobile deployment
- Real-time video inference
- Foods outside the 34 LeFoodSet categories
- 3D volume estimation

---

## 10. End-to-End Verification

The system is working correctly when:

1. `LeFoodSet_Leftovers_Training.ipynb` runs all 5 folds without interruption. On Colab, Drive is mounted and the working directory is set to the project folder before execution, so checkpoints persist automatically.
2. Final mean MAE across folds is < 0.0926
3. `LeFoodSet_Leftovers_Inference.ipynb` accepts two image uploads and an optional serving weight, and returns a consumption ratio and predicted grams within 3 seconds
4. Results are reproducible: running training twice with the same seeds produces identical fold metrics
