# SPEC.md -- Food Waste Estimation System

**Project**: Multi-Task Deep Learning System for Automated Food Waste Estimation Using Dual-Stream CNN with EfficientNet-B0  
**Reference paper**: https://doi.org/10.1371/journal.pone.0320426  
**Dataset**: LeFoodSet -- 524 usable samples, 34 food categories (678 rows in Excel; 154 skipped automatically due to missing segmented images)  
**Environment**: Local (CPU/GPU) and Google Colab Pro (T4 GPU). The full project folder is stored on Google Drive and mounted in Colab, so all relative paths (`data/`, `checkpoints/`, `results/`) work identically in both environments.  
**Package manager**: `uv` for local development, `pip` for Colab.

---

## 1. Problem Statement

Estimate the weight of food leftover (in grams) from a pair of images taken before and after a meal. The system must outperform human visual observation (baseline MAE = 0.0926 on normalized scale).

---

## 2. Inputs & Outputs

### Input (at inference)

| Input        | Type    | Description                                           |
| ------------ | ------- | ----------------------------------------------------- |
| Before image | JPG/PNG | Segmented food image before eating (black background) |
| After image  | JPG/PNG | Segmented food image after eating (black background)  |

### Output

| Output                 | Type   | Range      | Description                     |
| ---------------------- | ------ | ---------- | ------------------------------- |
| Predicted leftover     | float  | 0.0-1.0    | Normalized leftover level       |
| Predicted leftover (g) | float  | 0-350g     | Denormalized to grams           |
| Food category          | string | 34 classes | Predicted food type (auxiliary) |

### NOT required at inference

- Segmentation masks
- Weight measurements
- Visual score labels

---

## 3. Data Pipeline

### 3.1 Metadata Loading (`src/dataset.py`)

```
Load data_original.xlsx
Compute: Weight Leftover (g) = Weight Before Eaten (g) - Weight After Eaten (g)
Normalize: leftover_normalized = Weight Leftover (g) / max_weight
Encode food categories: LabelEncoder -> integer class IDs (0-33)
Validate: assert no negative leftover weights after cleaning
Save: label_encoder.pkl, normalization_params.json
```

### 3.2 Image Loading

- Images are in subfolders within `data_before/` and `data_after/`
- Use recursive file search (`os.walk`) to find files by filename
- Load segmented versions (ground truth dataset), NOT raw images
- If segmented version not found, raise FileNotFoundError. Do not fall back to raw.

### 3.3 Transforms

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

### 3.4 Dataset Class (`FoodWasteDataset`)

```python
class FoodWasteDataset(Dataset):
    def __getitem__(self, idx):
        return {
            'before': tensor,           # (3, 224, 224)
            'after': tensor,            # (3, 224, 224)
            'leftover_norm': float,     # 0.0-1.0 regression target
            'category': int,            # 0-33 classification target
            'leftover_g': float,        # raw grams (for reporting only)
            'food_name': str            # human-readable (for logging)
        }
```

---

## 4. Model Architecture (`src/model.py`)

### 4.1 Dual-Stream EfficientNet-B0

```
Before image (3,224,224) -> EfficientNet-B0 -> feature_before (1280,)
After image  (3,224,224) -> EfficientNet-B0 -> feature_after  (1280,)
                                    |
                          Concatenate -> (2560,)
                                    |
                            FC Layer (2560 -> 1024) + ReLU + Dropout(0.3)
                                    |
                +-----------------------+-----------------------+
                |                                               |
    Regression head                             Classification head
    FC(1024 -> 512) -> FC(512 -> 1)           FC(1024 -> 512) -> FC(512 -> 34)
    Sigmoid -> leftover_norm (0-1)            Softmax -> food_category
```

### 4.2 Weight Sharing Strategy

- Both streams share the same EfficientNet-B0 weights (Siamese-style)
- Pretrained on ImageNet, freeze backbone for first 5 epochs, unfreeze after

### 4.3 Loss Function

```python
lambda_reg = 0.9
lambda_cls = 0.1

loss_regression     = MSELoss()(pred_leftover, true_leftover_norm)
loss_classification = CrossEntropyLoss()(pred_category, true_category)
total_loss = lambda_reg * loss_regression + lambda_cls * loss_classification
```

---

## 5. Training Pipeline (`src/train.py`)

### 5.1 K-Fold Cross Validation

```
- 10-fold cross-validation (StratifiedKFold on food category)
- Per fold: 70% train, 20% val, 10% test
- Save fold indices to results/fold_indices.json
- Fix seeds: random=42, numpy=42, torch=42, cuda deterministic=True
```

### 5.2 Training Loop Per Fold

```
For each fold:
    1. Initialize model (fresh weights each fold)
    2. Load pretrained EfficientNet-B0 backbone
    3. Freeze backbone, train heads for 5 epochs
    4. Unfreeze all, train with lr=0.0001 (Adam)
    5. Early stopping: patience=20 epochs on val MAE
    6. Save best checkpoint: checkpoints/fold_{n}_best.pth
    7. Log: epoch, train_loss, val_loss, val_mae, val_acc to results/fold_{n}_log.csv
```

### 5.3 Checkpointing

All checkpoints are saved to `checkpoints/` relative to the project root. No environment detection is needed: locally this is the repo directory; on Colab the working directory is set to the mounted project folder on Google Drive, so the path resolves to Drive automatically.

```python
import os

checkpoint = {
    'fold': fold_n,
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'val_mae': best_val_mae,
    'normalization_params': norm_params,
    'label_encoder_classes': label_encoder.classes_.tolist()
}

os.makedirs('checkpoints', exist_ok=True)
torch.save(checkpoint, f'checkpoints/fold_{fold_n}_best.pth')
```

### 5.4 Results Aggregation

```
After all folds:
- Compute mean +/- std of MAE, RMSE across folds
- Compare against human baseline MAE = 0.0926
- Save: results/summary.json, results/training_curves.png
```

---

## 6. Inference Pipeline (`src/inference.py` + `notebooks/inference.ipynb`)

### 6.1 Single Prediction

```python
def predict(before_path, after_path, checkpoint_path):
    # Load model and normalization params from checkpoint
    # Apply val_transform to both images
    # Forward pass -> leftover_norm, food_category
    # Denormalize: leftover_g = leftover_norm * max_weight
    return {
        'leftover_normalized': float,
        'leftover_grams': float,
        'food_category': str,
        'confidence': float
    }
```

### 6.2 Ensemble Inference (optional)

- Load all 10 fold checkpoints
- Average predictions across folds
- Report mean +/- std

---

## 7. File Deliverables

| File                                      | Description                            | Status         |
| ----------------------------------------- | -------------------------------------- | -------------- |
| `notebooks/EDA_LeFoodSet_Leftovers.ipynb` | Exploratory data analysis              | Exists         |
| `notebooks/training.ipynb`                | Full training pipeline (local + Colab) | Exists         |
| `notebooks/inference.ipynb`               | Demo: load image pair and predict      | Exists         |
| `src/dataset.py`                          | Dataset and transforms                 | Exists         |
| `src/model.py`                            | Dual-stream model definition           | Exists         |
| `src/train.py`                            | Training loop with k-fold              | Exists         |
| `src/utils.py`                            | Metrics, helpers, seed fixing          | Exists         |
| `checkpoints/`                            | Saved model weights per fold           | Auto-generated |
| `results/summary.json`                    | Final metrics across all folds         | Auto-generated |
| `pyproject.toml`                          | uv project config and dependencies     | Exists         |
| `requirements.txt`                        | pip dependencies for Colab             | Exists         |

---

## 8. Evaluation Criteria

| Metric                       | Target   | Baseline               |
| ---------------------------- | -------- | ---------------------- |
| MAE (normalized)             | < 0.0926 | Human observer: 0.0926 |
| Food classification accuracy | > 90%    | N/A                    |
| RMSE (normalized)            | Minimize | N/A                    |

---

## 9. Out of Scope

- Training a separate segmentation model (segmented images already provided)
- Web or mobile deployment
- Real-time video inference
- Foods outside the 34 LeFoodSet categories
- 3D volume estimation

---

## 10. End-to-End Verification

The system is working correctly when:

1. `training.ipynb` runs all 10 folds without interruption. On Colab, Drive is mounted and the working directory is set to the project folder before execution, so checkpoints persist automatically.
2. Final mean MAE across folds is < 0.0926
3. Food classification accuracy > 90%
4. `inference.ipynb` accepts two image uploads and returns a predicted waste in grams within 3 seconds
5. Results are reproducible: running training twice with same seeds produces identical fold metrics
