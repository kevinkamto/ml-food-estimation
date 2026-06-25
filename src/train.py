import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from loguru import logger
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset import FoodWasteDataset, get_transforms, load_metadata
from model import DualStreamEfficientNet
from utils import compute_mae, compute_rmse, set_seed


def train_fold(
    model,
    train_loader,
    val_loader,
    optimizer,
    device,
    epochs,
    patience,
    fold_n,
    checkpoint_dir,
    norm_params,
    label_classes,
):
    mse_loss = nn.MSELoss()
    ce_loss = nn.CrossEntropyLoss()

    best_val_mae = float("inf")
    epochs_no_improve = 0
    log_rows = []

    for epoch in range(1, epochs + 1):
        if epoch == 6:
            model.unfreeze_backbone()

        model.train()
        train_losses = []
        for batch in train_loader:
            before = batch["before"].to(device)
            after = batch["after"].to(device)
            leftover_norm = batch["leftover_norm"].to(device)
            category = batch["category"].to(device)

            optimizer.zero_grad()
            pred_leftover, pred_category = model(before, after)
            loss = 0.9 * mse_loss(pred_leftover, leftover_norm) + 0.1 * ce_loss(
                pred_category, category
            )
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        val_preds, val_targets = [], []
        val_cat_preds, val_cat_targets = [], []
        val_losses = []

        with torch.no_grad():
            for batch in val_loader:
                before = batch["before"].to(device)
                after = batch["after"].to(device)
                leftover_norm = batch["leftover_norm"].to(device)
                category = batch["category"].to(device)

                pred_leftover, pred_category = model(before, after)
                loss = 0.9 * mse_loss(pred_leftover, leftover_norm) + 0.1 * ce_loss(
                    pred_category, category
                )
                val_losses.append(loss.item())
                val_preds.append(pred_leftover.cpu())
                val_targets.append(leftover_norm.cpu())
                val_cat_preds.append(pred_category.argmax(dim=1).cpu())
                val_cat_targets.append(category.cpu())

        val_preds = torch.cat(val_preds)
        val_targets = torch.cat(val_targets)
        val_cat_preds = torch.cat(val_cat_preds)
        val_cat_targets = torch.cat(val_cat_targets)

        val_mae = compute_mae(val_preds, val_targets)
        val_rmse = compute_rmse(val_preds, val_targets)
        val_acc = float((val_cat_preds == val_cat_targets).float().mean())

        logger.info(
            f"Fold {fold_n} | Epoch {epoch:3d} | "
            f"Train Loss: {np.mean(train_losses):.4f} | "
            f"Val Loss: {np.mean(val_losses):.4f} | "
            f"Val MAE: {val_mae:.4f} | Val Acc: {val_acc:.4f}"
        )

        log_rows.append(
            {
                "epoch": epoch,
                "train_loss": float(np.mean(train_losses)),
                "val_loss": float(np.mean(val_losses)),
                "val_mae": val_mae,
                "val_rmse": val_rmse,
                "val_acc": val_acc,
            }
        )

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            epochs_no_improve = 0
            os.makedirs(checkpoint_dir, exist_ok=True)
            torch.save(
                {
                    "fold": fold_n,
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_mae": best_val_mae,
                    "normalization_params": norm_params,
                    "label_encoder_classes": label_classes,
                },
                os.path.join(checkpoint_dir, f"fold_{fold_n}_best.pth"),
            )
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                logger.info(
                    f"Early stopping at epoch {epoch} (no improvement for {patience} epochs)"
                )
                break

    return best_val_mae, pd.DataFrame(log_rows)


def main():
    parser = argparse.ArgumentParser(description="Train dual-stream food waste estimator")
    parser.add_argument("--folds", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=2)
    args = parser.parse_args()

    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    data_dir = "data"
    before_dir = os.path.join(data_dir, "segmented", "data_before")
    after_dir = os.path.join(data_dir, "segmented", "data_after")
    checkpoint_dir = "checkpoints"
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)

    df, norm_params, le = load_metadata(
        os.path.join(data_dir, "data_original.xlsx"),
        before_dir=before_dir,
        after_dir=after_dir,
        save_dir=results_dir,
    )
    label_classes = le.classes_.tolist()

    train_transform = get_transforms("train")
    val_transform = get_transforms("val")

    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    labels = df["category_id"].values
    fold_indices = {}
    fold_maes = []

    for fold_n, (trainval_idx, test_idx) in enumerate(
        skf.split(np.arange(len(df)), labels), start=1
    ):
        logger.info(f"{'=' * 60}")
        logger.info(f"Fold {fold_n}/{args.folds}")
        logger.info(f"{'=' * 60}")

        set_seed(args.seed + fold_n)

        n_val = int(len(trainval_idx) * (0.2 / 0.9))
        np.random.shuffle(trainval_idx)
        val_idx = trainval_idx[:n_val]
        train_idx = trainval_idx[n_val:]

        fold_indices[f"fold_{fold_n}"] = {
            "train": train_idx.tolist(),
            "val": val_idx.tolist(),
            "test": test_idx.tolist(),
        }

        train_ds = FoodWasteDataset(df.iloc[train_idx], before_dir, after_dir, train_transform)
        val_ds = FoodWasteDataset(df.iloc[val_idx], before_dir, after_dir, val_transform)

        train_loader = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
        )

        model = DualStreamEfficientNet(num_classes=len(label_classes), pretrained=True).to(device)
        model.freeze_backbone()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

        best_mae, log_df = train_fold(
            model,
            train_loader,
            val_loader,
            optimizer,
            device,
            args.epochs,
            args.patience,
            fold_n,
            checkpoint_dir,
            norm_params,
            label_classes,
        )
        fold_maes.append(best_mae)
        log_df.to_csv(os.path.join(results_dir, f"fold_{fold_n}_log.csv"), index=False)
        logger.info(f"Fold {fold_n} best val MAE: {best_mae:.4f}")

    with open(os.path.join(results_dir, "fold_indices.json"), "w") as f:
        json.dump(fold_indices, f, indent=2)

    summary = {
        "fold_maes": fold_maes,
        "mean_mae": float(np.mean(fold_maes)),
        "std_mae": float(np.std(fold_maes)),
        "human_baseline_mae": 0.0926,
        "beats_baseline": float(np.mean(fold_maes)) < 0.0926,
    }
    with open(os.path.join(results_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"Final Mean MAE: {summary['mean_mae']:.4f} +/- {summary['std_mae']:.4f}")
    logger.info(f"Beats human baseline (0.0926): {summary['beats_baseline']}")


if __name__ == "__main__":
    main()
