import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from loguru import logger
from sklearn.model_selection import GroupKFold
from torch.utils.data import DataLoader, WeightedRandomSampler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset import FoodWasteDataset, compute_class_weights, get_transforms, load_metadata
from model import DualStreamEfficientNet
from utils import compute_mae, compute_rmse, set_seed


def train_fold(
    model: DualStreamEfficientNet,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epochs: int,
    patience: int,
    frozen_epochs: int,
    fold_n: int,
    checkpoint_dir: str,
    norm_params: dict,
) -> tuple[float, float, float, float, pd.DataFrame]:
    criterion = nn.HuberLoss(delta=0.1)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-6
    )

    best_val_mae = float("inf")
    epochs_no_improve = 0
    log_rows = []

    for epoch in range(1, epochs + 1):
        if epoch == frozen_epochs + 1:
            model.unfreeze_backbone()
            optimizer.param_groups[1]["lr"] = optimizer.defaults["lr"]
            # Reset scheduler so LR-reduction patience counts from the unfreeze point.
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-6
            )

        # Train
        model.train()
        train_losses: list[float] = []
        for batch in train_loader:
            before = batch["before"].to(device)
            after = batch["after"].to(device)
            area_ratio = batch["area_ratio"].to(device)
            target = batch["consumption_ratio"].to(device)

            optimizer.zero_grad()
            pred = model(before, after, area_ratio)
            loss = criterion(pred, target)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        # Validate
        model.eval()
        val_preds: list[torch.Tensor] = []
        val_targets: list[torch.Tensor] = []
        val_losses: list[float] = []

        with torch.no_grad():
            for batch in val_loader:
                before = batch["before"].to(device)
                after = batch["after"].to(device)
                area_ratio = batch["area_ratio"].to(device)
                target = batch["consumption_ratio"].to(device)

                pred = model(before, after, area_ratio)
                loss = criterion(pred, target)
                val_losses.append(loss.item())
                val_preds.append(pred.cpu())
                val_targets.append(target.cpu())

        val_preds_t = torch.cat(val_preds)
        val_targets_t = torch.cat(val_targets)

        val_mae = compute_mae(val_preds_t, val_targets_t)
        val_rmse = compute_rmse(val_preds_t, val_targets_t)
        train_loss = float(np.mean(train_losses))
        val_loss = float(np.mean(val_losses))

        scheduler.step(val_mae)

        logger.info(
            f"Fold {fold_n} | Epoch {epoch:3d} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val MAE: {val_mae:.4f} | "
            f"LR: {optimizer.param_groups[0]['lr']:.2e}"
        )

        log_rows.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_mae": val_mae,
                "val_rmse": val_rmse,
                "lr": optimizer.param_groups[0]["lr"],
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

    # Evaluate on held-out test set using the best checkpoint
    best_ckpt = torch.load(
        os.path.join(checkpoint_dir, f"fold_{fold_n}_best.pth"),
        map_location=device,
        weights_only=False,
    )
    model.load_state_dict(best_ckpt["model_state_dict"])
    model.eval()
    test_preds: list[torch.Tensor] = []
    test_targets: list[torch.Tensor] = []
    test_wb: list[torch.Tensor] = []
    with torch.no_grad():
        for batch in test_loader:
            before = batch["before"].to(device)
            after = batch["after"].to(device)
            area_ratio = batch["area_ratio"].to(device)
            target = batch["consumption_ratio"].to(device)
            pred = model(before, after, area_ratio)
            test_preds.append(pred.cpu())
            test_targets.append(target.cpu())
            test_wb.append(batch["weight_before"].float())

    preds_t = torch.cat(test_preds)
    targets_t = torch.cat(test_targets)
    wb_t = torch.cat(test_wb)
    test_mae = compute_mae(preds_t, targets_t)
    test_gram_mae = compute_mae(preds_t * wb_t, targets_t * wb_t)
    test_gram_rmse = compute_rmse(preds_t * wb_t, targets_t * wb_t)
    logger.info(
        f"Fold {fold_n} | Test MAE: {test_mae:.4f} | "
        f"Gram MAE: {test_gram_mae:.2f}g | Gram RMSE: {test_gram_rmse:.2f}g (held-out)"
    )

    return best_val_mae, test_mae, test_gram_mae, test_gram_rmse, pd.DataFrame(log_rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train dual-stream food waste estimator")
    parser.add_argument("--folds", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--frozen_epochs", type=int, default=10)
    parser.add_argument(
        "--pretrained",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use ImageNet-pretrained EfficientNet-B0 backbone, or random init (default).",
    )
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

    df, norm_params = load_metadata(
        os.path.join(data_dir, "data_original.xlsx"),
        before_dir=before_dir,
        after_dir=after_dir,
        save_dir=results_dir,
    )

    train_transform = get_transforms("train")
    val_transform = get_transforms("val")

    # Outer 10-fold: each fold yields a held-out test set (1/10) and trainval (9/10)
    outer_gkf = GroupKFold(n_splits=args.folds)
    # Inner 5-fold on trainval: first split gives val (20% of trainval ~ 2/10 total)
    inner_gkf = GroupKFold(n_splits=5)

    groups = df["group"].values
    fold_indices: dict = {}
    fold_val_maes: list[float] = []
    fold_test_maes: list[float] = []
    fold_test_gram_maes: list[float] = []
    fold_test_gram_rmses: list[float] = []
    all_fold_logs: list[pd.DataFrame] = []

    pin_memory = device.type == "cuda"

    for fold_n, (trainval_idx, test_idx) in enumerate(
        outer_gkf.split(np.arange(len(df)), groups=groups), start=1
    ):
        logger.info(f"{'=' * 60}")
        logger.info(f"Fold {fold_n}/{args.folds}")
        logger.info(f"{'=' * 60}")

        set_seed(args.seed + fold_n)

        trainval_df = df.iloc[trainval_idx]
        test_df = df.iloc[test_idx]
        tv_groups = trainval_df["group"].values

        # Take only the first inner split to define val vs train
        inner_train_local, val_local = next(
            inner_gkf.split(np.arange(len(trainval_df)), groups=tv_groups)
        )
        train_df = trainval_df.iloc[inner_train_local]
        val_df = trainval_df.iloc[val_local]

        fold_indices[f"fold_{fold_n}"] = {
            "train": train_df.index.tolist(),
            "val": val_df.index.tolist(),
            "test": test_df.index.tolist(),
        }
        logger.info(
            f"Split sizes -- train: {len(train_df)}, val: {len(val_df)}, test: {len(test_df)}"
        )

        train_ds = FoodWasteDataset(train_df, before_dir, after_dir, train_transform)
        val_ds = FoodWasteDataset(val_df, before_dir, after_dir, val_transform)
        test_ds = FoodWasteDataset(test_df, before_dir, after_dir, val_transform)

        # Inverse-frequency sample weighting for imbalanced ratio distribution
        sample_weights = compute_class_weights(train_df)
        sampler = WeightedRandomSampler(
            weights=sample_weights, num_samples=len(sample_weights), replacement=True
        )

        train_loader = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            sampler=sampler,
            num_workers=args.num_workers,
            pin_memory=pin_memory,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=pin_memory,
        )
        test_loader = DataLoader(
            test_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=pin_memory,
        )

        model = DualStreamEfficientNet(pretrained=args.pretrained).to(device)
        effective_frozen_epochs = args.frozen_epochs
        if args.pretrained:
            model.freeze_backbone()
        else:
            effective_frozen_epochs = 0
            logger.info(
                "Training from scratch (--no-pretrained): skipping frozen warm-up, "
                "backbone trains from epoch 1 (frozen_epochs forced to 0)."
            )

        head_params = list(model.fusion.parameters()) + list(model.regression_head.parameters())
        backbone_params = list(model.backbone.parameters())
        optimizer = torch.optim.Adam(
            [{"params": head_params}, {"params": backbone_params}],
            lr=args.lr,
        )

        best_val_mae, test_mae, test_gram_mae, test_gram_rmse, log_df = train_fold(
            model,
            train_loader,
            val_loader,
            test_loader,
            optimizer,
            device,
            args.epochs,
            args.patience,
            effective_frozen_epochs,
            fold_n,
            checkpoint_dir,
            norm_params,
        )
        fold_val_maes.append(best_val_mae)
        fold_test_maes.append(test_mae)
        fold_test_gram_maes.append(test_gram_mae)
        fold_test_gram_rmses.append(test_gram_rmse)
        log_df["fold"] = fold_n
        all_fold_logs.append(log_df)
        log_df.to_csv(os.path.join(results_dir, f"fold_{fold_n}_log.csv"), index=False)
        logger.info(f"Fold {fold_n} -- val MAE: {best_val_mae:.4f} | test MAE: {test_mae:.4f}")

        # Partial save so results survive a mid-training crash
        with open(os.path.join(results_dir, "summary.json"), "w") as f:
            json.dump({
                "fold_val_maes": fold_val_maes,
                "fold_test_maes": fold_test_maes,
                "fold_test_gram_maes": fold_test_gram_maes,
                "fold_test_gram_rmses": fold_test_gram_rmses,
            }, f, indent=2)

    with open(os.path.join(results_dir, "fold_indices.json"), "w") as f:
        json.dump(fold_indices, f, indent=2)

    pd.concat(all_fold_logs, ignore_index=True).to_csv(
        os.path.join(results_dir, "all_folds_log.csv"), index=False
    )

    summary = {
        "fold_val_maes": fold_val_maes,
        "fold_test_maes": fold_test_maes,
        "fold_test_gram_maes": fold_test_gram_maes,
        "fold_test_gram_rmses": fold_test_gram_rmses,
        "mean_val_mae": float(np.mean(fold_val_maes)),
        "mean_test_mae": float(np.mean(fold_test_maes)),
        "mean_test_gram_mae": float(np.mean(fold_test_gram_maes)),
        "std_test_mae": float(np.std(fold_test_maes)),
        "human_baseline_mae": 0.0926,
        "beats_baseline": float(np.mean(fold_test_maes)) < 0.0926,
    }
    with open(os.path.join(results_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(
        f"Final Mean Test MAE: {summary['mean_test_mae']:.4f} +/- {summary['std_test_mae']:.4f}"
    )
    logger.info(f"Beats human baseline (0.0926): {summary['beats_baseline']}")


if __name__ == "__main__":
    main()
