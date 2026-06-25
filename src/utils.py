import random

import numpy as np
import torch


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_mae(preds, targets):
    return float(torch.mean(torch.abs(preds - targets)))


def compute_rmse(preds, targets):
    return float(torch.sqrt(torch.mean((preds - targets) ** 2)))


