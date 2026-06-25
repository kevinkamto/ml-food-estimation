import os
import json
import pickle
import random as _random

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from sklearn.preprocessing import LabelEncoder


def load_metadata(xlsx_path, save_dir='.'):
    df = pd.read_excel(xlsx_path)

    df['Weight Leftover (g)'] = df['Weight Before Eaten (g)'] - df['Weight After Eaten (g)']
    assert (df['Weight Leftover (g)'] >= 0).all(), "Negative leftover weights found in metadata"

    max_weight = float(df['Weight Before Eaten (g)'].max())
    df['leftover_normalized'] = df['Weight Leftover (g)'] / max_weight

    le = LabelEncoder()
    df['category_id'] = le.fit_transform(df['Name of the food'])

    os.makedirs(save_dir, exist_ok=True)
    norm_params = {'max_weight': max_weight}
    with open(os.path.join(save_dir, 'normalization_params.json'), 'w') as f:
        json.dump(norm_params, f, indent=2)
    with open(os.path.join(save_dir, 'label_encoder.pkl'), 'wb') as f:
        pickle.dump(le, f)

    return df, norm_params, le


def _seg_filename(raw_filename):
    # Segmented files are named {category}_{raw_filename}, e.g.
    # raw: 001_001_DSC_0059_bef.JPG -> segmented: 001_001_001_DSC_0059_bef.JPG
    cat = raw_filename[:3]
    return f"{cat}_{raw_filename}"


def find_image(root_dir, filename):
    for dirpath, _, files in os.walk(root_dir):
        if filename in files:
            return os.path.join(dirpath, filename)
    raise FileNotFoundError(f"Image '{filename}' not found under {root_dir}")


def get_transforms(mode='train'):
    if mode == 'train':
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=1/7),
            transforms.RandomVerticalFlip(p=1/7),
            transforms.RandomApply([transforms.RandomRotation(degrees=15)], p=1/7),
            transforms.RandomApply([
                transforms.Compose([transforms.Pad(20), transforms.Resize((224, 224))])
            ], p=1/7),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=1/7),
            transforms.RandomAdjustSharpness(sharpness_factor=2, p=1/7),
            transforms.RandomAutocontrast(p=1/7),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


def _apply_pair_transform(transform, img1, img2):
    seed = torch.randint(0, 2**32, (1,)).item()
    _random.seed(seed)
    torch.manual_seed(seed)
    t1 = transform(img1)
    _random.seed(seed)
    torch.manual_seed(seed)
    t2 = transform(img2)
    return t1, t2


class FoodWasteDataset(Dataset):
    def __init__(self, df, before_dir, after_dir, transform=None):
        self.df = df.reset_index(drop=True)
        self.before_dir = before_dir
        self.after_dir = after_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        before_path = find_image(self.before_dir, _seg_filename(row['Image Before Eaten']))
        after_path = find_image(self.after_dir, _seg_filename(row['Image After Eaten']))

        before = Image.open(before_path).convert('RGB')
        after = Image.open(after_path).convert('RGB')

        if self.transform:
            before, after = _apply_pair_transform(self.transform, before, after)

        return {
            'before': before,
            'after': after,
            'leftover_norm': torch.tensor(row['leftover_normalized'], dtype=torch.float32),
            'category': torch.tensor(row['category_id'], dtype=torch.long),
            'leftover_g': float(row['Weight Leftover (g)']),
            'food_name': str(row['Name of the food'])
        }
