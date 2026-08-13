import os

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class RestorationDataset(Dataset):
    """
    Expects data_dir/clean/*.png and data_dir/degraded/*.png with matching filenames
    (as produced by data/prepare_dataset.py).
    """

    def __init__(self, data_dir, img_size=256):
        self.clean_dir = os.path.join(data_dir, "clean")
        self.degraded_dir = os.path.join(data_dir, "degraded")
        self.files = sorted(os.listdir(self.clean_dir))
        self.img_size = img_size

    def __len__(self):
        return len(self.files)

    def _load(self, path):
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.img_size, self.img_size))
        img = img.astype(np.float32) / 255.0
        return torch.from_numpy(img).permute(2, 0, 1)

    def __getitem__(self, idx):
        fname = self.files[idx]
        clean = self._load(os.path.join(self.clean_dir, fname))
        degraded = self._load(os.path.join(self.degraded_dir, fname))
        return degraded, clean
