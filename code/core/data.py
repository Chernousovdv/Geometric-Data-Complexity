"""
Universal dataset class with train/test split.

Handles both synthetic data (numpy arrays) and image data (files on disk).
Provides torch Dataset/DataLoader integration for model training.
"""

import numpy as np
from pathlib import Path

try:
    import torch
    from torch.utils.data import Dataset as TorchDataset, DataLoader
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


class Dataset:
    """
    Universal dataset with train/test split.

    Data is always a numpy array of shape (N, *sample_shape).
    For images: (N, C, H, W) with values in [0, 1].
    For point data: (N, D).
    """

    def __init__(self, train_data, test_data, name=""):
        self.train = np.asarray(train_data, dtype=np.float32)
        self.test = np.asarray(test_data, dtype=np.float32)
        self.name = name
        self.shape = self.train.shape[1:]

    def get(self, split='train'):
        """Return all data for a split as numpy array."""
        if split == 'train':
            return self.train
        elif split == 'test':
            return self.test
        else:
            raise ValueError(f"Unknown split: {split}")

    def __len__(self):
        return len(self.train)

    def __getitem__(self, idx):
        return self.train[idx]

    # --- Torch integration ---

    def torch_dataset(self, split='train'):
        """Return a torch Dataset wrapping this split."""
        if not HAS_TORCH:
            raise ImportError("PyTorch is required for torch_dataset()")
        return _TorchWrapper(self.get(split))

    def torch_loader(self, split='train', batch_size=64, shuffle=True, **kwargs):
        """Return a DataLoader for this split."""
        ds = self.torch_dataset(split)
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, **kwargs)

    # --- Construction ---

    @classmethod
    def from_sampler(cls, sampler_fn, n_train, n_test, name=""):
        """
        Create dataset by calling sampler_fn(n) -> (n, *shape) array.

        Args:
            sampler_fn: callable, takes int n, returns (n, *shape) numpy array
            n_train: number of training samples
            n_test: number of test samples
            name: dataset name
        """
        train_data = sampler_fn(n_train)
        test_data = sampler_fn(n_test)
        return cls(train_data, test_data, name=name)

    @classmethod
    def from_folder(cls, path, name=""):
        """
        Load from path/train/{0,1,2,...}.png and path/test/{0,1,2,...}.png.

        Images are loaded as (N, C, H, W) numpy arrays with values in [0, 1].
        Grayscale images get C=1.
        """
        from PIL import Image

        path = Path(path)
        if not name:
            name = path.name

        train_data = _load_numbered_images(path / "train")
        test_data = _load_numbered_images(path / "test")
        return cls(train_data, test_data, name=name)

    # --- Persistence ---

    def save_images(self, path):
        """
        Save as numbered images: path/train/0.png, path/test/0.png, etc.

        Data must be image-shaped: (N, C, H, W) or (N, H, W).
        """
        from PIL import Image

        path = Path(path)
        for split in ['train', 'test']:
            data = self.get(split)
            split_dir = path / split
            split_dir.mkdir(parents=True, exist_ok=True)

            for i, sample in enumerate(data):
                img = _array_to_image(sample)
                img.save(split_dir / f"{i}.png")

    def save_numpy(self, path):
        """Save as path/train.npy and path/test.npy."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        np.save(path / "train.npy", self.train)
        np.save(path / "test.npy", self.test)

    @classmethod
    def load_numpy(cls, path, name=""):
        """Load from path/train.npy and path/test.npy."""
        path = Path(path)
        if not name:
            name = path.name
        train_data = np.load(path / "train.npy")
        test_data = np.load(path / "test.npy")
        return cls(train_data, test_data, name=name)

    def __repr__(self):
        return (f"Dataset(name='{self.name}', "
                f"train={self.train.shape}, test={self.test.shape})")


# --- Internal helpers ---

def _load_numbered_images(folder):
    """Load 0.png, 1.png, ... from folder. Returns (N, C, H, W) array in [0, 1]."""
    from PIL import Image

    folder = Path(folder)
    files = sorted(folder.glob("*.png"), key=lambda f: int(f.stem))
    if not files:
        raise FileNotFoundError(f"No .png files found in {folder}")

    images = []
    for f in files:
        img = Image.open(f)
        arr = np.array(img, dtype=np.float32) / 255.0
        images.append(arr)

    images = np.stack(images)

    # Ensure (N, C, H, W) format
    if images.ndim == 3:
        # (N, H, W) grayscale -> (N, 1, H, W)
        images = images[:, np.newaxis, :, :]
    elif images.ndim == 4 and images.shape[-1] in (1, 3, 4):
        # (N, H, W, C) -> (N, C, H, W)
        images = np.transpose(images, (0, 3, 1, 2))

    return images


def _array_to_image(arr):
    """Convert a single sample array to PIL Image. Input in [0, 1]."""
    from PIL import Image

    arr = np.clip(arr * 255, 0, 255).astype(np.uint8)

    if arr.ndim == 2:
        # (H, W) grayscale
        return Image.fromarray(arr, mode='L')
    elif arr.ndim == 3:
        if arr.shape[0] == 1:
            # (1, H, W) -> (H, W) grayscale
            return Image.fromarray(arr[0], mode='L')
        elif arr.shape[0] == 3:
            # (C, H, W) -> (H, W, C) RGB
            return Image.fromarray(np.transpose(arr, (1, 2, 0)), mode='RGB')
        elif arr.shape[0] == 4:
            # (C, H, W) -> (H, W, C) RGBA
            return Image.fromarray(np.transpose(arr, (1, 2, 0)), mode='RGBA')

    raise ValueError(f"Cannot convert array of shape {arr.shape} to image")


if HAS_TORCH:
    class _TorchWrapper(TorchDataset):
        """Thin wrapper: numpy array -> torch Dataset."""
        def __init__(self, data):
            self.data = torch.from_numpy(data).float()
        def __len__(self):
            return len(self.data)
        def __getitem__(self, idx):
            return self.data[idx]
