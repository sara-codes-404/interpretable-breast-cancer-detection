"""
Dataset utilities for the PatchCamelyon (PCam) breast cancer dataset.

This module provides two dataset implementations:

1. PCamDataset
   Loads the original HDF5 files.

2. HFPCamDataset
   Loads the HuggingFace version.

Both datasets return PyTorch tensors compatible with torchvision models.
"""

from typing import Callable, Optional, Tuple

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class PCamDataset(Dataset):
    """
Dataset class for the PatchCamelyon (PCam) dataset.

This class loads images and labels from HDF5 (.h5) files and prepares them
for use with a PyTorch DataLoader.

The HDF5 files are opened lazily instead of being opened in **init**.
This helps avoid problems when using the dataset with multiple DataLoader
workers.

Attributes:
x_path: Path to the HDF5 file containing the images.
y_path: Path to the HDF5 file containing the labels.
transform: Optional transformations applied to each image.
_x_file: HDF5 file handle for the images.
_y_file: HDF5 file handle for the labels.
_length: Number of samples in the dataset.
    """


    def __init__(
        self,
        x_path: str,
        y_path: str,
        transform: Optional[Callable] = None,
    ) -> None:
        """
Initialize the PCam dataset.

Args:
x_path: Path to the HDF5 file containing the images.
y_path: Path to the HDF5 file containing the labels.
transform: Optional torchvision transformations applied to the images.
If None, ToTensor is used as the default transformation.
        """
        self.x_path = x_path
        self.y_path = y_path
        self.transform = transform
        
        # These variables start as None and are initialized
        # when __getitem__ is called for the first time (lazy loading).
        # This helps avoid problems when using HDF5 files with DataLoader workers.
        self._x_file: Optional[h5py.File] = None
        self._y_file: Optional[h5py.File] = None

        # Get the dataset length here by opening the file briefly
        # just to read its shape, then closing it immediately.
        # This helps avoid multiprocessing problems with HDF5 files.
        with h5py.File(self.x_path, "r") as f:
            self._length = f["x"].shape[0]

    def _lazy_init(self) -> None:
        """Open the h5 files when they are needed."""
        if self._x_file is None:
            self._x_file = h5py.File(self.x_path, "r")
        if self._y_file is None:
            self._y_file = h5py.File(self.y_path, "r")

    def __len__(self) -> int:
        """Return the total number of samples in the dataset."""
        return self._length

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
Return one sample (image, label) by index.

Args:
    idx: Index of the sample to load.

Returns:
    A tuple containing:
        - image: Image tensor with shape (C, H, W) after applying the transform.
        - label: Float32 tensor containing the label (0 or 1).
        """
        self._lazy_init()

        # Read the image as (H, W, C) with uint8 values from 0 to 255.
        image = self._x_file["x"][idx]
        label = self._y_file["y"][idx].reshape(-1)[0]

        image = np.array(image, dtype=np.uint8)

        if self.transform is not None:
            image = self.transform(image)
        else:
            image = transforms.ToTensor()(image)

        label_tensor = torch.tensor(label, dtype=torch.float32)

        return image, label_tensor


class HFPCamDataset(Dataset):
    """
    
Dataset class for PCam data loaded using the Hugging Face datasets library.

Unlike PCamDataset, which reads images from h5 files, this class uses a
Hugging Face Dataset object. Each sample contains:
    - 'image': a PIL image (RGB, 96x96 pixels)
    - 'label': a boolean value (True = tumor, False = healthy)

Attributes:
    hf_dataset: The loaded Hugging Face Dataset.
    transform: Optional transformations applied to each image.
    """
    
    def __init__(self, hf_dataset, transform: Optional[Callable] = None) -> None:
        """
Initialize the dataset.

Args:
    hf_dataset: A Hugging Face Dataset object, for example:
        load_dataset("1aurent/PatchCamelyon", split="train[:5000]")
    transform: Optional torchvision transformations applied to the PIL image.
        If None, only ToTensor is applied.
        """
        self.hf_dataset = hf_dataset
        self.transform = transform

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.hf_dataset)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
Return one sample (image, label) by index.

Args:
    idx: Index of the sample to load.

Returns:
    A tuple containing:
        - image: Image tensor with shape (C, H, W) after applying the transform.
        - label: Float32 tensor containing the label (0.0 or 1.0).
        """
        sample = self.hf_dataset[idx]
        """
        Dataset class for PCam data from Hugging Face. 
        
        The dataset contains 96x96 RGB images and binary labels. 
        The label is 0 for healthy tissue and 1 for tumor tissue.
        
        Args: hf_dataset: Hugging Face dataset containing the images and labels.
        transform: Optional transformations applied to each image.
        """
        image = sample["image"]
        label = sample["label"]  #  True/False

        if image.mode != "RGB":
            image = image.convert("RGB")

        if self.transform is not None:
            image = self.transform(image)
        else:
            image = transforms.ToTensor()(image)
            
        # Convert the boolean label to float32 (0.0 or 1.0) for BCE loss compatibility.
        label_tensor = torch.tensor(float(label), dtype=torch.float32)

        return image, label_tensor


def load_diverse_pcam_subset(
    split: str = "train",
    n_samples: int = 5000,
    n_chunks: int = 10,
    full_split_size: int = 262144,
):
    """
    Load a subset of PCam samples from different parts of the dataset.
    
    Instead of taking only the first n_samples, this function selects 
    small groups of samples from different positions in the dataset.
    This helps the training data include more diverse images. 
    
    Using only the first samples may contain images from a limited
    number of slides. This can make the model learn slide-specific  
    features such as staining or scanner differences and may reduce
    performance on unseen slides. 
    
    By selecting samples from different parts of the dataset, we can
    get more diverse training data without loading the entire dataset. 
    
    Args:
        split: Dataset split to use ('train', 'validation', or 'test').
        n_samples: Total number of samples to load.
        n_chunks: Number of parts used to sample from the dataset. 
        full_split_size: Total number of samples in the selected split.
        
    Returns: A Hugging Face Dataset containing the selected samples.
    
    Example: >>> hf_data = load_diverse_pcam_subset("train", n_samples=5000)
    """
    from datasets import concatenate_datasets, load_dataset

    chunk_size = n_samples // n_chunks
    # Spread the chunks evenly across the dataset.
    stride = full_split_size // n_chunks

    chunks = []
    for i in range(n_chunks):
        start = i * stride
        end = start + chunk_size
        chunk = load_dataset(
            "1aurent/PatchCamelyon", split=f"{split}[{start}:{end}]"
        )
        chunks.append(chunk)

    combined = concatenate_datasets(chunks)
    return combined


def get_default_transforms(train: bool = True) -> transforms.Compose:
    """
Returns the default transformations for training or evaluation.

For training data, simple augmentations such as random rotation and
horizontal/vertical flips are used to help the model handle different
image orientations. For evaluation, only tensor conversion and
normalization are applied.

Args:
    train: If True, training augmentations are applied. If False,
        only tensor conversion and normalization are used.

Returns:
    A torchvision transform pipeline for PCam images.
    """
    # ImageNet normalization values. We use these values because the
    # ResNet-18 model was pretrained on ImageNet.
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]

    if train:
        return transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomRotation(degrees=20),
                transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
            ]
        )
    else:
        return transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
            ]
        )


if __name__ == "__main__":
    # Test with data loaded from Hugging Face.
    from datasets import load_dataset

    print("Loading subset of dataset from Hugging Face...")
    hf_data = load_dataset("1aurent/PatchCamelyon", split="train[:5000]")

    train_dataset = HFPCamDataset(
        hf_dataset=hf_data,
        transform=get_default_transforms(train=True),
    )

    print(f"Number of samples in the training dataset: {len(train_dataset)}")
    img, lbl = train_dataset[0]
    print(f"Image format: {img.shape}, label: {lbl.item()}")
    # --- Alternative method (if the original H5 files are available) ---

    
