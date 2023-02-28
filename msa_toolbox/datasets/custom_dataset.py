"""
custom_dataset.py

A module that contains the implementation of a custom dataset class that inherits from 
PyTorch's ImageFolder class. This class is used to load the image dataset from a directory
"""

import os
from torchvision.datasets.folder import ImageFolder

class CustomDataset(ImageFolder):
    """
    A custom dataset class that inherits from PyTorch's ImageFolder class.

    Args:
        root_dir (string): Root directory path of the dataset.
        transform (callable, optional): Optional image transform to be applied.
        target_transform (callable, optional): Optional transform to be applied to the label.
    Raises:
        ValueError: If the root directory path of the dataset does not exist.
    """
    def __init__(self, root_dir: str, transform=None, target_transform=None):
        if not os.path.exists(root_dir):
            raise ValueError(f"Dataset not found at {root_dir}. Please enter a valid path with the dataset.")
        super().__init__(root_dir, transform, target_transform)


    def __getitem__(self, index: int):
        """
        Overrides the default __getitem__ method from ImageFolder class.
        Args:
            index (int): Index of the data sample to be retrieved.
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img_path, target = self.samples[index]
        img = self.loader(img_path)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target
