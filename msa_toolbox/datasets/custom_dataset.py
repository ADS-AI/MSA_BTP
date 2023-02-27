import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision.datasets.folder import ImageFolder, default_loader
import msa_toolbox.config as cfg
from torchvision.transforms import transforms


class CustomDataset(ImageFolder):
    """
    A custom dataset class that inherits from PyTorch's ImageFolder class.

    Args:
        root_dir (string): Root directory path of the dataset.
        transform (callable, optional): Optional image transform to be applied.
        target_transform (callable, optional): Optional transform to be applied to the label.
    """
    def __init__(self, root_dir, transform=None, target_transform=None):
        if not os.path.exists(root_dir):
            raise ValueError('Dataset not found at {}. Please enter a valid path with the dataset.'.format(root_dir))
        super(CustomDataset, self).__init__(root_dir, transform, target_transform)
        
    def __getitem__(self, index):
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
