"""
This module provides a dataset class representing the Street View House Numbers (SVHN) dataset.
"""

import os
import numpy as np
from torchvision.datasets import SVHN as Old_SVHN
from .. import config as cfg


class SVHN(Old_SVHN):
    
    """
    A dataset class representing the Street View House Numbers (SVHN) dataset.

    Attributes:
        classes (numpy.ndarray): An array containing the possible classes of the SVHN dataset.

    Methods:
        __init__(self, train=True, transform=None, target_transform=None, download=False):
            Initializes the SVHN dataset object.
        get_image(self, index):
            Returns the image data at the given index.
    """
    
    def __init__(self, train=True, transform=None, target_transform=None, download=True):
        """
        Initializes the SVHN dataset object.

        Args:
            train (bool or str): If True, returns the training set, otherwise the test set. 
                If str, can be one of {'train', 'test', 'extra'}.
            transform (callable, optional): A transform that takes in a PIL image and 
                returns a transformed version.
            target_transform (callable, optional): A transform that takes in the target and transforms it.
            download (bool): If True, downloads the dataset from the internet and places it in root directory.

        Returns:
            None
        """
        root = os.path.join(cfg.DATASET_ROOT, 'svhn')

        if isinstance(train, bool):
            split = 'train' if train else 'test'
        else:
            split = train

        super().__init__(root, split, transform, target_transform, download)
        self.classes = np.arange(10)

    def get_image(self, index):
        """
        Returns the image at the specified index.

        Args:
            index (int): The index of the image to return.

        Returns:
            numpy.ndarray: The image data at the specified index.
        """
        return self.data[index]
