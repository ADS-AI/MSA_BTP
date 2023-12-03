"""
This module provides a class Indoor67 that represents the Indoor67 dataset in PyTorch.
The dataset contains 67 indoor categories with a total of 15620 images.

Example usage:
dataset = Indoor67(train=True, transform=train_transform)

Note that the dataset is not downloaded automatically. 
It must be downloaded manually from the following link: http://web.mit.edu/torralba/www/indoor.html
"""

import os
from torchvision.datasets.folder import ImageFolder

from ... import config as cfg

class Indoor67(ImageFolder):
    """ The Indoor67 dataset class """

    def __init__(self, root: str, train=True, transform=None, target_transform=None):
        """
        Initializes the Indoor67 dataset.

        Args:
            train (bool): If True, loads the training partition, otherwise loads the test partition.
            transform (callable, optional): A function/transform that takes in an PIL image and returns a transformed version. Default: None.
            target_transform (callable, optional): A function/transform that takes in the target and transforms it. Default: None.
        """
        root = os.path.join(root, 'indoor')
        if not os.path.exists(root):
            raise ValueError(f"Dataset not found at {root}. Please download it from http://web.mit.edu/torralba/www/indoor.html.")

        super().__init__(root=os.path.join(root, 'Images'), transform=transform, target_transform=target_transform)

        self.root = root
        self.partition_to_idxs = self.__get_partition_to_idxs()
        self.pruned_idxs = self.partition_to_idxs['train' if train else 'test']
        self.samples = [self.samples[i] for i in self.pruned_idxs]
        self.imgs = self.samples
        # print(f"=> done loading {self.__class__.__name__} ({'train' if train else 'test'}) with {len(self.samples)} examples")


    def __get_partition_to_idxs(self):
        """
        Returns:
            A dictionary mapping partition names to lists of indices. 
        """
        partition_to_idxs = {
            'train': [],    
            'test': []
        }

        test_images = set()
        with open(os.path.join(self.root, 'TestImages.txt')) as f:
            for line in f:
                test_images.add(line.strip())

        for idx, (filepath, _) in enumerate(self.samples):
            filepath = filepath.replace(os.path.join(self.root, 'Images') + '/', '')
            if filepath in test_images:
                partition_to_idxs['test'].append(idx)
            else:
                partition_to_idxs['train'].append(idx)

        return partition_to_idxs
