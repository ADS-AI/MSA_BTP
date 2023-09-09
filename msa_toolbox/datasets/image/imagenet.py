"""
This module defines the ImageNet1k dataset class, 
Example usage:
    from msa_toolbox.datasets import ImageNet1k
    train_dataset = ImageNet1k(train=True)
    test_dataset = ImageNet1k(train=False)

Note that the dataset is not downloaded automatically and must be downloaded manually
from http://image-net.org/download-images.
"""

import os
import numpy as np
from torchvision.datasets import ImageFolder
from ... import config as cfg

class ImageNet1k(ImageFolder):
    """
    Class level docstring describing the ImageNet1k class.
    Attributes:
        test_frac (float): The fraction of data to use for testing.
    """
    test_frac = 0.2

    def __init__(self, root: str, train=True, transform=None, target_transform=None):
        """
        Initializes the ImageNet1k dataset.
        Args:
            train (bool, optional): If True, loads the training dataset. If False, loads the validation dataset. Default is True.
            transform (callable, optional): A function/transform that takes in a PIL image and returns a transformed version. Default is None.
            target_transform (callable, optional): A function/transform that takes in the target and transforms it. Default is None.
        Raises:
            ValueError: If the dataset is not found at `cfg.DATASET_ROOT`.
        Returns:
            None
        """
        # root = os.path.join(root, 'ILSVRC2012')
        if not os.path.exists(root):
            raise ValueError(f'Dataset not found at {root}. Please download it from http://image-net.org/download-images')

        # Initialize ImageFolder
        # super().__init__(root=os.path.join(root, 'training_imgs'), transform=transform,target_transform=target_transform)
        super().__init__(root=root, transform=transform,target_transform=target_transform)
        self.root = root

        # self.partition_to_idxs = self.__get_partition_to_idxs()
        # self.pruned_idxs = self.partition_to_idxs['train' if train else 'test']

        # Prune (self.imgs, self.samples to only include examples from the required train/test partition
        # self.samples = [self.samples[i] for i in self.pruned_idxs]
        # print(f"=> done loading {self.__class__.__name__} ({'train' if train else 'test'}) with {len(self.samples)} examples")


    def __get_partition_to_idxs(self):
        """
        Returns a dictionary mapping dataset partitions to their corresponding indices.

        Returns:
            dict: A dictionary mapping dataset partitions to their corresponding indices.
        """
        partition_to_idxs = {
            'train': [],
            'test': []
        }

        # Note: we perform a 80-20 split of imagenet training
        # While this is not necessary, this is to simply to keep it consistent with the paper
        prev_state = np.random.get_state()
        np.random.seed(cfg.DS_SEED)

        idxs = np.arange(len(self.samples))
        n_test = int(self.test_frac * len(idxs))
        test_idxs = np.random.choice(idxs, replace=False, size=n_test).tolist()
        train_idxs = list(set(idxs) - set(test_idxs))

        partition_to_idxs['train'] = train_idxs
        partition_to_idxs['test'] = test_idxs

        np.random.set_state(prev_state)
        return partition_to_idxs
