"""
Diabetic5 Dataset module.

This module provides a dataset class, Diabetic5, which loads the Diabetic Retinopathy Detection
dataset and prunes it to reserve 200 images per class for evaluation. 

Example:
    To use this dataset, simply create an instance of the Diabetic5 class:
    >>> from msa_toolbox.datasets import Diabetic5
    >>> train_dataset = Diabetic5(train=True)

Note:
    This dataset requires downloading the Diabetic Retinopathy Detection dataset from 
    https://www.kaggle.com/c/diabetic-retinopathy-detection
    The data should be placed in the `DATASET_ROOT/diabetic_retinopathy/training_imgs` directory.
"""

import os
from collections import defaultdict as dd
import numpy as np
from torchvision.datasets.folder import ImageFolder
from ... import config as cfg


class Diabetic5(ImageFolder):
    """
    Diabetic5 Dataset class, a subclass of ImageFolder.
    This class loads the Diabetic Retinopathy Detection dataset and prunes it to reserve
        200 images per class for evaluation.
    """

    def __init__(self, root: str, train=True, transform=None, target_transform=None):
        """
        Initialize the Diabetic5 dataset.

        Args:
            train (bool): True if training set, False if testing set
            transform (callable, optional): A function/transform that takes in an 
                PIL image and returns a transformed version. E.g, ``transforms.RandomCrop``
            target_transform (callable, optional): A function/transform that takes in the 
                target and transforms it.
        """
        root = os.path.join(root, 'diabetic_retinopathy')
        if not os.path.exists(root):
            print(f'A dataset not found at {root}. Please download it from ')
            raise ValueError(f"Dataset not found at {root}. Please download it from "
                            f"https://www.kaggle.com/c/diabetic-retinopathy-detection")


        # Initialize ImageFolder
        super().__init__(root=os.path.join(root, 'training_imgs'), transform=transform,
                        target_transform=target_transform)
        self.root = root
        self.ntest = 200   # Reserve ntest images per class for evaluation
        self.partition_to_idxs = self.__get_partition_to_idxs()
        self.pruned_idxs = self.partition_to_idxs['train' if train else 'test']
        self.samples = [self.samples[i] for i in self.pruned_idxs]
        self.imgs = self.samples
        # print(f"=> done loading {self.__class__.__name__} " + f"{'train' if train else 'test'} set "f"with {len(self.samples)} examples")

    def __get_partition_to_idxs(self):
        """
        Create mapping: classidx -> idx and partition the data into train and test sets.
        """
        partition_to_idxs = {
            'train': [],
            'test': []
        }

        # Use this random seed to make partition consistent
        prev_state = np.random.get_state()
        np.random.seed(cfg.DS_SEED)

        # Create mapping: classidx -> idx
        classidx_to_idxs = dd(list)
        for idx, sample in enumerate(self.samples):
            classidx = sample[1]
            classidx_to_idxs[classidx].append(idx)

        # Shuffle classidx_to_idx
        for classidx, idxs in classidx_to_idxs.items():
            np.random.shuffle(idxs)

        for classidx, idxs in classidx_to_idxs.items():
            # A constant no. kept aside for evaluation
            partition_to_idxs['test'] += idxs[:self.ntest]
            # Train on remaining
            partition_to_idxs['train'] += idxs[self.ntest:]

        # Revert randomness to original state
        np.random.set_state(prev_state)
        return partition_to_idxs
    
    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return sample, target, index
