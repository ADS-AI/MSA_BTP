import argparse
import os.path as osp
import os
from collections import defaultdict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets.folder import ImageFolder, default_loader
import msa_toolbox.config as cfg


class Caltech256(ImageFolder):
    """
    Subclass of ImageFolder representing the Caltech 256 dataset.

    Attributes:
        ntest (int): The number of examples reserved per class for evaluation.
        partition_to_idxs (dict): A dictionary that maps train/test partition names to a list of indices for examples belonging to that partition.
        pruned_idxs (list): A list of indices for examples that belong to the required train/test partition.
        samples (list): A list of (image path, classidx) tuples for the examples that belong to the required train/test partition.
    
    Methods:
        __init__(self, train=True, transform=None, target_transform=None):
            Constructor method that initializes the Caltech256 instance.
            
            Args:
                train (bool): If True, loads the training partition of the dataset, else loads the testing partition.
                transform (callable, optional): A function/transform that takes in a PIL image and returns a transformed version.
                target_transform (callable, optional): A function/transform that takes in the target and transforms it.
        
        _cleanup(self):
            Removes examples belonging to class "clutter".
            
        get_partition_to_idxs(self):
            Creates a mapping of classidx to idx for train and test partitions.

    """
        
    def __init__(self, train=True, transform=None, target_transform=None):
    
        """
        Initializes the Caltech256 instance by checking if the dataset is available and then initializing ImageFolder 
        with the specified arguments. It reserves 25 examples per class for evaluation and prunes the `imgs` and `samples` 
        to only include examples from the required train/test partition.

        Args:
            train (bool): If True, loads the training partition of the dataset, else loads the testing partition.
            transform (callable, optional): A function/transform that takes in a PIL image and returns a transformed version.
            target_transform (callable, optional): A function/transform that takes in the target and transforms it.
        
        Returns:
            None
        """
        
        root = osp.join(cfg.DATASET_ROOT, '256_ObjectCategories')
        if not osp.exists(root):
            raise ValueError('Dataset not found at {}. Please download it from {}.'.format(
                root, 'http://www.vision.caltech.edu/Image_Datasets/Caltech256/'
            ))

        # Initialize ImageFolder
        super().__init__(root=root, transform=transform, target_transform=target_transform)

        # self._cleanup()
        self.ntest = 25  # Reserve these many examples per class for evaluation
        self.partition_to_idxs = self.get_partition_to_idxs()
        self.pruned_idxs = self.partition_to_idxs['train' if train else 'test']

        # Prune (self.imgs, self.samples to only include examples from the required train/test partition
        self.samples = [self.samples[i] for i in self.pruned_idxs]
        self.imgs = self.samples

        print('=> done loading {} ({}) with {} examples'.format(self.__class__.__name__, 'train' if train else 'test',
                                                                len(self.samples)))


    def _cleanup(self):
        """
        Removes examples belonging to class "clutter".
        
        Returns:
            None
        """
        
        clutter_idx = self.class_to_idx['257.clutter']
        self.samples = [s for s in self.samples if s[1] != clutter_idx]
        del self.class_to_idx['257.clutter']
        self.classes = self.classes[:-1]
        

    def get_partition_to_idxs(self):
        """
        Creates a mapping of classidx to idx for train and test partitions.
        
        Returns:
            partition_to_idxs (dict): A dictionary that maps train/test partition names to a list of indices for examples belonging to that partition.
        """
        
        partition_to_idxs = {
            'train': [],
            'test': []
        }

        # Use this random seed to make partition consistent
        prev_state = np.random.get_state()
        np.random.seed(cfg.DS_SEED)

        # ----------------- Create mapping: classidx -> idx
        classidx_to_idxs = defaultdict(list)
        for idx, s in enumerate(self.samples):
            classidx = s[1]
            classidx_to_idxs[classidx].append(idx)

        # Shuffle classidx_to_idx
        for classidx, idxs in classidx_to_idxs.items():
            np.random.shuffle(idxs)

        for classidx, idxs in classidx_to_idxs.items():
            partition_to_idxs['test'] += idxs[:self.ntest]     # A constant no. kept aside for evaluation
            partition_to_idxs['train'] += idxs[self.ntest:]    # Train on remaining

        # Revert randomness to original state
        np.random.set_state(prev_state)

        return partition_to_idxs

