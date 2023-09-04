"""
This module provides a PyTorch dataset class for the TinyImageNet200 dataset.
"""
import os
from torchvision.datasets import ImageFolder
from .. import config as cfg

from .. import config as cfg


class TinyImageNet200(ImageFolder):
    '''A Dataset class for the TinyImageNet200 dataset.'''

    def __init__(self, root: str, train=True, transform=None, target_transform=None):
        """
        Initializes the dataset and loads the data.

        Args:
            train (bool): If True, loads the training set, else loads the validation set.
            transform (callable, optional): A function/transform that takes in a PIL image and returns a transformed version.
            target_transform (callable, optional): A function/transform that takes in the target and transforms it.
        """
        root = os.path.join(root, 'tiny-imagenet-200')
        if not os.path.exists(root):
            raise ValueError(f"Dataset not found at {root}. Please download it from http://cs231n.stanford.edu/tiny-imagenet-200.zip")

        # Initialize ImageFolder
        _root = os.path.join(root, 'train' if train else 'val')
        super().__init__(root=_root, transform=transform,
                         target_transform=target_transform)
        self.root = root

        # print(f"=> done loading {self.__class__.__name__} ({'train' if train else 'test'}) with {len(self.samples)} examples")
        self._load_meta()


    def _load_meta(self):
        """
        Replace class names (synsets) with more descriptive labels
        """
        synset_to_desc = {}
        fpath = os.path.join(self.root, 'words.txt')
        with open(fpath, 'r') as rf:
            for line in rf:
                synset, desc = line.strip().split(maxsplit=1)
                synset_to_desc[synset] = desc

        # Replace
        for i in range(len(self.classes)):
            self.classes[i] = synset_to_desc[self.classes[i]]
        self.class_to_idx = {self.classes[i]: i for i in range(len(self.classes))}
