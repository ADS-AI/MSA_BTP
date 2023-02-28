"""
This module contains classes for loading MNIST, KMNIST, EMNIST, EMNISTLetters, and FashionMNIST 
datasets using torchvision.

Classes:
MNIST: Loads the MNIST dataset.
KMNIST: Loads the Kuzushiji-MNIST dataset.
EMNIST: Loads the EMNIST dataset.
EMNISTLetters: Loads the EMNIST letters dataset.
FashionMNIST: Loads the Fashion-MNIST dataset.
"""

import os
from torchvision.datasets import MNIST as Old_MNIST
from torchvision.datasets import EMNIST as Old_EMNIST
from torchvision.datasets import FashionMNIST as Old_FashionMNIST
from torchvision.datasets import KMNIST as Old_KMNIST
import msa_toolbox.config as cfg


class MNIST(Old_MNIST):
    ''' MNIST: A subclass of the torchvision.datasets.MNIST class.'''
    def __init__(self, train=True, transform=None, target_transform=None, download=True):
        '''
        Args:
        - train (bool): Whether to load the training or test set.
        - transform: Optional transform to be applied on a sample.
        - target_transform: Optional transform to be applied on a label.
        - download (bool): Whether to download the dataset if it is not found in the root directory.
        '''
        root = os.path.join(cfg.DATASET_ROOT, 'mnist')
        super().__init__(root, train, transform, target_transform, download)


class KMNIST(Old_KMNIST):
    '''' KMNIST: A subclass of the torchvision.datasets.KMNIST class.'''
    def __init__(self, train=True, transform=None, target_transform=None, download=True):
        '''
        Args:
        - train (bool): Whether to load the training or test set.
        - transform: Optional transform to be applied on a sample.
        - target_transform: Optional transform to be applied on a label.
        - download (bool): Whether to download the dataset if it is not found in the root directory.
        '''
        root = os.path.join(cfg.DATASET_ROOT, 'kmnist')
        super().__init__(root, train, transform, target_transform, download)


class EMNIST(Old_EMNIST):
    ''' EMNIST: A subclass of the torchvision.datasets.EMNIST class.'''
    def __init__(self, **kwargs):
        '''
        Args:
        - train (bool): Whether to load the training or test set.
        - transform: Optional transform to be applied on a sample.
        - target_transform: Optional transform to be applied on a label.
        - download (bool): Whether to download the dataset if it is not found in the root directory.
        '''
        root = os.path.join(cfg.DATASET_ROOT, 'emnist')
        super().__init__(root, split='balanced', download=True, **kwargs)
        self.data = self.data.permute(0, 2, 1)


class EMNISTLetters(Old_EMNIST):
    ''' EMNISTLetters: A subclass of the torchvision.datasets.EMNIST class.'''
    def __init__(self, **kwargs):
        '''
        Args:
        - train (bool): Whether to load the training or test set.
        - transform: Optional transform to be applied on a sample.
        - target_transform: Optional transform to be applied on a label.
        - download (bool): Whether to download the dataset if it is not found in the root directory.
        '''
        root = os.path.join(cfg.DATASET_ROOT, 'emnist')
        super().__init__(root, split='letters', download=True, **kwargs)
        self.data = self.data.permute(0, 2, 1)


class FashionMNIST(Old_FashionMNIST):
    ''' FashionMNIST: A subclass of the torchvision.datasets.FashionMNIST class.'''
    def __init__(self, train=True, transform=None, target_transform=None, download=True):
        '''
        Args:
        - train (bool): Whether to load the training or test set.
        - transform: Optional transform to be applied on a sample.
        - target_transform: Optional transform to be applied on a label.
        - download (bool): Whether to download the dataset if it is not found in the root directory.
        '''
        root = os.path.join(cfg.DATASET_ROOT, 'mnist_fashion')
        super().__init__(root, train, transform, target_transform, download)
