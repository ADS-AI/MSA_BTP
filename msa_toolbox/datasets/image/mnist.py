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
import numpy as np
import torch
from torchvision.datasets import MNIST as Old_MNIST
from torchvision.datasets import EMNIST as Old_EMNIST
from torchvision.datasets import FashionMNIST as Old_FashionMNIST
from torchvision.datasets import KMNIST as Old_KMNIST
from ... import config as cfg
from PIL import Image
from typing import Any, Callable, Optional, Tuple


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
    
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], int(self.targets[index])
        # # doing this so that it is consistent with all other datasets to return a PIL Image
        img = Image.fromarray(img.numpy())
        img = img.convert('RGB')
        img = np.array(img)
        img = Image.fromarray(img.astype('uint8'), 'RGB')
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target, index



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
        
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], int(self.targets[index])
        # # doing this so that it is consistent with all other datasets to return a PIL Image
        img = Image.fromarray(img.numpy())
        img = img.convert('RGB')
        img = np.array(img)
        img = Image.fromarray(img.astype('uint8'), 'RGB')
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target, index


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
    
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], int(self.targets[index])
        # # doing this so that it is consistent with all other datasets to return a PIL Image
        img = Image.fromarray(img.numpy())
        img = img.convert('RGB')
        img = np.array(img)
        img = Image.fromarray(img.astype('uint8'), 'RGB')
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target, index


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
        
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], int(self.targets[index])
        # # doing this so that it is consistent with all other datasets to return a PIL Image
        img = Image.fromarray(img.numpy())
        img = img.convert('RGB')
        img = np.array(img)
        img = Image.fromarray(img.astype('uint8'), 'RGB')
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target, index


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
        
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], int(self.targets[index])
        # # doing this so that it is consistent with all other datasets to return a PIL Image
        img = Image.fromarray(img.numpy())
        img = img.convert('RGB')
        img = np.array(img)
        img = Image.fromarray(img.astype('uint8'), 'RGB')
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target, index
