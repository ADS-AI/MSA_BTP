import sys
import os
import numpy as np
from torchvision.datasets.folder import ImageFolder
from torchvision.datasets import CIFAR10 as Old_CIFAR10
from torchvision.datasets import CIFAR100 as Old_CIFAR100
from torchvision.datasets import SVHN as Old_SVHN
from torchvision.datasets.utils import check_integrity
import pickle
import msa_toolbox.config as cfg


class CIFAR10(Old_CIFAR10):
    """ 
    CIFAR10 dataset class, subclass of Old_CIFAR10.

    Attributes:
    - base_folder (str): The folder name where the dataset files are stored.
    - meta (dict): A dictionary containing metadata about the dataset.

    Methods:
    - __init__(self, cfg, train=True, transform=None, target_transform=None, download=True):
        Constructor method that initializes the CIFAR10 instance. It takes five arguments:

    - get_image(self, index):
        Returns the image data at the given index.
    """
    
    base_folder = 'cifar-10-batches-pyx`'
    meta = {
        'filename': 'batches.meta',
        'key': 'label_names',
        'md5': '5ff9c542aee3614f3951f8cda6e48888',
    }

    def __init__(self, cfg, train=True, transform=None, target_transform=None, download=True):
        
        """
        Initializes the CIFAR10 dataset object.

        Args:
            cfg (object): Configuration object.
            train (bool): If True, return the training set, otherwise the test set.
            transform (callable, optional): A transform that takes in an PIL image and returns a transformed version.
            target_transform (callable, optional): A transform that takes in the target and transforms it.
            download (bool): If True, downloads the dataset from the internet and places it in root directory.
        Returns:
            None

        """
        
        root = os.path.join(cfg.DATASET_ROOT, 'cifar10')
        super().__init__(root, train, transform, target_transform, download)

    def get_image(self, index):
        """
        Returns the image at the specified index.

        Args:
            index (int): The index of the image to return.

        Returns:
            (numpy.ndarray): The image at the specified index.
        """
        return self.data[index]


class CIFAR100(Old_CIFAR100):
    """
    A subclass of the CIFAR-100 dataset that loads data from PyTorch's preprocessed CIFAR-100 dataset.
    
    Methods:
    - __init__(self, cfg, train=True, transform=None, target_transform=None, download=True):
        Constructor method that initializes the CIFAR100 instance. It takes five arguments:

    - get_image(self, index):
        Returns the image data at the given index.
    """
    
    base_folder = 'cifar-100-python'
    meta = {
        'filename': 'meta',
        'key': 'fine_label_names',
        'md5': '7973b15100ade9c7d40fb424638fde48',
    }

    def __init__(self, cfg, train=True, transform=None, target_transform=None, download=False):
        
        """
        Initializes the CIFAR10 dataset object.

        Args:
            cfg (object): Configuration object.
            train (bool): If True, return the training set, otherwise the test set.
            transform (callable, optional): A transform that takes in an PIL image and returns a transformed version.
            target_transform (callable, optional): A transform that takes in the target and transforms it.
            download (bool): If True, downloads the dataset from the internet and places it in root directory.
        Returns:
            None

        """
        
        root = os.path.join(cfg.DATASET_ROOT, 'cifar100')
        super().__init__(root, train, transform, target_transform, download)

    def get_image(self, index):
        """
        Returns the image at the specified index.

        Args:
            index (int): The index of the image to return.

        Returns:
            (numpy.ndarray): The image at the specified index.
        """
        return self.data[index]


class SVHN(Old_SVHN):
    
    """
    A dataset class representing the Street View House Numbers (SVHN) dataset.
    
    Methods:
    - __init__(self, cfg, train=True, transform=None, target_transform=None, download=False):
        Constructor method that initializes the SVHN instance. It takes five arguments:
    - get_image(self, index):
        Returns the image data at the given index.
    """
    
    def __init__(self, cfg, train=True, transform=None, target_transform=None, download=False):
        
        """
        Initializes the SVHN dataset object.
        
        Args:
            cfg (object): Configuration object.
            train (bool): If True, return the training set, otherwise the test set.
            transform (callable, optional): A transform that takes in an PIL image and returns a transformed version.
            target_transform (callable, optional): A transform that takes in the target and transforms it.
            download (bool): If True, downloads the dataset from the internet and places it in root directory.
        
        Returns:
            None
            
        """
        
        root = os.path.join(cfg.DATASET_ROOT, 'svhn')
        # split argument should be one of {‘train’, ‘test’, ‘extra’}
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
            None
        """
        return self.data[index]


class TinyImagesSubset(ImageFolder):
    """
    A 800K subset of the 80M TinyImages data consisting of 32x32 pixel images from the internet. 
    Note: that the dataset is unlabeled.
    
    Methods:
    - __init__(self, cfg, train=True, transform=None, target_transform=None):
        Constructor method that initializes the TinyImagesSubset instance. It takes five arguments:
    - get_image(self, index):
        Returns the image data at the given index.
    
    """

    def __init__(self, train=True, transform=None, target_transform=None):
        root = os.path.join(cfg.DATASET_ROOT, 'tiny-images-subset')
        if not os.path.exists(root):
            raise ValueError('Dataset not found at {}. Please download it from {}.'.format(
                root, 'https://github.com/Silent-Zebra/tiny-images-subset'
            ))

        # Initialize ImageFolder
        fold = 'train' if train else 'test'
        super().__init__(root=os.path.join(root, fold), transform=transform,
                         target_transform=target_transform)
        self.root = root

        print('=> done loading {} ({}) with {} examples'.format(self.__class__.__name__, 'train' if train else 'test',
                                                                len(self.samples)))
