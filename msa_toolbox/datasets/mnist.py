import os.path as osp
from torchvision.datasets import MNIST as Old_MNIST
from torchvision.datasets import EMNIST as Old_EMNIST
from torchvision.datasets import FashionMNIST as Old_FashionMNIST
from torchvision.datasets import KMNIST as Old_KMNIST
import msa_toolbox.config as cfg


class MNIST(Old_MNIST):
    def __init__(self, train=True, transform=None, target_transform=None, download=True):
        root = osp.join(cfg.DATASET_ROOT, 'mnist')
        super().__init__(root, train, transform, target_transform, download)


class KMNIST(Old_KMNIST):
    def __init__(self, train=True, transform=None, target_transform=None, download=True):
        root = osp.join(cfg.DATASET_ROOT, 'kmnist')
        super().__init__(root, train, transform, target_transform, download)


class EMNIST(Old_EMNIST):
    def __init__(self, **kwargs):
        root = osp.join(cfg.DATASET_ROOT, 'emnist')
        super().__init__(root, split='balanced', download=True, **kwargs)
        self.data = self.data.permute(0, 2, 1)


class EMNISTLetters(Old_EMNIST):
    def __init__(self, **kwargs):
        root = osp.join(cfg.DATASET_ROOT, 'emnist')
        super().__init__(root, split='letters', download=True, **kwargs)
        self.data = self.data.permute(0, 2, 1)


class FashionMNIST(Old_FashionMNIST):
    def __init__(self, train=True, transform=None, target_transform=None, download=True):
        root = osp.join(cfg.DATASET_ROOT, 'mnist_fashion')
        super().__init__(root, train, transform, target_transform, download)
