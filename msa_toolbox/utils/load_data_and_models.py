from ..datasets import caltech256
from ..datasets import cifar
from ..datasets import cubs200
from ..datasets import diabetic5
from ..datasets import imagenet
from ..datasets import mnist
from ..datasets import svhn
from ..datasets import tinyimagenet200
from ..datasets import indoor67
from ..datasets import custom_dataset
from ..datasets import dataset_to_modelfamily, modelfamily_to_mean_std, modelfamily_to_transforms, modelfamily_to_transforms_sans_normalization
from ..models import alexnet
from ..models import resnet
from ..models import efficientnet
from ..models import vgg
from ..models import efficientnet_v2
from ..models import mobilenet_v2
from ..models import mobilenet_v3
from ..models import densenet
from . cfg_reader import CfgNode
from torch.utils.data import Dataset, DataLoader, Subset
from torch.utils.data.sampler import SubsetRandomSampler, SequentialSampler
from typing import Any, Callable, Iterable, TypeVar, Generic, Sequence, List, Optional, Union


def load_dataset(dataset_name, data_root:str, train=True, transform=None, target_transform=None, download=True):
    '''
    Return the specified dataset along with transform
    '''
    dataset_name = dataset_name.lower()
    if transform:
        if type(transform) == bool:
            model_family = dataset_to_modelfamily[dataset_name]
            if train:
                transform = modelfamily_to_transforms[model_family]['train']
            else:
                transform = modelfamily_to_transforms[model_family]['test']
        else:
            transform = transform

    if dataset_name == 'cifar10':
        return cifar.CIFAR10(train=train, transform=transform, target_transform=target_transform, download=download)
    elif dataset_name == 'cifar100':
        return cifar.CIFAR100(train=train, transform=transform, target_transform=target_transform, download=download)
    elif dataset_name == 'imagenet':
        return imagenet.ImageNet1k(root=data_root, train=train, transform=transform, target_transform=target_transform)
    elif dataset_name == 'mnist':
        return mnist.MNIST(train=train, transform=transform, target_transform=target_transform, download=download)
    elif dataset_name == 'kmnist':
        return mnist.KMNIST(train=train, transform=transform, target_transform=target_transform, download=download)
    elif dataset_name == 'fashionmnist':
        return mnist.FashionMNIST(train=train, transform=transform, target_transform=target_transform, download=download)
    elif dataset_name == 'emnist':
        return mnist.EMNIST(train=train, transform=transform, target_transform=target_transform, download=download)
    elif dataset_name == 'emnistletters':
        return mnist.EMNISTLetters(train=train, transform=transform, target_transform=target_transform, download=download)
    elif dataset_name == 'svhn':
        return svhn.SVHN(train=train, transform=transform, target_transform=target_transform, download=download)
    elif dataset_name == 'tinyimagenet200':
        return tinyimagenet200.TinyImageNet200(root=data_root, train=train, transform=transform, target_transform=target_transform)
    elif dataset_name == 'tinyimagesubset':
        return cifar.TinyImagesSubset(root=data_root, train=train, transform=transform, target_transform=target_transform)
    elif dataset_name == 'cubs200':
        return cubs200.CUBS200(root=data_root, train=train, transform=transform, target_transform=target_transform)
    elif dataset_name == 'diabetic5':
        return diabetic5.Diabetic5(root=data_root, train=train, transform=transform, target_transform=target_transform)
    elif dataset_name == 'indoor67':
        return indoor67.Indoor67(root=data_root, train=train, transform=transform, target_transform=target_transform)
    elif dataset_name == 'caltech256':
        return caltech256.Caltech256(root=data_root, train=train, transform=transform, target_transform=target_transform)
    else:
        raise ValueError('Unknown dataset: {}'.format(dataset_name))


def load_custom_dataset(root_dir, transform=None, target_transform=None):
    '''
    Loads a custom dataset from the specified root directory and returns it in a format suitable for machine learning tasks.

    Args:
        root_dir (str): The root directory of the custom dataset.
        transform (callable): An optional function/transform to be applied on the input data. Defaults to None.
        target_transform (callable): An optional function/transform to be applied on the target labels. Defaults to None.

    Returns:
        The custom dataset loaded from the specified root directory in a format suitable for machine learning tasks.
    '''
    return custom_dataset.CustomDataset(root_dir=root_dir, transform=transform, target_transform=target_transform)


def load_victim_dataset(dataset_name, cfg:CfgNode, train=True, transform=None, target_transform=None, download=False):
    '''
    Loads the victim dataset specified by 'dataset_name' and returns it in a format suitable for machine learning tasks. 

    Args:
        dataset_name (str): The name of the victim dataset to load.
                dataset_name must be one of the following: 
                'cifar10', 'cifar100', 'imagenet', 'mnist', 'kmnist', 'fashionmnist', 'emnist', 
                'emnistletters', 'svhn', 'tinyimagenet200', 'tinyimagesubset', 'cubs200', 
                'diabetic5', 'indoor67', 'caltech256'.
        train (bool): A flag indicating whether to load the training set (if True) or the test set (if False). Defaults to True.
        transform (callable): 
                - True: The victim model's preprocessing transforms will be applied to the data.
                - False: The victim model's preprocessing transforms will not be applied to the data.
                - callable: The specified callable will be applied to the data.
        target_transform (callable): An optional function/transform to be applied on the target labels. Defaults to None.
        download (bool): A flag indicating whether to download the dataset if it is not already present. Defaults to True.
    Returns:
        The loaded victim dataset in a format suitable for machine learning tasks.
    '''
    return load_dataset(dataset_name, cfg.VICTIM.DATASET_ROOT, train=train, transform=transform,
                        target_transform=target_transform, download=download)


def load_thief_dataset(dataset_name, cfg:CfgNode,  train=True, transform=None, target_transform=None, download=False):
    '''
    Loads the thief dataset specified by 'dataset_name' and returns it in a format suitable for machine learning tasks. 

    Args:
        dataset_name (str): The name of the thief dataset to load.
                dataset_name must be one of the following: 
                'cifar10', 'cifar100', 'imagenet', 'mnist', 'kmnist', 'fashionmnist', 'emnist', 
                'emnistletters', 'svhn', 'tinyimagenet200', 'tinyimagesubset', 'cubs200', 
                'diabetic5', 'indoor67', 'caltech256'.
        train (bool): A flag indicating whether to load the training set (if True) or the test set (if False). Defaults to True.
        transform (callable): 
                - True: The thief model's preprocessing transforms will be applied to the data.
                - False: The thief model's preprocessing transforms will not be applied to the data.
                - callable: The specified callable will be applied to the data.
        target_transform (callable): An optional function/transform to be applied on the target labels. Defaults to None.
        download (bool): A flag indicating whether to download the dataset if it is not already present. Defaults to True.
    Returns:
        The loaded thief dataset in a format suitable for machine learning tasks.
    '''
    return load_dataset(dataset_name, cfg.THIEF.DATASET_ROOT, train=train, transform=transform,
                        target_transform=target_transform, download=download)


def get_data_loader(dataset, batch_size: int = 128, shuffle: bool = False, sampler=None, batch_sampler=None,
                    num_workers: int = 2, collate_fn=None, pin_memory: bool = False, drop_last: bool = False,
                    prefetch_factor: int = 2, persistent_workers: bool = False, pin_memory_device: str = ""):
    '''
    Returns a data loader for the specified dataset.
    '''
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, sampler=sampler, batch_sampler=batch_sampler,
                      num_workers=num_workers, collate_fn=collate_fn, pin_memory=pin_memory, drop_last=drop_last,
                      prefetch_factor=prefetch_factor, persistent_workers=persistent_workers, pin_memory_device=pin_memory_device)


def load_models(model_name, num_classes,  weights, progress, **kwargs):
    '''
    Return the specified model
    '''
    model_name = model_name.lower()
    if model_name == 'resnet18':
        return resnet.ResNet18(num_classes, weights, progress, **kwargs)
    elif model_name == 'resnet50':
        return resnet.ResNet50(num_classes, weights, progress, **kwargs)
    elif model_name == 'resnet34':
        return resnet.ResNet34(num_classes, weights, progress, **kwargs)
    elif model_name == 'resnet101':
        return resnet.ResNet101(num_classes, weights, progress, **kwargs)
    elif model_name == 'resnet152':
        return resnet.ResNet152(num_classes, weights, progress, **kwargs)
    elif model_name == 'vgg11':
        return vgg.VGG11(num_classes, weights, progress, **kwargs)
    elif model_name == 'vgg13':
        return vgg.VGG13(num_classes, weights, progress, **kwargs)
    elif model_name == 'vgg16':
        return vgg.VGG16(num_classes, weights, progress, **kwargs)
    elif model_name == 'vgg19':
        return vgg.VGG19(num_classes, weights, progress, **kwargs)
    elif model_name == 'vgg11_bn':
        return vgg.VGG11_BN(num_classes, weights, progress, **kwargs)
    elif model_name == 'vgg13_bn':
        return vgg.VGG13_BN(num_classes, weights, progress, **kwargs)
    elif model_name == 'vgg16_bn':
        return vgg.VGG16_BN(num_classes, weights, progress, **kwargs)
    elif model_name == 'vgg19_bn':
        return vgg.VGG19_BN(num_classes, weights, progress, **kwargs)
    elif model_name == 'alexnet':
        return alexnet.AlexNet(num_classes, weights, progress, **kwargs)
    elif model_name == 'efficientnet_b0':
        return efficientnet.EfficientNet_B0(num_classes, weights, progress, **kwargs)
    elif model_name == 'efficientnet_b1':
        return efficientnet.EfficientNet_B1(num_classes, weights, progress, **kwargs)
    elif model_name == 'efficientnet_b2':
        return efficientnet.EfficientNet_B2(num_classes, weights, progress, **kwargs)
    elif model_name == 'efficientnet_b3':
        return efficientnet.EfficientNet_B3(num_classes, weights, progress, **kwargs)
    elif model_name == 'efficientnet_b4':
        return efficientnet.EfficientNet_B4(num_classes, weights, progress, **kwargs)
    elif model_name == 'efficientnet_b5':
        return efficientnet.EfficientNet_B5(num_classes, weights, progress, **kwargs)
    elif model_name == 'efficientnet_b6':
        return efficientnet.EfficientNet_B6(num_classes, weights, progress, **kwargs)
    elif model_name == 'efficientnet_b7':
        return efficientnet.EfficientNet_B7(num_classes, weights, progress, **kwargs)
    elif model_name == 'effcientnet_v2_s':
        return efficientnet_v2.EfficientNet_V2_S(num_classes, weights, progress, **kwargs)
    elif model_name == 'effcientnet_v2_m':
        return efficientnet_v2.EfficientNet_V2_M(num_classes, weights, progress, **kwargs)
    elif model_name == 'effcientnet_v2_l':
        return efficientnet_v2.EfficientNet_V2_L(num_classes, weights, progress, **kwargs)
    elif model_name == 'mobilenet_v2':
        return mobilenet_v2.MobileNet_V2(num_classes, weights, progress, **kwargs)
    elif model_name == 'mobilenet_v3_small':
        return mobilenet_v3.MobileNet_V3_Small(num_classes, weights, progress, **kwargs)
    elif model_name == 'mobilenet_v3_large':
        return mobilenet_v3.MobileNet_V3_Large(num_classes, weights, progress, **kwargs)
    elif model_name == 'densenet121':
        return densenet.DenseNet121(num_classes, weights, progress, **kwargs)
    elif model_name == 'densenet169':
        return densenet.DenseNet169(num_classes, weights, progress, **kwargs)
    elif model_name == 'densenet161':
        return densenet.DenseNet161(num_classes, weights, progress, **kwargs)
    else:
        return ValueError('Unknown Model: {}'.format(model_name))


def load_thief_model(model_name: str,  num_classes: int, weights: str = "default",  progress: bool = True, **kwargs: Any):
    '''
    Loads a pre-trained thief model specified by 'model_name' and returns it in a format suitable for machine learning tasks.

    Args:
        model_name (str): The name of the pre-trained thief model to load.
            - model_name must be one of the following: 
                'resnet18', 'resnet50', 'resnet34', 'vgg11', 'vgg13', 'vgg16', 'vgg19', 
                'vgg11_bn', 'vgg13_bn', 'vgg16_bn', 'vgg19_bn', 'alexnet', 
                'efficientnet_b0', 'efficientnet_b1', 'efficientnet_b2', 'efficientnet_b3', 
                'efficientnet_b4', 'efficientnet_b5', 'efficientnet_b6', 'efficientnet_b7', 
                'effcientnet_v2_s', 'effcientnet_v2_m', 'effcientnet_v2_l', 'mobilenet_v2', 
                'mobilenet_v3_small', 'mobilenet_v3_large', 'densenet121', 'densenet169', 'densenet161'.
        num_classes (int): The number of output classes for the loaded model.
        weights (str): Specifies which weights to load for the model. If 'default', loads the pre-trained weights. If 'random', initializes the model with random weights. Defaults to 'default'.
            - weights must be one of the following: 
                1. 'default' - for any pre-trained models.
                2. 'imagenet1k_v1' - for any pre-trained models.
                3. 'imagenet1k_v2' - only for 'efficientnet_b1', 'mobilenet_v2', 'mobilenet_v3_large', 'resnet50'
                4. 'imagenet1k_features' - only for 'vgg16'
                5. None - for no weights.
        progress (bool): A flag indicating whether to display a progress bar while downloading the pre-trained weights. Defaults to True.
        **kwargs (Any): Additional keyword arguments to be passed to the underlying 'load_models' function.

    Returns:
        The loaded pre-trained thief model in a format suitable for machine learning tasks.
    '''
    return load_models(model_name, num_classes=num_classes,  weights=weights,
                       progress=progress, **kwargs)


def load_victim_model(model_name: str, num_classes: int,  weights: str = "default",  progress: bool = True, **kwargs: Any):
    '''
    Loads a pre-trained victim model specified by 'model_name' and returns it in a format suitable for machine learning tasks.

    Args:
        model_name (str): The name of the pre-trained victim model to load.
            - model_name must be one of the following: 
                'resnet18', 'resnet50', 'vgg11', 'vgg13', 'vgg16', 'vgg19', 
                'vgg11_bn', 'vgg13_bn', 'vgg16_bn', 'vgg19_bn', 'alexnet', 
                'efficientnet_b0', 'efficientnet_b1', 'efficientnet_b2', 'efficientnet_b3', 
                'efficientnet_b4', 'efficientnet_b5', 'efficientnet_b6', 'efficientnet_b7', 
                'effcientnet_v2_s', 'effcientnet_v2_m', 'effcientnet_v2_l', 'mobilenet_v2', 
                'mobilenet_v3_small', 'mobilenet_v3_large'.
        num_classes (int): The number of output classes for the loaded model.
        weights (str): Specifies which weights to load for the model. If 'default', loads the pre-trained weights. If 'random', initializes the model with random weights. Defaults to 'default'.
            - weights must be one of the following: 
                1. 'default' - for any pre-trained models.
                2. 'imagenet1k_v1' - for any pre-trained models.
                3. 'imagenet1k_v2' - only for 'efficientnet_b1', 'mobilenet_v2', 'mobilenet_v3_large', 'resnet50'
                4. 'imagenet1k_features' - only for 'vgg16'
                5. None - for no weights.
        progress (bool): A flag indicating whether to display a progress bar while downloading the pre-trained weights. Defaults to True.
        **kwargs (Any): Additional keyword arguments to be passed to the underlying 'load_models' function.

    Returns:
        The loaded pre-trained victim model in a format suitable for machine learning tasks.
    '''
    return load_models(model_name, num_classes=num_classes,  weights=weights,
                       progress=progress, **kwargs)
