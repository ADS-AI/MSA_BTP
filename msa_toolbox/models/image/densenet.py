'''
This module provides three functions: DenseNet121, DenseNet169, and DenseNet161 which return 
instances of the corresponding DenseNet model.
'''

from typing import Any
import torch.nn as nn
import torch
from torchvision.models import densenet121, densenet169, densenet161
from torchvision.models.densenet import DenseNet121_Weights, DenseNet169_Weights, DenseNet161_Weights

def DenseNet121(num_classes, weights: str = "default",  progress: bool = True, **kwargs: Any):
    '''
    DenseNet121(num_classes, weights=None, progress=True, **kwargs)
    Returns a DenseNet121 model pre-trained on ImageNet dataset with a linear classifier layer at the end.
    
    Args:
        num_classes (int): The number of output classes in the final linear layer.
        weights (optional, str or None): Which weights to use for initialization. Can be one of "default", "imagenet1k_v1", or None. Default is "default".
        progress (bool, optional): If True, displays a progress bar of the download. Default is True.
        **kwargs: Additional arguments passed to the densenet121() function from torchvision.models.
    
    Returns:
        DenseNet121 model with the specified number of output classes.
    '''
    if weights == 'default':
        weights = DenseNet121_Weights.DEFAULT
    elif weights == 'imagenet1k_v1':
        weights = DenseNet121_Weights.IMAGENET1K_V1
    else:
        weights = None
    weights = torch.hub.load("pytorch/vision:v0.14.1", "get_weight", name="DenseNet121_Weights.IMAGENET1K_V1")
    model = densenet121(weights=weights, progress=progress, **kwargs)
    model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    model.transforms = weights.transforms()
    return model


def DenseNet169(num_classes, weights: str = "default",  progress: bool = True, **kwargs: Any):
    '''
    DenseNet169(num_classes, weights=None, progress=True, **kwargs)
    Returns a DenseNet169 model pre-trained on ImageNet dataset with a linear classifier layer at the end.
    
    Args:
        num_classes (int): The number of output classes in the final linear layer.
        weights (optional, str or None): Which weights to use for initialization. Can be one of "default", "imagenet1k_v1", or None. Default is "default".
        progress (bool, optional): If True, displays a progress bar of the download. Default is True.
        **kwargs: Additional arguments passed to the densenet169() function from torchvision.models.
    
    Returns:
        DenseNet169 model with the specified number of output classes.
    '''
    if weights == 'default':
        weights = DenseNet169_Weights.DEFAULT
    elif weights == 'imagenet1k_v1':
        weights = DenseNet169_Weights.IMAGENET1K_V1
    else:
        weights = None
    weights = torch.hub.load("pytorch/vision:v0.14.1", "get_weight", name="DenseNet169_Weights.IMAGENET1K_V1")
    model = densenet169(weights=weights, progress=progress, **kwargs)
    model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    model.transforms = weights.transforms()
    return model


def DenseNet161(num_classes, weights: str = "default",  progress: bool = True, **kwargs: Any):
    '''
    DenseNet161(num_classes, weights=None, progress=True, **kwargs)
    Returns a DenseNet161 model pre-trained on ImageNet dataset with a linear classifier layer at the end.
    
    Args:
        num_classes (int): The number of output classes in the final linear layer.
        weights (optional, str or None): Which weights to use for initialization. Can be one of "default", "imagenet1k_v1", or None. Default is "default".
        progress (bool, optional): If True, displays a progress bar of the download. Default is True.
        **kwargs: Additional arguments passed to the densenet161() function from torchvision.models.
    
    Returns:
        DenseNet161 model with the specified number of output classes.
    '''
    if weights == 'default':
        weights = DenseNet161_Weights.DEFAULT
    elif weights == 'imagenet1k_v1':
        weights = DenseNet161_Weights.IMAGENET1K_V1
    else:
        weights = None
    weights = torch.hub.load("pytorch/vision:v0.14.1", "get_weight", name="DenseNet161_Weights.IMAGENET1K_V1")
    model = densenet161(weights=weights, progress=progress, **kwargs)
    model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    model.transforms = weights.transforms()
    return model