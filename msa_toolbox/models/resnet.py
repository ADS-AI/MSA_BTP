'''
This module provides two functions, ResNet50 and ResNet18, which return instances of the 
ResNet-50 and ResNet-18 models respectively.
'''

from typing import Any
import torch.nn as nn
from torchvision.models import resnet50, resnet18,  ResNet50_Weights, ResNet18_Weights


def ResNet50(num_classes, weights: str ='default',  progress: bool = True, **kwargs: Any):
    '''
    ResNet50(num_classes, weights=None, progress=True, **kwargs)
    Returns a ResNet-50 model pre-trained on ImageNet dataset with a linear classifier layer at the end.
    
    Args:
        num_classes (int): The number of output classes in the final linear layer.
        weights (optional, str or None): Which weights to use for initialization. Can be one of "default", "imagenet1k_v1", "imagenet1k_v2", or None. Default is "default".
        progress (bool, optional): If True, displays a progress bar of the download. Default is True.
        **kwargs: Additional arguments passed to the resnet50() function from torchvision.models.

    Returns:
        ResNet-50 model with the specified number of output classes.
    '''
    if weights == 'default':
        weights = ResNet50_Weights.DEFAULT
    elif weights == 'imagenet1k_v1':
        weights = ResNet50_Weights.IMAGENET1K_V1
    elif weights == 'imagenet1k_v2':
        weights = ResNet50_Weights.IMAGENET1K_V2
    else:
        weights = None
    model = resnet50(weights=weights, progress=progress, **kwargs)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def ResNet18(num_classes, weights: str = 'default',  progress: bool = True, **kwargs: Any):
    '''
    ResNet18(num_classes, weights=None, progress=True, **kwargs)
    Returns a ResNet-18 model pre-trained on ImageNet dataset with a linear classifier layer at the end.
    
    Args:
        num_classes (int): The number of output classes in the final linear layer.
        weights (optional, str or None): Which weights to use for initialization. Can be one of "default", "imagenet1k_v1", or None. Default is "default".
        progress (bool, optional): If True, displays a progress bar of the download. Default is True.
        **kwargs: Additional arguments passed to the resnet18() function from torchvision.models.

    Returns:
        ResNet-18 model with the specified number of output classes.
    '''
    if weights == 'default':
        weights = ResNet18_Weights.DEFAULT
    elif weights == 'imagenet1k_v1':
        weights = ResNet18_Weights.IMAGENET1K_V1
    else:
        weights = None
    model = resnet18(weights=weights, progress=progress, **kwargs)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model
