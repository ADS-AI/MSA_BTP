'''
This module provides two functions, ResNet50 and ResNet18, which return instances of the 
ResNet-50 and ResNet-18 models respectively.
'''

import torch
from typing import Any
import torch.nn as nn
from torchvision.models import resnet50, resnet18, resnet34, resnet101, resnet152  
from torchvision.models.resnet import ResNet50_Weights, ResNet18_Weights, ResNet34_Weights, ResNet101_Weights, ResNet152_Weights
from torchvision import transforms


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
    weights = torch.hub.load("pytorch/vision:v0.14.1", "get_weight", name="ResNet50_Weights.IMAGENET1K_V1")    
    model = resnet50(weights=weights, progress=progress, **kwargs)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3), 
        weights.transforms()
    ])
    model.transforms = transform
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
    weights = torch.hub.load("pytorch/vision:v0.14.1", "get_weight", name="ResNet18_Weights.IMAGENET1K_V1")    
    model = resnet18(weights=weights, progress=progress, **kwargs)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3), 
        weights.transforms()
    ])
    model.transforms = transform
    return model


def ResNet34(num_classes, weights: str = 'default',  progress: bool = True, **kwargs: Any):
    '''
    ResNet34(num_classes, weights=None, progress=True, **kwargs)
    Returns a ResNet-34 model pre-trained on ImageNet dataset with a linear classifier layer at the end.
    
    Args:
        num_classes (int): The number of output classes in the final linear layer.
        weights (optional, str or None): Which weights to use for initialization. Can be one of "default", "imagenet1k_v1", or None. Default is "default".
        progress (bool, optional): If True, displays a progress bar of the download. Default is True.
        **kwargs: Additional arguments passed to the resnet34() function from torchvision.models.

    Returns:
        ResNet-34 model with the specified number of output classes.
    '''
    if weights == 'default':
        weights = ResNet34_Weights.DEFAULT
    elif weights == 'imagenet1k_v1':
        weights = ResNet34_Weights.IMAGENET1K_V1
    else:
        weights = None
    weights = torch.hub.load("pytorch/vision:v0.14.1", "get_weight", name="ResNet34_Weights.IMAGENET1K_V1")    
    model = resnet34(weights=weights, progress=progress, **kwargs)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3), 
        weights.transforms()
    ])
    model.transforms = transform
    return model

def ResNet101(num_classes, weights: str = 'default',  progress: bool = True, **kwargs: Any):
    '''
    ResNet101(num_classes, weights=None, progress=True, **kwargs)
    Returns a ResNet-101 model pre-trained on ImageNet dataset with a linear classifier layer at the end.
    
    Args:
        num_classes (int): The number of output classes in the final linear layer.
        weights (optional, str or None): Which weights to use for initialization. Can be one of "default", "imagenet1k_v1", or None. Default is "default".
        progress (bool, optional): If True, displays a progress bar of the download. Default is True.
        **kwargs: Additional arguments passed to the resnet101() function from torchvision.models.

    Returns:
        ResNet-101 model with the specified number of output classes.
    '''
    if weights == 'default':
        weights = ResNet101_Weights.DEFAULT
    elif weights == 'imagenet1k_v1':
        weights = ResNet101_Weights.IMAGENET1K_V1
    else:
        weights = None
    weights = torch.hub.load("pytorch/vision:v0.14.1", "get_weight", name="ResNet101_Weights.IMAGENET1K_V1")    
    model = resnet101(weights=weights, progress=progress, **kwargs)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3), 
        weights.transforms()
    ])
    model.transforms = transform
    return model


def ResNet152(num_classes, weights: str = 'default',  progress: bool = True, **kwargs: Any):
    '''
    ResNet152(num_classes, weights=None, progress=True, **kwargs)
    Returns a ResNet-152 model pre-trained on ImageNet dataset with a linear classifier layer at the end.
    
    Args:
        num_classes (int): The number of output classes in the final linear layer.
        weights (optional, str or None): Which weights to use for initialization. Can be one of "default", "imagenet1k_v1", or None. Default is "default".
        progress (bool, optional): If True, displays a progress bar of the download. Default is True.
        **kwargs: Additional arguments passed to the resnet152() function from torchvision.models.

    Returns:
        ResNet-152 model with the specified number of output classes.
    '''
    if weights == 'default':
        weights = ResNet152_Weights.DEFAULT
    elif weights == 'imagenet1k_v1':
        weights = ResNet152_Weights.IMAGENET1K_V1
    else:
        weights = None
    weights = torch.hub.load("pytorch/vision:v0.14.1", "get_weight", name="ResNet152_Weights.IMAGENET1K_V1")    
    model = resnet152(weights=weights, progress=progress, **kwargs)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3), 
        weights.transforms()
    ])
    model.transforms = transform
    return model