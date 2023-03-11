'''
This module provides three functions: EfficientNet_V2_S, EfficientNet_V2_M, and EfficientNet_V2_L
which return instances of the corresponding EfficientNet V2 model.
'''

from typing import  Any
import torch.nn as nn
from torchvision.models import efficientnet_v2_s, efficientnet_v2_m, efficientnet_v2_l  
from torchvision.models.efficientnet import EfficientNet_V2_L_Weights, EfficientNet_V2_M_Weights, EfficientNet_V2_S_Weights


def EfficientNet_V2_S(num_classes, weights: str = "default",  progress: bool = True, **kwargs: Any):
    '''
    EfficientNet_V2_S(num_classes, weights=None, progress=True, **kwargs)
    Returns a EfficientNet_V2_S model pre-trained on ImageNet dataset with a linear classifier layer at the end.
    
    Args:
        num_classes (int): The number of output classes in the final linear layer.
        weights (optional, str or None): Which weights to use for initialization. Can be one of "default", "imagenet1k_v1", or None. Default is "default".
        progress (bool, optional): If True, displays a progress bar of the download. Default is True.
        **kwargs: Additional arguments passed to the efficientnet_v2_s() function from torchvision.models.
    
    Returns:
        EfficientNet_V2_S model with the specified number of output classes.
    '''
    if weights == 'default':
        weights = EfficientNet_V2_S_Weights.DEFAULT
    elif weights == 'imagenet1k_v1':
        weights = EfficientNet_V2_S_Weights.IMAGENET1K_V1
    else:
        weights = None
    model = efficientnet_v2_s(weights=weights, progress=progress, **kwargs)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    return model


def EfficientNet_V2_M(num_classes, weights: str = "default",  progress: bool = True, **kwargs: Any):
    '''
    EfficientNet_V2_M(num_classes, weights=None, progress=True, **kwargs)
    Returns a EfficientNet_V2_M model pre-trained on ImageNet dataset with a linear classifier layer at the end.
    
    Args:
        num_classes (int): The number of output classes in the final linear layer.
        weights (optional, str or None): Which weights to use for initialization. Can be one of "default", "imagenet1k_v1", or None. Default is "default".
        progress (bool, optional): If True, displays a progress bar of the download. Default is True.
        **kwargs: Additional arguments passed to the efficientnet_v2_m() function from torchvision.models.
    
    Returns:
        EfficientNet_V2_M model with the specified number of output classes.
    '''
    if weights == 'default':
        weights = EfficientNet_V2_M_Weights.DEFAULT
    elif weights == 'imagenet1k_v1':
        weights = EfficientNet_V2_M_Weights.IMAGENET1K_V1
    else:
        weights = None
    model = efficientnet_v2_m(weights=weights, progress=progress, **kwargs)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    return model


def EfficientNet_V2_L(num_classes, weights: str = "default",  progress: bool = True, **kwargs: Any):
    '''
    EfficientNet_V2_L(num_classes, weights=None, progress=True, **kwargs)
    Returns a EfficientNet_V2_L model pre-trained on ImageNet dataset with a linear classifier layer at the end.
    
    Args:
        num_classes (int): The number of output classes in the final linear layer.
        weights (optional, str or None): Which weights to use for initialization. Can be one of "default", "imagenet1k_v1", or None. Default is "default".
        progress (bool, optional): If True, displays a progress bar of the download. Default is True.
        **kwargs: Additional arguments passed to the efficientnet_v2_l() function from torchvision.models.
    
    Returns:
        EfficientNet_V2_L model with the specified number of output classes.
    '''
    if weights == 'default':
        weights = EfficientNet_V2_L_Weights.DEFAULT
    elif weights == 'imagenet1k_v1':
        weights = EfficientNet_V2_L_Weights.IMAGENET1K_V1
    else:
        weights = None
    model = efficientnet_v2_l(weights=weights, progress=progress, **kwargs)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    return model