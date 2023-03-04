'''
This module provides two functions, MobileNet_V3_Small and MobileNet_V3_Large, which return the
instances of the MobileNet V3 models respectively.
'''

from typing import Any
import torch.nn as nn
from torchvision.models import mobilenet_v3_small, mobilenet_v3_large, MobileNet_V3_Small_Weights, MobileNet_V3_Large_Weights


def MobileNet_V3_Small(num_classes, weights: str = "default",  progress: bool = True, **kwargs: Any):
    '''
    MobileNet_V3_Small(num_classes, weights=None, progress=True, **kwargs)
    Returns a MobileNetV3_Small model pre-trained on ImageNet dataset with a linear classifier layer at the end.
    
    Args:
        num_classes (int): The number of output classes in the final linear layer.
        weights (optional, str or None): Which weights to use for initialization. Can be one of "default", "imagenet1k_v1", or None. Default is "default".
        progress (bool, optional): If True, displays a progress bar of the download. Default is True.
        **kwargs: Additional arguments passed to the mobilenet_v3_small() function from torchvision.models.

    Returns:
        MobileNet_V3_Small model with the specified number of output classes.
    '''
    if weights == 'default':
        weights = MobileNet_V3_Small_Weights.DEFAULT
    elif weights == 'imagenet1k_v1':
        weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1
    else:
        weights = None
    model = mobilenet_v3_small(weights=weights, progress=progress, **kwargs)
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
    return model


def MobileNet_V3_Large(num_classes, weights: str = "default",  progress: bool = True, **kwargs: Any):
    '''
    MobileNet_V3_Large(num_classes, weights=None, progress=True, **kwargs)
    Returns a MobileNetV3_Large model pre-trained on ImageNet dataset with a linear classifier layer at the end.
    
    Args:
        num_classes (int): The number of output classes in the final linear layer.
        weights (optional, str or None): Which weights to use for initialization. Can be one of "default", "imagenet1k_v1", or None. Default is "default".
        progress (bool, optional): If True, displays a progress bar of the download. Default is True.
        **kwargs: Additional arguments passed to the mobilenet_v3_large() function from torchvision.models.

    Returns:
        MobileNet_V3_Large model with the specified number of output classes.
    '''
    if weights == 'default':
        weights = MobileNet_V3_Large_Weights.DEFAULT
    elif weights == 'imagenet1k_v1':
        weights = MobileNet_V3_Large_Weights.IMAGENET1K_V1
    elif weights == 'imagenet1k_v2':
        weights = MobileNet_V3_Large_Weights.IMAGENET1K_V2
    else:
        weights = None
    model = mobilenet_v3_large(weights=weights, progress=progress, **kwargs)
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
    return model