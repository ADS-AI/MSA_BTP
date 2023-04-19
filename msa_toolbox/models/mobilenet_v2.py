'''
This module provides one function, MobileNet_V2 which return instance of the MobileNet V2 model.
'''

from typing import Any
import torch
import torch.nn as nn
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights


def MobileNet_V2(num_classes, weights: str = "default",  progress: bool = True, **kwargs: Any):
    '''
    MobileNetV2(num_classes, weights=None, progress=True, **kwargs)
    Returns a MobileNetV2 model pre-trained on ImageNet dataset with a linear classifier layer at the end.
    
    Args:
        num_classes (int): The number of output classes in the final linear layer.
        weights (optional, str or None): Which weights to use for initialization. Can be one of "default", "imagenet1k_v1", "imagenet1k_v2", or None. Default is "default".
        progress (bool, optional): If True, displays a progress bar of the download. Default is True.
        **kwargs: Additional arguments passed to the mobilenet_v2() function from torchvision.models.

    Returns:
        MobileNetV2 model with the specified number of output classes.
    '''
    if weights == 'default':
        weights = MobileNet_V2_Weights.DEFAULT
    elif weights == 'imagenet1k_v1':
        weights = MobileNet_V2_Weights.IMAGENET1K_V1
    elif weights == 'imagenet1k_v2':
        weights = MobileNet_V2_Weights.IMAGENET1K_V2
    else:
        weights = None
    weights = torch.hub.load("pytorch/vision:v0.14.1", "get_weight", name="MobileNet_V2_Weights.IMAGENET1K_V1")    
    model = mobilenet_v2(weights=weights, progress=progress, **kwargs)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    model.transforms = weights.transforms()
    return model
