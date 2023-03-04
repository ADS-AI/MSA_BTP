'''
This module provides a function, AlexNet, which returns an instance of the AlexNet model. 
AlexNet is a convolutional neural network designed for image classification tasks.
'''

from typing import Any
import torch.nn as nn
from torchvision.models import alexnet, AlexNet_Weights


def AlexNet(num_classes, weights: str = 'default', progress: bool = True, **kwargs: Any):
    '''
    AlexNet(num_classes, weights=None, progress=True, **kwargs)
    Returns an AlexNet model pre-trained on the ImageNet dataset with a linear classifier layer at the end.

    Args:
        num_classes (int): The number of output classes in the final linear layer.
        weights (optional, str or None): Which weights to use for initialization. Can be one of "default", "imagenet1k_v1", or None. Default is 'default.
        progress (bool, optional): If True, displays a progress bar of the download. Default is True.
        **kwargs: Additional arguments passed to the alexnet() function from torchvision.models.

    Returns:
        AlexNet model with the specified number of output classes.
    '''
    if weights == 'default':
        weights = AlexNet_Weights.DEFAULT
    elif weights == 'imagenet1k_v1':
        weights = AlexNet_Weights.IMAGENET1K_V1
    else:
        weights = None
    model = alexnet(weights=weights, progress=progress, **kwargs)
    model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
    return model