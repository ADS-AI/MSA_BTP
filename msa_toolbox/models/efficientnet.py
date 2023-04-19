'''
This module provides eight functions, EfficientNet_B0, EfficientNet_B1, EfficientNet_B2,
EfficientNet_B3, EfficientNet_B4, EfficientNet_B5, EfficientNet_B6, EfficientNet_B7 which return
the instances of the corresponding EfficientNet model.
'''

from typing import  Any
import torch.nn as nn
import torch
from torchvision.models import efficientnet_b0, efficientnet_b1, efficientnet_b2, efficientnet_b3, efficientnet_b4, efficientnet_b5, efficientnet_b6, efficientnet_b7
from torchvision.models.efficientnet import EfficientNet_B0_Weights, EfficientNet_B1_Weights, EfficientNet_B2_Weights, EfficientNet_B3_Weights, EfficientNet_B4_Weights, EfficientNet_B5_Weights, EfficientNet_B6_Weights, EfficientNet_B7_Weights


def EfficientNet_B0(num_classes, weights: str = 'default', progress: bool = True, **kwargs: Any):
    '''
    EfficientNet_B0(num_classes, weights=None, progress=True, **kwargs)
    Returns an EfficientNet_B0 model pre-trained on the ImageNet dataset with a linear classifier layer at the end.

    Args:
        num_classes (int): The number of output classes in the final linear layer.
        weights (optional, str or None): Which weights to use for initialization. Can be one of "default", "imagenet1k_v1", or None. Default is 'default.
        progress (bool, optional): If True, displays a progress bar of the download. Default is True.
        **kwargs: Additional arguments passed to the efficientnet_b0() function from torchvision.models.

    Returns:
        EfficientNet_B0 model with the specified number of output classes.
    '''
    if weights == 'default':
        weights = EfficientNet_B0_Weights.DEFAULT
    elif weights == 'imagenet1k_v1':
        weights = EfficientNet_B0_Weights.IMAGENET1K_V1
    else:
        weights = None
    weights = torch.hub.load("pytorch/vision:v0.14.1", "get_weight", name="EfficientNet_B0_Weights.IMAGENET1K_V1")    
    model = efficientnet_b0(weights=weights, progress=progress, **kwargs)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    model.transforms = weights.transforms()
    return model


def EfficientNet_B1(num_classes, weights: str = 'default', progress: bool = True, **kwargs: Any):
    '''
    EfficientNet_B1(num_classes, weights=None, progress=True, **kwargs)
    Returns an EfficientNet_B1 model pre-trained on the ImageNet dataset with a linear classifier layer at the end.

    Args:
        num_classes (int): The number of output classes in the final linear layer.
        weights (optional, str or None): Which weights to use for initialization. Can be one of "default", "imagenet1k_v1", "imagenet1k_v2" or None. Default is 'default.
        progress (bool, optional): If True, displays a progress bar of the download. Default is True.
        **kwargs: Additional arguments passed to the efficientnet_b1() function from torchvision.models.

    Returns:
        EfficientNet_B1 model with the specified number of output classes.
    '''
    if weights == 'default':
        weights = EfficientNet_B1_Weights.DEFAULT
    elif weights == 'imagenet1k_v1':
        weights = EfficientNet_B1_Weights.IMAGENET1K_V1
    elif weights == 'imagenet1k_v2':
        weights = EfficientNet_B1_Weights.IMAGENET1K_V2
    else:
        weights = None
    weights = torch.hub.load("pytorch/vision:v0.14.1", "get_weight", name="EfficientNet_B1_Weights.IMAGENET1K_V1")
    model = efficientnet_b1(weights=weights, progress=progress, **kwargs)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    model.transforms = weights.transforms()
    return model 


def EfficientNet_B2(num_classes, weights: str = 'default', progress: bool = True, **kwargs: Any):
    '''
    EfficientNet_B2(num_classes, weights=None, progress=True, **kwargs)
    Returns an EfficientNet_B2 model pre-trained on the ImageNet dataset with a linear classifier layer at the end.

    Args:
        num_classes (int): The number of output classes in the final linear layer.
        weights (optional, str or None): Which weights to use for initialization. Can be one of "default", "imagenet1k_v1" or None. Default is 'default.
        progress (bool, optional): If True, displays a progress bar of the download. Default is True.
        **kwargs: Additional arguments passed to the efficientnet_b2() function from torchvision.models.

    Returns:
        EfficientNet_B2 model with the specified number of output classes.
    '''
    if weights == 'default':
        weights = EfficientNet_B2_Weights.DEFAULT
    elif weights == 'imagenet1k_v1':
        weights = EfficientNet_B2_Weights.IMAGENET1K_V1
    else:
        weights = None
    weights = torch.hub.load("pytorch/vision:v0.14.1", "get_weight", name="EfficientNet_B2_Weights.IMAGENET1K_V1")    
    model = efficientnet_b2(weights=weights, progress=progress, **kwargs)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    model.transforms = weights.transforms()
    return model


def EfficientNet_B3(num_classes, weights: str = 'default', progress: bool = True, **kwargs: Any):
    '''
    EfficientNet_B3(num_classes, weights=None, progress=True, **kwargs)
    Returns an EfficientNet_B3 model pre-trained on the ImageNet dataset with a linear classifier layer at the end.

    Args:
        num_classes (int): The number of output classes in the final linear layer.
        weights (optional, str or None): Which weights to use for initialization. Can be one of "default", "imagenet1k_v1" or None. Default is 'default.
        progress (bool, optional): If True, displays a progress bar of the download. Default is True.
        **kwargs: Additional arguments passed to the efficientnet_b3() function from torchvision.models.

    Returns:
        EfficientNet_B3 model with the specified number of output classes.
    '''
    if weights == 'default':
        weights = EfficientNet_B3_Weights.DEFAULT
    elif weights == 'imagenet1k_v1':
        weights = EfficientNet_B3_Weights.IMAGENET1K_V1
    else:
        weights = None
    weights = torch.hub.load("pytorch/vision:v0.14.1", "get_weight", name="EfficientNet_B3_Weights.IMAGENET1K_V1")    
    model = efficientnet_b3(weights=weights, progress=progress, **kwargs)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    model.transforms = weights.transforms()
    return model

def EfficientNet_B4(num_classes, weights: str = 'default', progress: bool = True, **kwargs: Any):
    '''
    EfficientNet_B4(num_classes, weights=None, progress=True, **kwargs)
    Returns an EfficientNet_B4 model pre-trained on the ImageNet dataset with a linear classifier layer at the end.

    Args:
        num_classes (int): The number of output classes in the final linear layer.
        weights (optional, str or None): Which weights to use for initialization. Can be one of "default", "imagenet1k_v1" or None. Default is 'default.
        progress (bool, optional): If True, displays a progress bar of the download. Default is True.
        **kwargs: Additional arguments passed to the efficientnet_b4() function from torchvision.models.

    Returns:
        EfficientNet_B4 model with the specified number of output classes.
    '''
    if weights == 'default':
        weights = EfficientNet_B4_Weights.DEFAULT
    elif weights == 'imagenet1k_v1':
        weights = EfficientNet_B4_Weights.IMAGENET1K_V1
    else:
        weights = None
    weights = torch.hub.load("pytorch/vision:v0.14.1", "get_weight", name="EfficientNet_B4_Weights.IMAGENET1K_V1")    
    model = efficientnet_b4(weights=weights, progress=progress, **kwargs)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    model.transforms = weights.transforms()
    return model


def EfficientNet_B5(num_classes, weights: str = 'default', progress: bool = True, **kwargs: Any):
    '''
    EfficientNet_B5(num_classes, weights=None, progress=True, **kwargs)
    Returns an EfficientNet_B5 model pre-trained on the ImageNet dataset with a linear classifier layer at the end.

    Args:
        num_classes (int): The number of output classes in the final linear layer.
        weights (optional, str or None): Which weights to use for initialization. Can be one of "default", "imagenet1k_v1" or None. Default is 'default.
        progress (bool, optional): If True, displays a progress bar of the download. Default is True.
        **kwargs: Additional arguments passed to the efficientnet_b5() function from torchvision.models.

    Returns:
        EfficientNet_B5 model with the specified number of output classes.
    '''
    if weights == 'default':
        weights = EfficientNet_B5_Weights.DEFAULT
    elif weights == 'imagenet1k_v1':
        weights = EfficientNet_B5_Weights.IMAGENET1K_V1
    else:
        weights = None
    weights = torch.hub.load("pytorch/vision:v0.14.1", "get_weight", name="EfficientNet_B5_Weights.IMAGENET1K_V1")    
    model = efficientnet_b5(weights=weights, progress=progress, **kwargs)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    model.transforms = weights.transforms()
    return model


def EfficientNet_B6(num_classes, weights: str = 'default', progress: bool = True, **kwargs: Any):
    '''
    EfficientNet_B6(num_classes, weights=None, progress=True, **kwargs)
    Returns an EfficientNet_B6 model pre-trained on the ImageNet dataset with a linear classifier layer at the end.

    Args:
        num_classes (int): The number of output classes in the final linear layer.
        weights (optional, str or None): Which weights to use for initialization. Can be one of "default", "imagenet1k_v1" or None. Default is 'default.
        progress (bool, optional): If True, displays a progress bar of the download. Default is True.
        **kwargs: Additional arguments passed to the efficientnet_b6() function from torchvision.models.

    Returns:
        EfficientNet_B6 model with the specified number of output classes.
    '''
    if weights == 'default':
        weights = EfficientNet_B6_Weights.DEFAULT
    elif weights == 'imagenet1k_v1':
        weights = EfficientNet_B6_Weights.IMAGENET1K_V1
    else:
        weights = None
    weights = torch.hub.load("pytorch/vision:v0.14.1", "get_weight", name="EfficientNet_B6_Weights.IMAGENET1K_V1")    
    model = efficientnet_b6(weights=weights, progress=progress, **kwargs)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    model.transforms = weights.transforms()
    return model


def EfficientNet_B7(num_classes, weights: str = 'default', progress: bool = True, **kwargs: Any):
    '''
    EfficientNet_B7(num_classes, weights=None, progress=True, **kwargs)
    Returns an EfficientNet_B7 model pre-trained on the ImageNet dataset with a linear classifier layer at the end.

    Args:
        num_classes (int): The number of output classes in the final linear layer.
        weights (optional, str or None): Which weights to use for initialization. Can be one of "default", "imagenet1k_v1" or None. Default is 'default.
        progress (bool, optional): If True, displays a progress bar of the download. Default is True.
        **kwargs: Additional arguments passed to the efficientnet_b7() function from torchvision.models.

    Returns:
        EfficientNet_B7 model with the specified number of output classes.
    '''
    if weights == 'default':
        weights = EfficientNet_B7_Weights.DEFAULT
    elif weights == 'imagenet1k_v1':
        weights = EfficientNet_B7_Weights.IMAGENET1K_V1
    else:
        weights = None
    weights = torch.hub.load("pytorch/vision:v0.14.1", "get_weight", name="EfficientNet_B7_Weights.IMAGENET1K_V1")    
    model = efficientnet_b7(weights=weights, progress=progress, **kwargs)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    model.transforms = weights.transforms()
    return model
