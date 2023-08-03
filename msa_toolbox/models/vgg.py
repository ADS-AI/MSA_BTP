'''
This module provides six functions, VGG11, VGG11_BN, VGG13, VGG13_BN, VGG16, VGG16_BN, VGG19, VGG19_BN
which return the instances of the corresponding VGG model.
'''

from typing import  Any
import torch.nn as nn
import torch
from torchvision.models import vgg11, vgg11_bn, vgg13, vgg13_bn, vgg16, vgg16_bn, vgg19, vgg19_bn
from torchvision.models.vgg import VGG11_Weights, VGG11_BN_Weights, VGG13_Weights, VGG13_BN_Weights, VGG16_Weights, VGG16_BN_Weights, VGG19_Weights, VGG19_BN_Weights


def VGG11(num_classes, weights: str = "default",  progress: bool = True, **kwargs: Any):
    '''
    VGG11(num_classes, weights=None, progress=True, **kwargs)
    Returns a VGG11 model pre-trained on ImageNet dataset with a linear classifier layer at the end.
    
    Args:
        num_classes (int): The number of output classes in the final linear layer.
        weights (optional, str or None): Which weights to use for initialization. Can be one of "default", "imagenet1k_v1" or None. Default is "default".
        progress (bool, optional): If True, displays a progress bar of the download. Default is True.
        **kwargs: Additional arguments passed to the vgg11() function from torchvision.models.
    
    Returns:
        VGG11 model with the specified number of output classes.
    '''
    if weights == 'default':
        weights = VGG11_Weights.DEFAULT
    elif weights == 'imagenet1k_v1':
        weights = VGG11_Weights.IMAGENET1K_V1
    else:
        weights = None
    weights = torch.hub.load("pytorch/vision:v0.14.1", "get_weight", name="VGG11_Weights.IMAGENET1K_V1")    
    model = vgg11(weights=weights, progress=progress, **kwargs)
    model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
    model.transforms = weights.transforms()
    return model


def VGG11_BN(num_classes, weights: str = "default",  progress: bool = True, **kwargs: Any):
    '''
    VGG11_BN(num_classes, weights=None, progress=True, **kwargs)
    Returns a VGG11_BN model pre-trained on ImageNet dataset with a linear classifier layer at the end.
    
    Args:
        num_classes (int): The number of output classes in the final linear layer.
        weights (optional, str or None): Which weights to use for initialization. Can be one of "default", "imagenet1k_v1" or None. Default is "default".
        progress (bool, optional): If True, displays a progress bar of the download. Default is True.
        **kwargs: Additional arguments passed to the vgg11_bn() function from torchvision.models.
    
    Returns:
        VGG11_BN model with the specified number of output classes.
    '''
    if weights == 'default':
        weights = VGG11_BN_Weights.DEFAULT
    elif weights == 'imagenet1k_v1':
        weights = VGG11_BN_Weights.IMAGENET1K_V1
    else:
        weights = None
    weights = torch.hub.load("pytorch/vision:v0.14.1", "get_weight", name="VGG11_BN_Weights.IMAGENET1K_V1")    
    model = vgg11_bn(weights=weights, progress=progress, **kwargs)
    model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
    model.transforms = weights.transforms()
    return model


def VGG13(num_classes, weights: str = "default",  progress: bool = True, **kwargs: Any):
    '''
    VGG13(num_classes, weights=None, progress=True, **kwargs)
    Returns a VGG13 model pre-trained on ImageNet dataset with a linear classifier layer at the end.
    
    Args:
        num_classes (int): The number of output classes in the final linear layer.
        weights (optional, str or None): Which weights to use for initialization. Can be one of "default", "imagenet1k_v1" or None. Default is "default".
        progress (bool, optional): If True, displays a progress bar of the download. Default is True.
        **kwargs: Additional arguments passed to the vgg13() function from torchvision.models.
    
    Returns:
        VGG13 model with the specified number of output classes.
    '''
    if weights == 'default':
        weights = VGG13_Weights.DEFAULT
    elif weights == 'imagenet1k_v1':
        weights = VGG13_Weights.IMAGENET1K_V1
    else:
        weights = None
    weights = torch.hub.load("pytorch/vision:v0.14.1", "get_weight", name="VGG13_Weights.IMAGENET1K_V1")    
    model = vgg13(weights=weights, progress=progress, **kwargs)
    model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
    model.transforms = weights.transforms()
    return model


def VGG13_BN(num_classes, weights: str = "default",  progress: bool = True, **kwargs: Any):
    '''
    VGG13_BN(num_classes, weights=None, progress=True, **kwargs)
    Returns a VGG13_BN model pre-trained on ImageNet dataset with a linear classifier layer at the end.
    
    Args:
        num_classes (int): The number of output classes in the final linear layer.
        weights (optional, str or None): Which weights to use for initialization. Can be one of "default", "imagenet1k_v1" or None. Default is "default".
        progress (bool, optional): If True, displays a progress bar of the download. Default is True.
        **kwargs: Additional arguments passed to the vgg13_bn() function from torchvision.models.
    
    Returns:
        VGG13_BN model with the specified number of output classes.
    '''
    if weights == 'default':
        weights = VGG13_BN_Weights.DEFAULT
    elif weights == 'imagenet1k_v1':
        weights = VGG13_BN_Weights.IMAGENET1K_V1
    else:
        weights = None
    weights = torch.hub.load("pytorch/vision:v0.14.1", "get_weight", name="VGG13_BN_Weights.IMAGENET1K_V1")    
    model = vgg13_bn(weights=weights, progress=progress, **kwargs)
    model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
    model.transforms = weights.transforms()
    return model


def VGG16(num_classes, weights: str = "default",  progress: bool = True, **kwargs: Any):
    '''
    VGG16(num_classes, weights=None, progress=True, **kwargs)
    Returns a VGG16 model pre-trained on ImageNet dataset with a linear classifier layer at the end.
    
    Args:
        num_classes (int): The number of output classes in the final linear layer.
        weights (optional, str or None): Which weights to use for initialization. Can be one of "default", "imagenet1k_v1", "IMAGENET1K_FEATURES" or None. Default is "default".
        progress (bool, optional): If True, displays a progress bar of the download. Default is True.
        **kwargs: Additional arguments passed to the vgg16() function from torchvision.models.
    
    Returns:
        VGG16 model with the specified number of output classes.
    '''
    if weights == 'default':
        weights = VGG16_Weights.DEFAULT
    elif weights == 'imagenet1k_v1':
        weights = VGG16_Weights.IMAGENET1K_V1
    elif weights == 'imagenet1k_features':
        weights = VGG16_Weights.IMAGENET1K_FEATURES
    else:
        weights = None
    weights = torch.hub.load("pytorch/vision:v0.14.1", "get_weight", name="VGG16_Weights.IMAGENET1K_V1")    
    model = vgg16(weights=weights, progress=progress, **kwargs)
    model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
    model.transforms = weights.transforms()
    return model


def VGG16_BN(num_classes, weights: str = "default",  progress: bool = True, **kwargs: Any):
    '''
    VGG16_BN(num_classes, weights=None, progress=True, **kwargs)
    Returns a VGG16_BN model pre-trained on ImageNet dataset with a linear classifier layer at the end.
    
    Args:
        num_classes (int): The number of output classes in the final linear layer.
        weights (optional, str or None): Which weights to use for initialization. Can be one of "default", "imagenet1k_v1" or None. Default is "default".
        progress (bool, optional): If True, displays a progress bar of the download. Default is True.
        **kwargs: Additional arguments passed to the vgg16_bn() function from torchvision.models.
    
    Returns:
        VGG16_BN model with the specified number of output classes.
    '''
    if weights == 'default':
        weights = VGG16_BN_Weights.DEFAULT
    elif weights == 'imagenet1k_v1':
        weights = VGG16_BN_Weights.IMAGENET1K_V1
    else:
        weights = None
    weights = torch.hub.load("pytorch/vision:v0.14.1", "get_weight", name="VGG16_BN_Weights.IMAGENET1K_V1")    
    model = vgg16_bn(weights=weights, progress=progress, **kwargs)
    model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
    model.transforms = weights.transforms()
    return model


def VGG19(num_classes, weights: str = "default",  progress: bool = True, **kwargs: Any):
    '''
    VGG19(num_classes, weights=None, progress=True, **kwargs)
    Returns a VGG19 model pre-trained on ImageNet dataset with a linear classifier layer at the end.
    
    Args:
        num_classes (int): The number of output classes in the final linear layer.
        weights (optional, str or None): Which weights to use for initialization. Can be one of "default", "imagenet1k_v1" or None. Default is "default".
        progress (bool, optional): If True, displays a progress bar of the download. Default is True.
        **kwargs: Additional arguments passed to the vgg19() function from torchvision.models.
    
    Returns:
        VGG19 model with the specified number of output classes.
    '''
    if weights == 'default':
        weights = VGG19_Weights.DEFAULT
    elif weights == 'imagenet1k_v1':
        weights = VGG19_Weights.IMAGENET1K_V1
    else:
        weights = None
    weights = torch.hub.load("pytorch/vision:v0.14.1", "get_weight", name="VGG19_Weights.IMAGENET1K_V1")    
    model = vgg19(weights=weights, progress=progress, **kwargs)
    model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
    model.transforms = weights.transforms()
    return model


def VGG19_BN(num_classes, weights: str = "default",  progress: bool = True, **kwargs: Any):
    '''
    VGG19_BN(num_classes, weights=None, progress=True, **kwargs)
    Returns a VGG19_BN model pre-trained on ImageNet dataset with a linear classifier layer at the end.
    
    Args:
        num_classes (int): The number of output classes in the final linear layer.
        weights (optional, str or None): Which weights to use for initialization. Can be one of "default", "imagenet1k_v1" or None. Default is "default".
        progress (bool, optional): If True, displays a progress bar of the download. Default is True.
        **kwargs: Additional arguments passed to the vgg19_bn() function from torchvision.models.
    
    Returns:
        VGG19_BN model with the specified number of output classes.
    '''
    if weights == 'default':
        weights = VGG19_BN_Weights.DEFAULT
    elif weights == 'imagenet1k_v1':
        weights = VGG19_BN_Weights.IMAGENET1K_V1
    else:
        weights = None
    weights = torch.hub.load("pytorch/vision:v0.14.1", "get_weight", name="VGG19_BN_Weights.IMAGENET1K_V1")    
    model = vgg19_bn(weights=weights, progress=progress, **kwargs)
    model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
    model.transforms = weights.transforms()
    return model