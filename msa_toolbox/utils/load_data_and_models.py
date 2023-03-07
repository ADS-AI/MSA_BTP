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
from ..models import alexnet
from ..models import resnet
from ..models import efficientnet
from ..models import vgg
from ..models import efficientnet_v2
from ..models import mobilenet_v2
from ..models import mobilenet_v3

from typing import Any

def load_dataset(dataset_name, train=True, transform=None, target_transform=None, download=True):
    dataset_name = dataset_name.lower()
    if dataset_name == 'cifar10':
        return cifar.CIFAR10(train=train, transform=transform, target_transform=target_transform, download=download)
    elif dataset_name == 'cifar100':
        return cifar.CIFAR100(train=train, transform=transform, target_transform=target_transform, download=download)
    elif dataset_name == 'imagenet':
        return imagenet.ImageNet1k(train=train, transform=transform, target_transform=target_transform)
    elif dataset_name == 'mnist':
        return mnist.MNIST(train=train, transform=transform, target_transform=target_transform, download=download)
    elif dataset_name == 'kmnist':
        return mnist.KMNIST(train=train, transform=transform, target_transform=target_transform, download=download)
    elif dataset_name == 'fashion_mnist':
        return mnist.FashionMNIST(train=train, transform=transform, target_transform=target_transform, download=download)
    elif dataset_name == 'emist':
        return mnist.EMNIST(train=train, transform=transform, target_transform=target_transform, download=download)
    elif dataset_name =='emistleters':
        return mnist.EMNISTLetters(train=train, transform=transform, target_transform=target_transform, download=download)
    elif dataset_name == 'svhn':
        return svhn.SVHN(train=train, transform=transform, target_transform=target_transform, download=download)
    elif dataset_name == 'tiny_imagenet':
        return tinyimagenet200.TinyImageNet200(train=train, transform=transform, target_transform=target_transform)
    elif dataset_name == 'cubs200':
        return cubs200.CUBS200(train=train, transform=transform, target_transform=target_transform)
    elif dataset_name == 'diabetic5':
        return diabetic5.Diabetic5(train=train, transform=transform, target_transform=target_transform)
    elif dataset_name == 'indoor67':
        return indoor67.Indoor67(train=train, transform=transform, target_transform=target_transform)
    else:
        raise ValueError('Unknown dataset: {}'.format(dataset_name))
    


def load_victim_dataset(dataset_name, train=True, transform=None, target_transform=None, download=True):
    return load_dataset(dataset_name, train=train, transform=transform, 
                        target_transform=target_transform, download=download)


def load_thief_dataset(dataset_name, train=True, transform=None, target_transform=None, download=True):
    return load_dataset(dataset_name, train=train, transform=transform, 
                        target_transform=target_transform, download=download)


def load_models(model_name, num_classes,  weights, progress, **kwargs):
    model_name = model_name.lower()
    if model_name == 'resnet18':
        return resnet.ResNet18(num_classes, weights, progress, **kwargs)
    elif model_name == 'resnet50':
        return resnet.ResNet50(num_classes, weights, progress, **kwargs)
    
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


def load_thief_model(model_name:str,  num_classes:int ,weights: str = "default",  progress: bool = True, **kwargs: Any):
    return load_models(model_name, num_classes=num_classes,  weights=weights, 
                    progress=progress, **kwargs)

def load_victim_model(model_name:str, num_classes:int,  weights: str = "default",  progress: bool = True, **kwargs: Any):
    return load_models(model_name, num_classes=num_classes,  weights=weights, 
                    progress=progress, **kwargs)