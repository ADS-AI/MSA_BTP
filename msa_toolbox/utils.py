import datasets
import models

def load_dataset(dataset_name, cfg, train=True, download=True, transform=None):
    dataset_name = dataset_name.lower()
    if dataset_name == 'cifar10':
        return datasets.CIFAR10(cfg, train, download, transform)
    elif dataset_name == 'cifar100':
        return datasets.CIFAR100(cfg, train, download, transform)
    elif dataset_name == 'imagenet':
        return datasets.ImageNet1k(cfg, train, download, transform)
    elif dataset_name == 'mnist':
        return datasets.MNIST(cfg, train, download, transform)
    elif dataset_name == 'kmnist':
        return datasets.KMNIST(cfg, train, download, transform)
    elif dataset_name == 'fashion_mnist':
        return datasets.FashionMNIST(cfg, train, download, transform)
    elif dataset_name == 'emist':
        return datasets.EMNIST(cfg, train, download, transform)
    elif dataset_name =='emistleters':
        return datasets.EMNISTLetters(cfg, train, download, transform)
    elif dataset_name == 'svhn':
        return datasets.SVHN(cfg, train, download, transform)
    elif dataset_name == 'tiny_imagenet':
        return datasets.TinyImageNet200(cfg, train, download, transform)
    elif dataset_name == 'cubs200':
        return datasets.CUBS200(cfg, train, download, transform)
    elif dataset_name == 'diabetic5':
        return datasets.Diabetic5(train, transform)
    elif dataset_name == 'indoor67':
        return datasets.Indoor67(train, transform)
    
    else:
        raise ValueError('Unknown dataset: {}'.format(dataset_name))
    
    
def load_victim_dataset(dataset_name, cfg, train=True, download=True, transform=None):
    return load_dataset(dataset_name, cfg, train, download, transform)

def load_thief_dataset(dataset_name, cfg, train=True, download=True, transform=None):
    return load_dataset(dataset_name, cfg, train, download, transform)

def load_models(model_name, num_classes,  weights, progress):
    model_name = model_name.lower()
    if model_name == 'resnet18':
        return models.resnet.ResNet18(num_classes, weights, progress)
    elif model_name == 'resnet50':
        return models.resnet.ResNet50(num_classes, weights, progress)
    
    elif model_name == 'vgg11':
        return models.vgg.VGG11(num_classes, weights, progress)
    elif model_name == 'vgg13':
        return models.vgg.VGG13(num_classes, weights, progress)
    elif model_name == 'vgg16':
        return models.vgg.VGG16(num_classes, weights, progress)
    elif model_name == 'vgg19':
        return models.vgg.VGG19(num_classes, weights, progress)
    
    elif model_name == 'vgg11_bn':
        return models.vgg.VGG11_bn(num_classes, weights, progress)
    elif model_name == 'vgg13_bn':
        return models.vgg.VGG13_bn(num_classes, weights, progress)
    elif model_name == 'vgg16_bn':
        return models.vgg.VGG16_bn(num_classes, weights, progress)
    elif model_name == 'vgg19_bn':
        return models.vgg.VGG19_bn(num_classes, weights, progress)
    
    elif model_name == 'alexnet':
        return models.alexnet.AlexNet(num_classes, weights, progress)
    
    elif model_name == 'efficientnet_b0':
        return models.efficientnet.EfficientNet_B0(num_classes, weights, progress)
    elif model_name == 'efficientnet_b1':
        return models.efficientnet.EfficientNet_B1(num_classes, weights, progress)
    elif model_name == 'efficientnet_b2':
        return models.efficientnet.EfficientNet_B2(num_classes, weights, progress)
    elif model_name == 'efficientnet_b3':
        return models.efficientnet.EfficientNet_B3(num_classes, weights, progress)
    elif model_name == 'efficientnet_b4':
        return models.efficientnet.EfficientNet_B4(num_classes, weights, progress)
    elif model_name == 'efficientnet_b5':
        return models.efficientnet.EfficientNet_B5(num_classes, weights, progress)
    elif model_name == 'efficientnet_b6':
        return models.efficientnet.EfficientNet_B6(num_classes, weights, progress)
    elif model_name == 'efficientnet_b7':
        return models.efficientnet.EfficientNet_B7(num_classes, weights, progress)
    
    elif model_name == 'effcientnet_v2_s':
        return models.efficientnet_v2.EfficientNet_V2_S(num_classes, weights, progress)
    elif model_name == 'effcientnet_v2_m':
        return models.efficientnet_v2.EfficientNet_V2_M(num_classes, weights, progress)
    elif model_name == 'effcientnet_v2_l':
        return models.efficientnet_v2.EfficientNet_V2_L(num_classes, weights, progress)
    
    elif model_name == 'mobilenet_v2':
        return models.mobilenet_v2.MobileNet_V2(num_classes, weights, progress)
    elif model_name == 'mobilenet_v3_small':
        return models.mobilenet_v2.MobileNet_V3_Small(num_classes, weights, progress)
    elif model_name == 'mobilenet_v3_large':
        return models.mobilenet_v2.MobileNet_V3_Large(num_classes, weights, progress)
    
def load_thief_model(model_name, num_classes,  weights, progress):
    return load_models(model_name, num_classes,  weights, progress)

def load_victim_model(model_name, num_classes,  weights, progress):
    return load_models(model_name, num_classes,  weights, progress)