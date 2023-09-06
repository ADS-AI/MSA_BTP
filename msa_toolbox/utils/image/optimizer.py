import torch
import torch.nn as nn
import numpy as np
from torch.optim import Optimizer
from torch.optim import SGD, Adam, Adagrad, Adadelta, Adamax


def SGD_Optimizer(model: nn.Module, lr: float = 1e-3, momentum: float = 0.9, weight_decay: float = 0.0,
                  dampening: float = 0, nesterov: bool = False, maximize: bool = False):
    '''
    PyTorch's SGD Optimizer
    '''
    return torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, dampening=dampening,
                           weight_decay=weight_decay, nesterov=nesterov, maximize=maximize)


def Adam_Optimizer(model: nn.Module, lr: float = 1e-3, betas: tuple = (0.9, 0.999), eps: float = 1e-8,
                   weight_decay: float = 0.0, amsgrad: bool = False, foreach: bool = False,
                   maximize: bool = False, capturable: bool = False, fused: bool = False):
    '''
    PyTorch's Adam Optimizer
    '''
    return torch.optim.Adam(model.parameters(), lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
                            amsgrad=amsgrad, foreach=foreach, maximize=maximize, capturable=capturable, fused=fused)


def Adagrad_Optimizer(model: nn.Module, lr: float = 1e-3, lr_decay: float = 0, weight_decay: float = 0.0,
                      eps: float = 1e-10, maximize: bool = False, foreach: bool = False):
    '''
    PyTorch's Adagrad Optimizer
    '''
    return torch.optim.Adagrad(model.parameters(), lr=lr, lr_decay=lr_decay, weight_decay=weight_decay,
                               eps=eps, maximize=maximize, foreach=foreach)


def Adadelta_Optimizer(model: nn.Module, lr: float = 1.0, rho: float = 0.9, eps: float = 1e-6,
                       weight_decay: float = 0.0, foreach: bool = False, maximize: bool = False):
    '''
    PyTorch's Adadelta Optimizer
    '''
    return torch.optim.Adadelta(model.parameters(), lr=lr, rho=rho, eps=eps, weight_decay=weight_decay,
                                foreach=foreach, maximize=maximize)


def Adamax_Optimizer(model: nn.Module, lr: float = 1e-3, betas: tuple = (0.9, 0.999), eps: float = 1e-8,
                     weight_decay: float = 0.0, foreach: bool = False, maximize: bool = False):
    '''
    PyTorch's Adamax Optimizer
    '''
    return torch.optim.Adamax(model.parameters(), lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
                              foreach=foreach, maximize=maximize)


all_optimizers = {
    'sgd': SGD_Optimizer,
    'adam': Adam_Optimizer,
    'adagrad': Adagrad_Optimizer,
    'adadelta': Adadelta_Optimizer,
    'adamax': Adamax_Optimizer,
}


def get_optimizer(optimizer_name: str, model: nn.Module, lr: float = 1e-3, weight_decay: float = 0.0, **kwargs):
    '''
    Returns the optimizer with the given name and the given parameters
    Optimizer names are case insensitive and must be one of the following:
        - sgd
        - adam
        - adagrad
        - adadelta
        - adamax
    '''
    optimizer = all_optimizers.get(optimizer_name.lower(), None)
    if optimizer is None:
        raise ValueError(f"Optimizer name {optimizer_name} is not valid.")
    return optimizer(model, lr=lr, weight_decay=weight_decay, **kwargs)
