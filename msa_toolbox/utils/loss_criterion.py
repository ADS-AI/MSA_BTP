import torch
import torch.nn as nn
import numpy as np
from torch.nn.modules.loss import _Loss
from torch.nn import L1Loss, MSELoss, CrossEntropyLoss, NLLLoss, CTCLoss, PoissonNLLLoss, GaussianNLLLoss, BCELoss, SoftMarginLoss, MultiLabelSoftMarginLoss


def L1Loss_Criterion(size_average=None, reduce=None, reduction: str = 'mean'):
    '''
    PyTorch's L1Loss Criterion
    '''
    return L1Loss(size_average=size_average, reduce=reduce, reduction=reduction)


def MSELoss_Criterion(size_average=None, reduce=None, reduction: str = 'mean'):
    '''
    PyTorch's MSELoss Criterion
    '''
    return MSELoss(size_average=size_average, reduce=reduce, reduction=reduction)


def CrossEntropyLoss_Criterion(weight=None, size_average=None, ignore_index: int = -100,
                                reduce=None, reduction: str = 'mean', label_smoothing: float = 0.0):
    '''
    PyTorch's CrossEntropyLoss Criterion
    '''
    return CrossEntropyLoss(weight=weight, size_average=size_average, ignore_index=ignore_index,
                            reduce=reduce, reduction=reduction, label_smoothing=label_smoothing)


def NLLLoss_Criterion(weight=None, size_average=None, ignore_index: int = -100,
                        reduce=None, reduction: str = 'mean'):
    '''
    PyTorch's NLLLoss Criterion
    '''
    return NLLLoss(weight=weight, size_average=size_average, ignore_index=ignore_index,
                    reduce=reduce, reduction=reduction)


def CTCLoss_Criterion(blank: int = 0, reduction: str = 'mean', zero_infinity: bool = False):
    '''
    PyTorch's CTCLoss Criterion
    '''
    return CTCLoss(blank=blank, reduction=reduction, zero_infinity=zero_infinity)


def PoissonNLLLoss_Criterion(log_input: bool = True, full: bool = False, size_average=None,
                            eps: float = 1e-08, reduce=None, reduction: str = 'mean'):
    '''
    PyTorch's PoissonNLLLoss Criterion
    '''
    return PoissonNLLLoss(log_input=log_input, full=full, size_average=size_average, eps=eps, 
                        reduce=reduce, reduction=reduction)


def GaussianNLLLoss_Criterion(full: bool = False, eps: float = 1e-08, reduction: str = 'mean'):
    '''
    PyTorch's GaussianNLLLoss Criterion
    '''
    return GaussianNLLLoss(full=full, eps=eps, reduction=reduction)


def BCELoss_Criterion(weight=None, size_average=None, reduce=None, reduction: str = 'mean'):
    '''
    PyTorch's BCELoss Criterion
    '''
    return BCELoss(weight=weight, size_average=size_average, reduce=reduce, reduction=reduction)


def SoftMarginLoss_Criterion(size_average=None, reduce=None, reduction: str = 'mean'):
    '''
    PyTorch's SoftMarginLoss Criterion
    '''
    return SoftMarginLoss(size_average=size_average, reduce=reduce, reduction=reduction)


def MultiLabelSoftMarginLoss_Criterion(weight=None, size_average=None, reduce=None, reduction: str = 'mean'):
    '''
    PyTorch's MultiLabelSoftMarginLoss Criterion
    '''
    return MultiLabelSoftMarginLoss(weight=weight, size_average=size_average, reduce=reduce, reduction=reduction)
