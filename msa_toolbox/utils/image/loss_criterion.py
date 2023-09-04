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


all_loss_criterion = {
    'l1_loss': L1Loss_Criterion,
    'mse_loss': MSELoss_Criterion,
    'cross_entropy_loss': CrossEntropyLoss_Criterion,
    'nll_loss': NLLLoss_Criterion,
    'ctc_loss': CTCLoss_Criterion,
    'poisson_nll_loss': PoissonNLLLoss_Criterion,
    'gaussian_nll_loss': GaussianNLLLoss_Criterion,
    'bce_loss': BCELoss_Criterion,
    'soft_margin_loss': SoftMarginLoss_Criterion,
    'multi_label_soft_margin_loss': MultiLabelSoftMarginLoss_Criterion
}


def get_loss_criterion(loss_criterion_name: str, **kwargs):
    '''
    Returns the loss criterion function based on the loss criterion name
    Loss criterion name must be one of the following:
        - l1_loss
        - mse_loss
        - cross_entropy_loss
        - nll_loss
        - ctc_loss
        - poisson_nll_loss
        - gaussian_nll_loss
        - bce_loss
        - soft_margin_loss
        - multi_label_soft_margin_loss
    '''
    if loss_criterion_name in all_loss_criterion:
        return all_loss_criterion[loss_criterion_name](**kwargs)
    else:
        raise ValueError(f"Loss criterion {loss_criterion_name} not found")
