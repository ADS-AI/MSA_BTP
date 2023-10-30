import os
import torch
import random
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from ..utils.image.cfg_reader import CfgNode
from torch.nn.modules.loss import _Loss
from torch.optim import Optimizer
from torch.optim.lr_scheduler import MultiStepLR
from typing import Any, Dict, List
from .entropy.entropy_method import train_entropy, select_samples_entropy
from .vaal.vaal_method import train_vaal, select_samples_vaal
from .montecarlo.montecarlo_method import train_montecarlo, select_samples_montecarlo
from .random.random_method import train_random, select_samples_random
from .kcenter.kcenter_method import train_kcenter, select_samples_kcenter


def train_active_learning(cfg: CfgNode, thief_model: nn.Module, criterion: _Loss, optimizer: Optimizer,
            scheduler:MultiStepLR, dataloader: Dict[str, DataLoader], trail_num: int, cycle_num: int, log_interval=1000, *args, **kwargs):
    if cfg.ACTIVE.METHOD == "entropy":
        return train_entropy(cfg, thief_model, criterion, optimizer, scheduler, dataloader, trail_num, cycle_num, log_interval)
    elif cfg.ACTIVE.METHOD == "vaal":
        return train_vaal(cfg, thief_model, criterion, optimizer, scheduler, dataloader, trail_num, cycle_num, log_interval)
    elif cfg.ACTIVE.METHOD == "kcenter":
        return train_kcenter(cfg, thief_model, criterion, optimizer, scheduler, dataloader, trail_num, cycle_num, log_interval)
    elif cfg.ACTIVE.METHOD == "montecarlo":
        return train_montecarlo(cfg, thief_model, criterion, optimizer, scheduler, dataloader, trail_num, cycle_num, log_interval)
    elif cfg.ACTIVE.METHOD == "random":
        return train_random(cfg, thief_model, criterion, optimizer, scheduler, dataloader, trail_num, cycle_num, log_interval)
    

def select_samples_active_learning(cfg: CfgNode, theif_model: nn.Module, unlabeled_loader: DataLoader,
            thief_data:Dataset, labeled_indices:List, unlabeled_indices:List, *args, **kwargs):
    if cfg.ACTIVE.METHOD == "entropy":
        new_training_samples_indices = select_samples_entropy(cfg, theif_model, unlabeled_loader)
        return unlabeled_indices[new_training_samples_indices]
    elif cfg.ACTIVE.METHOD == "vaal":
        return select_samples_vaal(cfg, theif_model, unlabeled_loader)
    elif cfg.ACTIVE.METHOD == "kcenter":
        return select_samples_kcenter(cfg, theif_model, thief_data, labeled_indices, unlabeled_indices)
    elif cfg.ACTIVE.METHOD == "montecarlo":
        new_training_samples_indices = select_samples_montecarlo(cfg, theif_model, unlabeled_loader)
        return unlabeled_indices[new_training_samples_indices]
    elif cfg.ACTIVE.METHOD == "random":
        new_training_samples_indices = select_samples_random(cfg, theif_model, unlabeled_loader)
        return unlabeled_indices[new_training_samples_indices]
