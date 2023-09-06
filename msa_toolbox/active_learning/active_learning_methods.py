import os
import torch
import random
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from ..utils.cfg_reader import CfgNode
from torch.nn.modules.loss import _Loss
from torch.optim import Optimizer
from typing import Any, Dict
from .entropy.entropy_method import train_entropy, select_samples_entropy
from .vaal.vaal_method import train_vaal


def train_active_learning(cfg: CfgNode, thief_model: nn.Module, victim_model: nn.Module, criterion: _Loss, optimizer: Optimizer,
          dataloader: Dict[str, DataLoader], trail_num: int, cycle_num: int, log_interval=1000):
    if cfg.ACTIVE.METHOD == "entropy":
        return train_entropy(cfg, thief_model, victim_model, criterion, optimizer, dataloader, trail_num, cycle_num, log_interval)
    elif cfg.ACTIVE.METHOD == "vaal":
        return train_vaal(cfg, thief_model, victim_model, criterion, optimizer, dataloader, trail_num, cycle_num, log_interval)
    elif cfg.ACTIVE.METHOD == "kcenter":
        pass
    elif cfg.ACTIVE.METHOD == "montecarlo":
        pass
    

def select_samples_active_learning(cfg: CfgNode, theif_model: nn.Module, unlabeled_loader: DataLoader):
    if cfg.ACTIVE.METHOD == "entropy":
        return select_samples_entropy(cfg, theif_model, unlabeled_loader)
    elif cfg.ACTIVE.METHOD == "vaal":
        pass
    elif cfg.ACTIVE.METHOD == "kcenter":
        pass
    elif cfg.ACTIVE.METHOD == "montecarlo":
        pass
