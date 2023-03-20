import os
import torch
import random
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from . cfg_reader import CfgNode


def active_learning_technique(cfg: CfgNode, theif_model: nn.Module, unlabeled_loader: DataLoader):
    if cfg.ACTIVE.METHOD == "entropy":
        return entropy_technique(cfg, theif_model, unlabeled_loader)
    elif cfg.ACTIVE.METHOD == "kcenter":
        pass


def entropy_technique(cfg: CfgNode, theif_model: nn.Module, unlabeled_loader: DataLoader):
    theif_model.eval()
    theif_model = theif_model.to(cfg.DEVICE)
    uncertainty = []
    indices = []
    with torch.no_grad():
        for i, (images, _) in enumerate(unlabeled_loader):
            images = images.to(cfg.DEVICE)
            outputs = theif_model(images)
            prob = F.softmax(outputs, dim=1)
            entropy = -torch.sum(prob * torch.log(prob), dim=1)
            uncertainty.append(entropy)
            indices.append(i)
    arg = np.argsort(uncertainty)
    selected_index_list = indices[arg][-(cfg.ACTIVE.ADDENDUM):].numpy().astype('int')
    return selected_index_list
