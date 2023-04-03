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
    elif cfg.ACTIVE.METHOD == "montecarlo":
        pass


def entropy_technique(cfg: CfgNode, theif_model: nn.Module, unlabeled_loader: DataLoader):
    theif_model.eval()
    theif_model = theif_model.to(cfg.DEVICE)
    uncertainty = torch.tensor([])
    indices = torch.tensor([])
    # print("Calculating Entropy")
    ind = 0
    with torch.no_grad():
        for i, (images, _) in enumerate(unlabeled_loader):
            images = images.to(cfg.DEVICE)
            outputs = theif_model(images)
            prob = F.softmax(outputs, dim=1)
            entropy = -torch.sum(prob * torch.log(prob), dim=1)
            # uncertainty = torch.cat((uncertainty, torch.tensor(entropy)), dim=0)
            uncertainty = torch.cat(
                (uncertainty, entropy.clone().detach().cpu()), dim=0)
            indices = torch.cat((indices, torch.tensor(
                np.arange(i*cfg.TRAIN.BATCH_SIZE, i*cfg.TRAIN.BATCH_SIZE + images.shape[0]))), dim=0)

    arg = np.argsort(uncertainty)
    # print(indices[0], indices[-1], len(unlabeled_loader.dataset), len(indices))
    # print(uncertainty.shape, indices.shape, arg.shape)
    # print("Done with Entropy calculation")
    selected_index_list = indices[arg][-(cfg.ACTIVE.ADDENDUM):].numpy().astype('int')
    return selected_index_list
