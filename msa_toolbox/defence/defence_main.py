import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from ..utils.image.cfg_reader import CfgNode
from ..utils.image.load_data_and_models import get_data_loader
from .no_defence.no_defence import label_samples_with_no_defence
from .prada.prada import prada_defence
from .adaptive_misinformation.adaptive_misinformation import adaptive_misinformation_defence



def query_victim_for_new_labels(cfg:CfgNode, victim_model:nn.Module, thief_data:Dataset, 
            next_training_samples_indices:np.array, take_action:bool=True):

    if cfg.VICTIM.DEFENCE == False or cfg.VICTIM.DEFENCE.lower() == 'false' or cfg.VICTIM.DEFENCE.lower() == 'none':
        return label_samples_with_no_defence(cfg, victim_model, thief_data, next_training_samples_indices)
    if cfg.VICTIM.DEFENCE.lower() == 'prada':
        return prada_defence(cfg, victim_model, thief_data, next_training_samples_indices, take_action=take_action)
    elif cfg.VICTIM.DEFENCE.lower() == 'adaptive-misinformation':
        return adaptive_misinformation_defence(cfg, victim_model, thief_data, next_training_samples_indices, take_action=take_action)

