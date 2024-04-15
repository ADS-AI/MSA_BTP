import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from ..utils.image.cfg_reader import CfgNode
from ..utils.image.load_data_and_models import get_data_loader
from .no_defence.no_defence_clarifai_api import label_samples_with_no_defence_clarafai
from .no_defence.no_defence_huggingface_api import label_samples_with_no_defence_huggingface
# TODO: Import prada_defence_api
# from .prada.prada import prada_defence



def query_victim_for_new_labels_api(cfg:CfgNode, thief_data:Dataset, 
            next_training_samples_indices:np.array, take_action:bool=True):

    if cfg.VICTIM.DEFENCE == False or cfg.VICTIM.DEFENCE.lower() == 'false' or cfg.VICTIM.DEFENCE.lower() == 'none':
        if cfg.VICTIM.PLATFORM.lower() == 'clarifai':
            return label_samples_with_no_defence_clarafai(cfg, thief_data, next_training_samples_indices)
        elif cfg.VICTIM.PLATFORM.lower() == 'hugging-face':
            return label_samples_with_no_defence_huggingface(cfg, thief_data, next_training_samples_indices)
        else:
            raise NotImplementedError('Victim API platform not supported')
            
    if cfg.VICTIM.DEFENCE.lower() == 'prada':
        # TODO: Implement prada_defence_api
        # return prada_defence_api(cfg, thief_data, next_training_samples_indices, take_action=take_action)
        return None


