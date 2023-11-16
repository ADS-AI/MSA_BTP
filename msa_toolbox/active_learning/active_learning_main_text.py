import os
import torch
import random
import torch.nn as nn
import numpy as np
import json
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from torch.nn.modules.loss import _Loss
from torch.optim import Optimizer
from typing import Any, Dict
from torchvision import transforms
from ..utils.text.load_data_and_models import load_untrained_model
from ..utils.image.cfg_reader import load_cfg, CfgNode
from ..datasets.text import load_existing_dataset
from ..utils.text.stealing_methods import active_learning_technique

def active_learning_text(cfg: CfgNode,  victim_dataset, victim_model, victim_tokenizer, victim_config):
    '''
    Performs active learning on the victim dataset according to the user configuration
    '''
    thief_model, thief_tokenizer, thief_config = load_untrained_model(cfg, cfg.THIEF.MAODEL)
    thief_dataset = load_existing_dataset(cfg.THIEF.DATASET)
    thief_dataset = thief_dataset()
    active_learning_technique(victim_dataset, thief_dataset, victim_model, thief_model, victim_tokenizer, thief_tokenizer, victim_config, thief_config, cfg)

    # log_thief_data_model(cfg.LOG_PATH, thief_data, model, cfg.THIEF.ARCHITECTURE)
    # log_thief_data_model(cfg.INTERNAL_LOG_PATH, thief_data, model, cfg.THIEF.ARCHITECTURE)