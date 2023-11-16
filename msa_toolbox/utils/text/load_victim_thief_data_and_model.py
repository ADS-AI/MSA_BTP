import os
import torch
import random
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, Subset
from torch.nn.modules.loss import _Loss
from torch.optim import Optimizer
from typing import Any, Dict
from . cfg_reader import load_cfg, CfgNode
from transformers import (
    WEIGHTS_NAME,
    AdamW,
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)
from .load_data_and_models import load_victim_model 


def load_victim_model_text(cfg: CfgNode): #done
    '''
    Loads the victim dataset and model
    '''
    model , tokenizer , config = load_victim_model(cfg)
    return model , tokenizer , config
    

def load_victim_dataset(cfg: CfgNode):
    '''
    Load Victim Dataset
    '''
    pass
def create_thief_loaders(cfg: CfgNode):
    '''
    Creates the loaders for the thief model
    '''
    pass

def change_thief_loader_labels(cfg: CfgNode):
    '''
    Changes the labels of the thief dataset to the labels predicted by the victim model
    '''
    pass

