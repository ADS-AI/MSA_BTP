import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from torch.nn.modules.loss import _Loss
from torch.optim import Optimizer
from typing import Any, Dict
from torchvision import transforms
from ...utils.text.cfg_reader import load_cfg, CfgNode
# from ...utils.text.stealing_methods import active_learning_technique
from ...utils.text.load_data_and_models import load_untrained_thief_model
from .active_methods import entropy_stealing , qbc_stealing
from .basic_method import all_data_stealing
from .k_centre_methods import k_centre_stealing , k_centre_qbc_stealing
from .simple_semi_supervised_methods import simple_semi_supervised_entropy_stealing , simple_semi_supervised_qbc_stealing

def active_learning(cfg: CfgNode,  victim_dataset,thief_dataset, victim_model, victim_tokenizer, victim_config, thief_model, thief_tokenizer, thief_config):
    '''
    Performs active learning on the victim dataset according to the user configuration
    - [1] all_data_stealing
    - [2] entropy_stealing
    - [3] qbc_stealing
    - [4] k-centre_entropy
    - [5] k-centre_qbc_stealing
    - [6] simple_semi_supervised_entropy
    - [7] simple_semi_supervised_qbc
    - [8] complex_semi_supervised_entropy
    - [9] complex_semi_supervised_qbc
    '''


    if cfg.ACTIVE.METHOD == "all_data_stealing":
        print("Performing all_data_stealing")
        all_data_stealing(cfg, victim_dataset, thief_dataset, victim_model, thief_model, victim_tokenizer, thief_tokenizer, victim_config, thief_config)
    elif cfg.ACTIVE.METHOD == "entropy_stealing":
        print("Performing entropy_stealing")
        entropy_stealing(cfg, victim_dataset, thief_dataset, victim_model, thief_model, victim_tokenizer, thief_tokenizer, victim_config, thief_config)
    elif cfg.ACTIVE.METHOD == "qbc_stealing":
        print("Performing qbc_stealing")
        thief_list_models = {}
        thief_list_configs = {}
        thief_list_tokenizers = {}
        models = cfg.ACTIVE.MODELS
        for mod in models:
            thief_list_models[mod] , thief_list_tokenizers[mod] , thief_list_configs[mod] = load_untrained_thief_model(cfg, mod)
        qbc_stealing(cfg, victim_dataset, thief_dataset, victim_model, victim_tokenizer, victim_config, thief_list_models, thief_list_tokenizers, thief_list_configs)
    elif cfg.ACTIVE.METHOD == "k-centre_stealing":
        print("Performing k-centre_stealing")
        k_centre_stealing(cfg, victim_dataset, thief_dataset, victim_model, victim_tokenizer, victim_config, thief_model, thief_tokenizer, thief_config)
    elif cfg.ACTIVE.METHOD == "k-centre_qbc_stealing":
        print("Performing k-centre_qbc_stealing")
        thief_list_models = {}
        thief_list_configs = {}
        thief_list_tokenizers = {}
        models = cfg.ACTIVE.MODELS
        for mod in models:
            thief_list_models[mod] , thief_list_tokenizers[mod] , thief_list_configs[mod] = load_untrained_thief_model(cfg, mod)
        k_centre_qbc_stealing(cfg, victim_dataset, thief_dataset, victim_model, victim_tokenizer, victim_config, thief_list_models, thief_list_tokenizers, thief_list_configs)
        pass
    elif cfg.ACTIVE.METHOD == "simple_semi_supervised_entropy":
        print("Performing simple_semi_supervised_entropy")
        simple_semi_supervised_entropy_stealing(cfg, victim_dataset, thief_dataset, victim_model, victim_tokenizer, victim_config, thief_model, thief_tokenizer, thief_config)
    elif cfg.ACTIVE.METHOD == "simple_semi_supervised_qbc":
        print("Performing simple_semi_supervised_qbc")
        thief_list_models = {}
        thief_list_configs = {}
        thief_list_tokenizers = {}
        models = cfg.ACTIVE.MODELS
        for mod in models:
            thief_list_models[mod] , thief_list_tokenizers[mod] , thief_list_configs[mod] = load_untrained_thief_model(cfg, mod)
        simple_semi_supervised_qbc_stealing(cfg, victim_dataset, thief_dataset, victim_model, victim_tokenizer, victim_config, thief_list_models, thief_list_tokenizers, thief_list_configs)
    elif cfg.ACTIVE.METHOD == "complex_semi_supervised_entropy":
        print("Performing complex_semi_supervised_entropy")
        pass
    elif cfg.ACTIVE.METHOD == "complex_semi_supervised_qbc":
        print("Performing complex_semi_supervised_qbc")
        pass

    # if cfg.ACTIVE.METHOD == "entropy_stealing":
    #     return entropy_stealing(cfg, victim_dataset, thief_dataset, victim_model, thief_model, victim_tokenizer, thief_tokenizer, victim_config, thief_config)
    # elif cfg.ACTIVE.METHOD == "all_data_stealing":
    #     return all_data_stealing(cfg, victim_dataset, thief_dataset, victim_model, thief_model, victim_tokenizer, thief_tokenizer, victim_config, thief_config)
    # elif cfg.ACTIVE.METHOD == "qbc_stealing":
    #     thief_list_models = {}
    #     thief_list_configs = {}
    #     thief_list_tokenizers = {}
    #     models = cfg.ACTIVE.MODELS
    #     for mod in models:
    #         thief_list_models[mod] , thief_list_tokenizers[mod] , thief_list_configs[mod] = load_untrained_thief_model(cfg, mod)
    #     qbc_stealing(cfg, victim_dataset, thief_dataset, victim_model, victim_tokenizer, victim_config, thief_list_models, thief_list_tokenizers, thief_list_configs)

    # active_learning_technique(cfg, victim_dataset, thief_dataset, victim_model, victim_tokenizer, victim_config, thief_model, thief_tokenizer, thief_config)

    # log_thief_data_model(cfg.LOG_PATH, thief_data, model, cfg.THIEF.ARCHITECTURE)
    # log_thief_data_model(cfg.INTERNAL_LOG_PATH, thief_data, model, cfg.THIEF.ARCHITECTURE)