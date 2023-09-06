import os
import torch
import random
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torch.nn as nn
import torch
import json
import numpy as np
from .active_learning.active_learning_main import active_learning
from .utils.image.load_victim_thief_data_and_model import load_victim_data_and_model
from .utils.image.cfg_reader import load_cfg
from .utils.image.all_logs import log_victim_data_model, log_start_active_learning, log_finish_active_learning


def app(cfg_path):
    '''
    Main function to run the application
    '''
    # os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    # os.environ["CUDA_VISIBLE_DEVICES"] = "3"

    # Config file
    cfg = load_cfg(cfg_path)
    log_start_active_learning(cfg.LOG_PATH)
    log_start_active_learning(cfg.INTERNAL_LOG_PATH)

    # Load victim data and model
    (victim_data, num_class), victim_data_loader, victim_model = load_victim_data_and_model(cfg)

    log_victim_data_model(cfg.LOG_PATH, victim_data, victim_model, cfg.VICTIM.ARCHITECTURE)
    log_victim_data_model(cfg.INTERNAL_LOG_PATH, victim_data, victim_model, cfg.VICTIM.ARCHITECTURE)
    
    # Start Active Learning
    active_learning(cfg, victim_data_loader, num_class, victim_model)

    log_finish_active_learning(cfg.LOG_PATH)
    log_finish_active_learning(cfg.INTERNAL_LOG_PATH)



if __name__ == "__main__":
    cfg_path = input("Enter the path to the config file: ")
    app(cfg_path)
