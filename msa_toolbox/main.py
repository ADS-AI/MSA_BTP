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
from .utils.active_learning import active_learning
from .utils.load_victim_thief_data_and_model import load_victim_data_and_model
from .utils.cfg_reader import load_cfg


def app(cfg_path):
    '''
    Main function to run the application
    '''
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "3"

    # Config file
    cfg = load_cfg(cfg_path)
    log_start_active_learning(cfg.LOG_DEST)
    log_start_active_learning(cfg.INTERNAL_LOG_PATH)

    # Load victim data and model
    (victim_data, num_class), victim_data_loader, victim_model = load_victim_data_and_model(cfg)

    log_victim_data_model(cfg.LOG_DEST, victim_data, victim_model, cfg.VICTIM.ARCHITECTURE)
    log_victim_data_model(cfg.INTERNAL_LOG_PATH, victim_data, victim_model, cfg.VICTIM.ARCHITECTURE)
    
    # Start Active Learning
    active_learning(cfg, victim_data_loader, num_class, victim_model)

    log_finish_active_learning(cfg.LOG_DEST)
    log_finish_active_learning(cfg.INTERNAL_LOG_PATH)


def log_victim_data_model(path: str, victim_data, victim_model, victim_model_name: str = None):
    log_dest = os.path.join(path, 'log.txt')
    with open(log_dest, 'a') as f:
        f.write('\n======================================> Victim Data and Model Loaded <======================================\n')
        f.write(f"Victim Data: {victim_data}\n")
        f.write(f"\nVictim Model: {type(victim_model)}: {victim_model_name}\n")
        

def log_start_active_learning(path: str):
    log_dest = os.path.join(path, 'log.txt')
    with open(log_dest, 'w') as f:
        f.write('======================================> Starting Active Learning <======================================\n\n')
        
    log_dest = os.path.join(path, 'log_tqdm.txt')
    with open(log_dest, 'w') as f:
        f.write('======================================>  Active Learning TQDM <======================================\n')
    log_dest = os.path.join(path, 'log_metrics.json')
    metrics = {}
    with open(log_dest, 'w') as f:
        json.dump(metrics, f)  

def log_finish_active_learning(path: str):
    log_dest = os.path.join(path, 'log.txt')
    with open(log_dest, 'a') as f:
        f.write('\n======================================> Active Learning Finished <======================================\n\n') 


if __name__ == "__main__":
    cfg_path = input("Enter the path to the config file: ")
    app(cfg_path)
