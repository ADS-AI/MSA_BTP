import os
import torch
import random
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torch.nn as nn
import torch
import numpy as np
from .utils.active_learning import active_learning
from .utils.load_victim_thief_data_and_model import load_victim_data_and_model
from .utils.cfg_reader import load_cfg


def app(cfg_path):
    '''
    Main function to run the application
    '''
    # Config file
    cfg = load_cfg(cfg_path)

    log_dest = os.path.join(cfg.LOG_DEST, 'log.txt')
    with open(log_dest, 'w') as f:
        f.write('======================================> Starting Active Learning <======================================\n\n')

    # Load victim data and model
    (victim_data, num_class), victim_data_loader, victim_model = load_victim_data_and_model(cfg)

    with open(log_dest, 'a') as f:
        f.write('\n======================================> Victim Data and Model Loaded <======================================\n')
        f.write(f"Victim Data: {victim_data}\n")
        f.write(f"\nVictim Model: {type(victim_model)}\n")

    active_learning(cfg, victim_data_loader, num_class, victim_model)


if __name__ == "__main__":
    cfg_path = input("Enter the path to the config file: ")
    app(cfg_path)
