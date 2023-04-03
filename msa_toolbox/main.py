import os
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

    # Load victim data and model
    (victim_data, num_class), victim_data_loader, victim_model = load_victim_data_and_model(cfg)

    print("\nVictim Data:", victim_data)
    print("\nVictim Model:", type(victim_model), '\n')

    active_learning(cfg, victim_data_loader, num_class, victim_model)


if __name__ == "__main__":
    cfg_path = input("Enter the path to the config file: ")
    app(cfg_path)
    
