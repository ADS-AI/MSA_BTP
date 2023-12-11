print(__package__)
import os
import torch
import random
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from .active_learning.text.active_learning_main import active_learning
from .utils.text.load_data_and_models import load_victim_dataset, load_untrained_thief_model, load_victim_model, load_thief_dataset
from .utils.text.cfg_reader import load_cfg
import argparse

def app(cfg_path):
    '''
    Main function to run the application
    '''
    # os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    # os.environ["CUDA_VISIBLE_DEVICES"] = "3"

    # Config file
    cfg = load_cfg(cfg_path)
    # log_start_active_learning(cfg.LOG_PATH)
    # log_start_active_learning(cfg.INTERNAL_LOG_PATH)

    # Load victim data and model
    print("Loading victim model...") 
    victim_model , victim_tokenizer , victim_config = load_victim_model(cfg)
    print("Loading thief model...")
    thief_model, thief_tokenizer, thief_config = load_untrained_thief_model(cfg, cfg.THIEF.ARCHITECTURE)
    print("Loading victim data...")
    victim_dataset = load_victim_dataset(cfg)
    print("Loading thief data...")
    thief_dataset = load_thief_dataset(cfg)

    # log_victim_data_model(cfg.LOG_PATH, victim_data, victim_model, cfg.VICTIM.ARCHITECTURE)
    # log_victim_data_model(cfg.INTERNAL_LOG_PATH, victim_data, victim_model, cfg.VICTIM.ARCHITECTURE)
    
    # Start Active Learning
    active_learning(cfg, victim_dataset, thief_dataset, victim_model, victim_tokenizer, victim_config , thief_model, thief_tokenizer, thief_config)


    # log_finish_active_learning(cfg.LOG_PATH)
    # log_finish_active_learning(cfg.INTERNAL_LOG_PATH)



if __name__ == "__main__":
    # take input using args
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg_path', type=str, default=None)
    args = parser.parse_args()
    cfg_path = args.cfg_path
    # cfg_path = input("Enter the path to the config file: ")
    app(cfg_path)
