import os
import torch
import random
import torch.nn as nn
import json
from torch.utils.data import Dataset, DataLoader, Subset
from typing import Any, Dict


# Functions in main.py

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




# Functions in active_learning.py

def log_metrics(path:str, cycle:int, metrics_victim:Dict[str, float], agree_victim:float, metrics_thief:Dict[str, float], agree_thief:float):
    with open(os.path.join(path, 'log_metrics.json'), 'r') as f:
        old_metrics = json.load(f)
        old_metrics['Cycle_'+str(cycle+1)] = {'metrics_victim': metrics_victim, 'agreement_victim': agree_victim, 'metrics_thief': metrics_thief, 'agreement_thief': agree_thief}
    with open(os.path.join(path, 'log_metrics.json'), 'w') as f:
        json.dump(old_metrics, f)
        
    with open(os.path.join(path, 'log.txt'), 'a') as f:
        f.write("Metrics of Thief Model on Victim Dataset: " +
                str(metrics_victim) + "\n")
        f.write("Agreement of Thief Model on Victim Dataset: " +
                str(agree_victim) + "\n")
    with open(os.path.join(path, 'log.txt'), 'a') as f:
        f.write("Metrics of Thief Model on Validation Dataset: " +
                str(metrics_thief) + "\n")
        f.write("Agreement of Thief Model on Validation Dataset: " +
            str(agree_thief) + "\n")


def log_new_cycle(path:str, cycle:int, dataloader:Dict[str, DataLoader]):
    with open(os.path.join(path, 'log.txt'), 'a') as f:
        f.write("\n\n===============================> Cycle:" +
                str(cycle+1) + " <===============================\n")
        f.write("\nLength of Datasets: {labeled=" + str(len(dataloader['train'].dataset)) + ', val:' + str(len(
            dataloader['val'].dataset)) + ', unlabeled:' + str(len(dataloader['unlabeled'].dataset)) + '}' + '\n')
    with open(os.path.join(path, 'log_tqdm.txt'), 'a') as f:
        f.write("Cycle:" +str(cycle+1) + "\n")

def log_calculating_metrics(path:str):
    with open(os.path.join(path, 'log.txt'), 'a') as f:
        f.write("Calculating Metrics and Agreement on Victim and Thief Datasets\n")

def log_thief_data_model(path: str, thief_data, thief_model, thief_model_name:str):
    log_dest = os.path.join(path, 'log.txt')
    with open(log_dest, 'a') as f:
        f.write('\n======================================> Thief Data and Model Loaded <======================================\n')
        f.write(f"Thief Data: {thief_data}\n")
        f.write(f"\nThief Model: {type(thief_model)}: {thief_model_name}\n")

    


# Functions in active_learning_train.py

def log_training(path: str, total_epoch:int):
    with open(os.path.join(path, 'log.txt'), 'a') as f:
        f.write(f"Training Thief Model on Thief Dataset with {total_epoch} epochs\n")
        

def log_finish_training(path:str):
    with open(os.path.join(path, 'log.txt'), 'a') as f:
        f.write("Training Completed\n\n")
    
def log_epoch(path:str, epoch:int):
    with open(os.path.join(path, 'log.txt'), 'a') as f:
        f.write(f"Epoch {epoch+1} started\n")
    with open(os.path.join(path, 'log_tqdm.txt'), 'a') as f:
        f.write(f"Epoch:{epoch+1}\n")
        
def log_epoch_vaal(path:str, iter:int, total_iter:int):
    with open(os.path.join(path, 'log.txt'), 'a') as f:
        f.write(f"VAAL Iteration {iter+1}/{total_iter} started\n")
    with open(os.path.join(path, 'log_tqdm.txt'), 'a') as f:
        f.write(f"VAAL Iteration:{iter+1}/{total_iter}\n")
        


# Functions in load_victim_thief_data_and_model.py

def log_victim_data(path: str, victim_data: Dataset, num_class: int):
    with open(os.path.join(path, 'log.txt'), 'a') as f:
        f.write(f"Loaded Victim Datset of size {len(victim_data)} with {num_class} classes\n")


def log_victim_metrics(path: str, metrics: Dict[str, float]):
    with open(os.path.join(path, 'log.txt'), 'a') as f:
        f.write(f"Metrics of Victim Model on Victim Dataset: {metrics}\n")
        
def log_weights(path: str, weights: str):
    with open(os.path.join(path, 'log.txt'), 'a') as f:
        f.write(f"Loaded Victim Model weights from '{weights}'\n")