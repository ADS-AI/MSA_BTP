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
        f.write(f"\n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n")
        f.write(f'===========================================> Victim Data and Model Loaded <==========================================\n')
        f.write(f"%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n\n")
        f.write(f"Victim Data: {victim_data}\n")
        f.write(f"Number of Classes: {len(victim_data.classes)}\n")
        f.write(f"Victim Model: {type(victim_model)}: {victim_model_name}\n")
        

def log_start_active_learning(path: str):
    log_dest = os.path.join(path, 'log.txt')
    with open(log_dest, 'w') as f:
        f.write(f"%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n")
        f.write(f'=============================================> Starting Active Learning <============================================\n')
        f.write(f"%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n\n")
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
        f.write(f"\n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n")
        f.write(f'=============================================> Active Learning Finished <============================================\n') 
        f.write(f"%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n\n")
        




# Functions in active_learning.py

def log_metrics(path:str, cycle:int, metrics_victim_test:Dict[str, float], agree_victim_test:float, 
                metrics_thief_val:Dict[str, float], agree_thief_val:float, 
                metrics_thief_train:Dict[str, float], agree_thief_train:float):
    with open(os.path.join(path, 'log_metrics.json'), 'r') as f:
        old_metrics = json.load(f)
        old_metrics['Cycle_'+str(cycle+1)] = {
            'metrics_victim_test': metrics_victim_test, 'agreement_victim_test': agree_victim_test,
            'metrics_thief_val': metrics_thief_val, 'agreement_thief_val': agree_thief_val,
            'metrics_thief_train': metrics_thief_train, 'agreement_thief_train': agree_thief_train
        }
    with open(os.path.join(path, 'log_metrics.json'), 'w') as f:
        json.dump(old_metrics, f)
        
    with open(os.path.join(path, 'log.txt'), 'a') as f:
        f.write("Metrics on Thief Train Dataset: " + str(metrics_thief_train) + "\n")
        f.write("Metrics on Thief Validation Dataset: " + str(metrics_thief_val) + "\n")
        f.write("Metrics on Victim Test Dataset: " + str(metrics_victim_test) + "\n")
        f.write("Agreement on Thief Train Dataset: " + str(agree_thief_train) + "\n")
        f.write("Agreement on Thief Validation Dataset: " + str(agree_thief_val) + "\n")
        f.write("Agreement on Victim Test Dataset: " + str(agree_victim_test) + "\n")

def log_metrics_intervals(path:str, metrics_thief_train:Dict[str, float], 
                metrics_thief_val:Dict[str, float], metrics_victim_test:Dict[str, float]):
    with open(os.path.join(path, 'log.txt'), 'a') as f:
        f.write("Performance of Thief Model on Datasets: \n")
        f.write("Metrics on Thief Training Dataset: " + str(metrics_thief_train) + "\n")
        f.write("Metrics on Thief Validation Dataset: " + str(metrics_thief_val) + "\n")
        f.write("Metrics on Victim Test Dataset: " + str(metrics_victim_test) + "\n")


def log_new_cycle(path:str, cycle:int, dataloader:Dict[str, DataLoader]):
    with open(os.path.join(path, 'log.txt'), 'a') as f:
        f.write("\n==============================================> Cycle:" + str(cycle+1) + " <===============================================\n")
        f.write("\nLength of Datasets: {labeled=" + str(len(dataloader['train'].dataset)) + ', val:' + str(len(
            dataloader['val'].dataset)) + ', unlabeled:' + str(len(dataloader['unlabeled'].dataset)) + '}' + '\n')
    with open(os.path.join(path, 'log_tqdm.txt'), 'a') as f:
        f.write("Cycle:" +str(cycle+1) + "\n")

def log_calculating_metrics(path:str, best_model_cycle:int, best_model_epoch:int):
    with open(os.path.join(path, 'log.txt'), 'a') as f:
        f.write("===============> Calculating Metrics and Agreement for Best Thief Model on Victim and Thief Datasets ......\n")
        f.write(f"Best Model found in Cycle {best_model_cycle} at Epoch {best_model_epoch}\n")

def log_thief_data_model(path: str, thief_data, thief_model, thief_model_name:str):
    log_dest = os.path.join(path, 'log.txt')
    with open(log_dest, 'a') as f:
        f.write('\n======================================> Thief Data and Model Loaded <======================================\n')
        f.write(f"Thief Data: {thief_data}\n")
        f.write(f"Number of Classes: {len(thief_data.classes)}\n")
        f.write(f"Thief Model: {type(thief_model)}: {thief_model_name}\n")
                
def log_active_learning_trail_start(path:str, trial_num:int, method:str, training_type:str):
    with open(os.path.join(path, 'log.txt'), 'a') as f:
        f.write(f"\n\n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n")
        f.write(f"=========================================> STARTING ACTIVE LEARNING TRIAL {trial_num} <=======================================\n")
        f.write(f"%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n")
        f.write(f"\nActive Learning Method Used: {method}\n")
        f.write(f"Training Type: {training_type}\n")
    with open(os.path.join(path, 'log_tqdm.txt'), 'a') as f:
        f.write(f"\n\n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n")
        f.write(f"=========================================> STARTING ACTIVE LEARNING TRIAL {trial_num} <=======================================\n")
        f.write(f"%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n")
        f.write(f"\nActive Learning Method Used: {method}\n")
        f.write(f"Training Type: {training_type}\n")

def log_data_distribution(path, dist:Dict[str, int], data_type:str):
    with open(os.path.join(path, 'log.txt'), 'a') as f:
        f.write(f"Data Distribution of {data_type} samples: {dist}\n")

def log_metrics_before_training(path:str, metrics_victim_test:Dict[str, float], agree_victim_test:float, 
                metrics_thief_val:Dict[str, float], agree_thief_val:float):
    with open(os.path.join(path, 'log.txt'), 'a') as f:
        f.write(f"Metrics of Thief Model before Training \n")
        f.write("Metrics on Victim Test Dataset: " + str(metrics_victim_test) + "\n")
        f.write("Metrics on Thief Validation Dataset: " + str(metrics_thief_val) + "\n")
        f.write("Agreement on Victim Test Dataset: " + str(agree_victim_test) + "\n")
        f.write("Agreement on Thief Validation Dataset: " + str(agree_thief_val) + "\n")


# Functions in active_learning_train.py

def log_training(path: str, total_epoch:int):
    with open(os.path.join(path, 'log.txt'), 'a') as f:
        f.write(f"\nTraining Thief Model on Thief Dataset (Labeled) with {total_epoch} epochs\n")

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



# Functions in KCenterGreedy.py

def log_max_dist_kcenter(path:str, max_val:float, mean_val:float, median_val:float, std_val:float):
    with open(os.path.join(path, 'log.txt'), 'a') as f:
        f.write(f"Max Distance to Cluster Center: {max_val:.3f}\n")
        f.write(f"Mean and Std. Distance to Cluster Center: {mean_val:.3f}, {std_val:.3f}\n")
        f.write(f"Median Distance to Cluster Center: {median_val:.3f}\n")
