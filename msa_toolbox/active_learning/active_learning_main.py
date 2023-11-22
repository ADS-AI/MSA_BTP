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
from torch.optim.lr_scheduler import MultiStepLR
from typing import Any, Dict
from torchvision import transforms
from ..utils.image.loss_criterion import get_loss_criterion
from ..utils.image.optimizer import get_optimizer
from ..utils.image.load_data_and_models import load_thief_dataset, load_victim_dataset, get_data_loader, load_custom_dataset
from ..utils.image.load_data_and_models import load_thief_model, load_victim_model
from ..utils.image.cfg_reader import load_cfg, CfgNode
from ..utils.image.train_utils import accuracy_f1_precision_recall, agreement, data_distribution
from .active_learning_methods import train_active_learning, select_samples_active_learning
from ..utils.image.load_victim_thief_data_and_model import load_victim_data_and_model
from ..utils.image.all_logs import log_thief_data_model, log_new_cycle, log_metrics, log_calculating_metrics, log_active_learning_trail_start, log_data_distribution, log_metrics_before_training
from ..defence.defence_main import query_victim_for_new_labels
from ..defence.no_defence.no_defence import label_samples_with_no_defence


def one_trial(cfg: CfgNode, trial_num: int, num_class: int, victim_data_loader: DataLoader,
              victim_model: nn.Module, thief_data: Dataset):
    indices = np.arange(min(len(thief_data), cfg.THIEF.NUM_TRAIN))
    random.shuffle(indices)

    indices = indices[:cfg.THIEF.SUBSET]
    val_indices = indices[:cfg.ACTIVE.VAL]
    labeled_indices = indices[cfg.ACTIVE.VAL:cfg.ACTIVE.VAL+cfg.ACTIVE.INITIAL]
    unlabeled_indices = indices[cfg.ACTIVE.VAL+cfg.ACTIVE.INITIAL:]

    dataloader = create_thief_loaders(
        cfg, victim_model, thief_data, labeled_indices, unlabeled_indices, val_indices)
    dataloader['victim'] = victim_data_loader

    for cycle in range(cfg.ACTIVE.CYCLES):
        log_new_cycle(cfg.LOG_PATH, cycle, dataloader)
        log_new_cycle(cfg.INTERNAL_LOG_PATH, cycle, dataloader)
        
        # LOG THE DATA DISTRIBUTION OF THE DATASETS 
        dist_val = data_distribution(cfg, dataloader['val'])
        dist_train = data_distribution(cfg, dataloader['train'])
        log_data_distribution(cfg.LOG_PATH, dist_train, 'train')
        log_data_distribution(cfg.LOG_PATH, dist_val, 'validation')
        log_data_distribution(cfg.INTERNAL_LOG_PATH, dist_train, 'train')
        log_data_distribution(cfg.INTERNAL_LOG_PATH, dist_val, 'validation')

        # LOAD THIEF MODEL
        thief_model = load_thief_model(
            cfg.THIEF.ARCHITECTURE, num_classes=num_class, weights=cfg.THIEF.WEIGHTS, progress=False)
        
        # CALCULATE METRICS BEFORE TRAINING
        metrics_victim_test = accuracy_f1_precision_recall(cfg, thief_model, dataloader['victim'], cfg.DEVICE, is_victim_loader=True)
        metrics_thief_val = accuracy_f1_precision_recall(cfg, thief_model, dataloader['val'], cfg.DEVICE)
        agree_victim_test = agreement(thief_model, victim_model, dataloader['victim'], cfg.DEVICE)
        agree_thief_val = agreement(thief_model, victim_model, dataloader['val'], cfg.DEVICE)
        log_metrics_before_training(cfg.LOG_PATH, metrics_victim_test, agree_victim_test, metrics_thief_val, agree_thief_val)
        log_metrics_before_training(cfg.INTERNAL_LOG_PATH, metrics_victim_test, agree_victim_test, metrics_thief_val, agree_thief_val)
        
        # CREATE OPTIMIZER, SCHEDULER AND CRITERIA
        optimizer = get_optimizer(cfg.TRAIN.OPTIMIZER, thief_model,
                                    lr=cfg.TRAIN.LR, weight_decay=cfg.TRAIN.WEIGHT_DECAY)
        scheduler = MultiStepLR(optimizer, milestones=cfg.TRAIN.MILESTONES, gamma=0.1)
        criteria = get_loss_criterion(cfg.TRAIN.LOSS_CRITERION)
    
        # TRAIN THE THIEF MODEL
        train_active_learning(cfg, thief_model, criteria, optimizer, scheduler,
              dataloader, trial_num, cycle, log_interval=cfg.TRAIN.LOG_INTERVAL)

        # LOAD THE BEST MODEL
        best_model_path = os.path.join(
            cfg.OUT_DIR_MODEL, f"thief_model__trial_{trial_num+1}_cycle_{cycle+1}.pth")
        best_model = torch.load(best_model_path)
        best_model_cycle, best_model_epoch = best_model['cycle'], best_model['epoch']
        best_state = best_model['state_dict']
        thief_model.load_state_dict(best_state)

        # CALCULATE METRICS AFTER TRAINING
        log_calculating_metrics(cfg.LOG_PATH, best_model_cycle, best_model_epoch)
        log_calculating_metrics(cfg.INTERNAL_LOG_PATH, best_model_cycle, best_model_epoch)
        
        metrics_victim_test = accuracy_f1_precision_recall(cfg, thief_model, dataloader['victim'], cfg.DEVICE, is_victim_loader=True)
        metrics_thief_val = accuracy_f1_precision_recall(cfg, thief_model, dataloader['val'], cfg.DEVICE)
        metrics_thief_train = accuracy_f1_precision_recall(cfg, thief_model, dataloader['train'], cfg.DEVICE)
        agree_victim_test = agreement(thief_model, victim_model, dataloader['victim'], cfg.DEVICE)
        agree_thief_val = agreement(thief_model, victim_model, dataloader['val'], cfg.DEVICE)
        agree_thief_train = agreement(thief_model, victim_model, dataloader['train'], cfg.DEVICE)
    
        log_metrics(cfg.LOG_PATH, cycle, metrics_victim_test, agree_victim_test, 
                    metrics_thief_val, agree_thief_val, metrics_thief_train, agree_thief_train)
        log_metrics(cfg.INTERNAL_LOG_PATH, cycle, metrics_victim_test, agree_victim_test, 
                    metrics_thief_val, agree_thief_val, metrics_thief_train, agree_thief_train)

        # SELECT NEW TRAINING SAMPLES USING THE ACTIVE LEARNING METHOD
        if cycle != cfg.ACTIVE.CYCLES-1:
            new_training_samples_indices = select_samples_active_learning(
                cfg, thief_model, dataloader['unlabeled'], thief_data, labeled_indices, unlabeled_indices)
            if len(new_training_samples_indices) == 0:
                return
            labeled_indices = np.concatenate(
                [labeled_indices, new_training_samples_indices])
            labeled_indices = np.array(list(set(labeled_indices)))
            unlabeled_indices = np.array(
                list(set(unlabeled_indices) - set(new_training_samples_indices)))
            
            # replace addendum labels with victim labels
            query_victim_for_new_labels(cfg, victim_model, thief_data, new_training_samples_indices, take_action=True)
            
            addendum_loader = get_data_loader(Subset(thief_data, new_training_samples_indices), 
                        batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=False, num_workers=cfg.NUM_WORKERS)
            dist = data_distribution(cfg, addendum_loader)
            log_data_distribution(cfg.LOG_PATH, dist, 'addendum')
            log_data_distribution(cfg.INTERNAL_LOG_PATH, dist, 'addendum')

            dataloader['train'] = get_data_loader(Subset(thief_data, labeled_indices), 
                        batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=False, num_workers=cfg.NUM_WORKERS)
            dataloader['unlabeled'] = get_data_loader(Subset(thief_data, unlabeled_indices), 
                        batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=False, num_workers=cfg.NUM_WORKERS)



def create_thief_loaders(cfg: CfgNode, victim_model: nn.Module, thief_data: Dataset,
                         labeled_indices: np.ndarray, unlabeled_indices: np.ndarray,
                         val_indices: np.ndarray):
    '''
    Creates the loaders for the thief model
    '''
    # replace train labels with victim labels
    query_victim_for_new_labels(cfg, victim_model, thief_data, labeled_indices, take_action=False)
    
    # replace val labels with victim labels
    label_samples_with_no_defence(cfg, victim_model, thief_data, val_indices)
                
    train_loader = get_data_loader(Subset(
        thief_data, labeled_indices), batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=False, num_workers=cfg.NUM_WORKERS)
    val_loader = get_data_loader(Subset(
        thief_data, val_indices), batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=False, num_workers=cfg.NUM_WORKERS)
    unlabeled_loader = get_data_loader(Subset(
        thief_data, unlabeled_indices), batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=False, num_workers=cfg.NUM_WORKERS)
    dataloader = {'train': train_loader, 'val': val_loader, 'unlabeled': unlabeled_loader}
    return dataloader



def active_learning(cfg: CfgNode, victim_data_loader: DataLoader, num_class: int, victim_model: nn.Module):
    '''
    Performs active learning on the victim dataset according to the user configuration
    '''
    cfg.VICTIM.NUM_CLASSES = num_class
    model = load_thief_model(cfg.THIEF.ARCHITECTURE, num_classes=num_class, weights=cfg.THIEF.WEIGHTS, progress=False)

    for trial in range(cfg.TRIALS):
        training_type = 'Black-box Training' if cfg.TRAIN.BLACKBOX_TRAINING else 'White-box Training'
        log_active_learning_trail_start(cfg.LOG_PATH, trial + 1, cfg.ACTIVE.METHOD, training_type)
        log_active_learning_trail_start(cfg.INTERNAL_LOG_PATH, trial + 1, cfg.ACTIVE.METHOD, training_type)
        
        if cfg.THIEF.DATASET.lower() == 'custom_dataset':
            thief_data = load_custom_dataset(
                root_dir=cfg.THIEF.DATASET_ROOT, transform=model.transforms)
        else:
            thief_data = load_thief_dataset(
                cfg.THIEF.DATASET, cfg, train=True, transform=model.transforms, download=True)    

        log_thief_data_model(cfg.LOG_PATH, thief_data, model, cfg.THIEF.ARCHITECTURE, cfg.ACTIVE.BUDGET)
        log_thief_data_model(cfg.INTERNAL_LOG_PATH, thief_data, model, cfg.THIEF.ARCHITECTURE, cfg.ACTIVE.BUDGET)
    
        one_trial(cfg, trial, num_class, victim_data_loader, victim_model, thief_data)
