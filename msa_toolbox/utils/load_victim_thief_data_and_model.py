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
from . loss_criterion import get_loss_criterion
from . optimizer import get_optimizer
from . load_data_and_models import load_thief_dataset, load_victim_dataset, get_data_loader, load_custom_dataset
from . load_data_and_models import load_thief_model, load_victim_model
from . cfg_reader import load_cfg, CfgNode
from . train_utils import accuracy_f1_precision_recall, agreement
from . active_learning_train import train_one_epoch


def load_victim_data_and_model(cfg: CfgNode):
    '''
    Loads the victim dataset and model
    '''
    victim_data, num_class = load_victim_data(cfg, None)

    # Load victim model    
    if len(cfg.VICTIM.ARCHITECTURE.split('/')) > 1 or len(cfg.VICTIM.ARCHITECTURE.split('\\'[0])) > 1:
        victim_model = torch.load(cfg.VICTIM.ARCHITECTURE)
    else:
        victim_model = load_victim_model(
            cfg.VICTIM.ARCHITECTURE, num_classes=num_class, weights=cfg.VICTIM.WEIGHTS, progress=False)
        
    # Load Victim Dataset
    victim_data, num_class = load_victim_data(cfg, victim_model)

    log_victim_data(cfg.LOG_DEST, victim_data, num_class)
    log_victim_data(cfg.INTERNAL_LOG_PATH, victim_data, num_class)

    # Load Victim Data Loader
    victim_data_loader = get_data_loader(
        victim_data, batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=True, num_workers=cfg.NUM_WORKERS)

    # Load victim model weights if present, else train the victim model (just to measure performance)
    if cfg.VICTIM.WEIGHTS is not None and cfg.VICTIM.WEIGHTS != 'None' and (len(cfg.VICTIM.WEIGHTS.split('\\'[0])) > 1 or len(cfg.VICTIM.WEIGHTS.split('/')) > 1):
        victim_model.load_state_dict(torch.load(cfg.VICTIM.WEIGHTS)['state_dict'])
        log_weights(cfg.LOG_DEST, cfg.VICTIM.WEIGHTS)
        log_weights(cfg.INTERNAL_LOG_PATH, cfg.VICTIM.WEIGHTS)

    else:
        optimizer = get_optimizer(cfg.TRAIN.OPTIMIZER, victim_model,
                                lr=cfg.TRAIN.LR, weight_decay=cfg.TRAIN.WEIGHT_DECAY)
        criteria = get_loss_criterion(cfg.TRAIN.LOSS_CRITERION)

        train_one_epoch(cfg, victim_model, victim_data_loader,
                        0, optimizer, criteria, verbose=False)

    # Measure performance of victim model on victim dataset
    metrics = accuracy_f1_precision_recall(
        victim_model, victim_data_loader, cfg.DEVICE)

    log_victim_metrics(cfg.LOG_DEST, metrics)
    log_victim_metrics(cfg.INTERNAL_LOG_PATH, metrics)

    return (victim_data, num_class), victim_data_loader, victim_model



def load_victim_data(cfg: CfgNode, victim_model: nn.Module):
    '''
    Load Victim Dataset
    '''
    if cfg.VICTIM.DATASET.lower() == 'custom_dataset':
        transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor()
        ])
        transform = transform if victim_model is None else victim_model.transforms
        victim_data = load_custom_dataset(
            root_dir=cfg.VICTIM.DATASET_ROOT, transform=transform)
    else:
        transform = True if victim_model is None else victim_model.transforms
        victim_data = load_victim_dataset(
            cfg.VICTIM.DATASET, cfg, train=False, transform=transform, download=True)
    num_class = len(victim_data.classes)
    return victim_data, num_class


def create_thief_loaders(cfg: CfgNode, victim_model: nn.Module, thief_data: Dataset,
                         labeled_indices: np.ndarray, unlabeled_indices: np.ndarray,
                         val_indices: np.ndarray):
    '''
    Creates the loaders for the thief model
    '''
    train_loader = get_data_loader(Subset(
        thief_data, labeled_indices), batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=True, num_workers=cfg.NUM_WORKERS)
    val_loader = get_data_loader(Subset(
        thief_data, val_indices), batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=True, num_workers=cfg.NUM_WORKERS)
    unlabeled_loader = get_data_loader(Subset(
        thief_data, unlabeled_indices), batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=True, num_workers=cfg.NUM_WORKERS)
    dataloader = {'train': train_loader,
                  'val': val_loader, 'unlabeled': unlabeled_loader}
    return dataloader


def change_thief_loader_labels(cfg: CfgNode, data_loader: DataLoader, victim_model: nn.Module):
    '''
    Changes the labels of the thief dataset to the labels predicted by the victim model
    '''
    victim_model = victim_model.to(cfg.DEVICE)
    victim_model.eval()
    with torch.no_grad():
        new_labels = torch.tensor([])
        for image, label in data_loader:
            image, label = image.to(cfg.DEVICE), label.to(cfg.DEVICE)
            outputs = victim_model(image)
            _, predicted = torch.max(outputs, 1)
            new_labels = torch.cat((new_labels, predicted.cpu()), dim=0)        
    return new_labels


def log_victim_data(path: str, victim_data: Dataset, num_class: int):
    with open(os.path.join(path, 'log.txt'), 'a') as f:
        f.write(f"Loaded Victim Datset of size {len(victim_data)} with {num_class} classes\n")


def log_victim_metrics(path: str, metrics: Dict[str, float]):
    with open(os.path.join(path, 'log.txt'), 'a') as f:
        f.write(f"Metrics of Victim Model on Victim Dataset: {metrics}\n")
        
def log_weights(path: str, weights: str):
    with open(os.path.join(path, 'log.txt'), 'a') as f:
        f.write(f"Loaded Victim Model weights from '{weights}'\n")