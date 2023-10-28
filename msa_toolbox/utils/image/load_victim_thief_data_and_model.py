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
from . all_logs import log_victim_data, log_weights, log_victim_metrics


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

    log_victim_data(cfg.LOG_PATH, victim_data, num_class)
    log_victim_data(cfg.INTERNAL_LOG_PATH, victim_data, num_class)

    # Load Victim Test Data Loader
    victim_test_loader = get_data_loader(
        victim_data, batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=True, num_workers=cfg.NUM_WORKERS)

    # Load victim model weights if present, else train the victim model (just to measure performance)
    if cfg.VICTIM.WEIGHTS is not None and cfg.VICTIM.WEIGHTS != 'None' and (len(cfg.VICTIM.WEIGHTS.split('\\'[0])) > 1 or len(cfg.VICTIM.WEIGHTS.split('/')) > 1):
        victim_model.load_state_dict(torch.load(cfg.VICTIM.WEIGHTS)['state_dict'])
        log_weights(cfg.LOG_PATH, cfg.VICTIM.WEIGHTS)
        log_weights(cfg.INTERNAL_LOG_PATH, cfg.VICTIM.WEIGHTS)

    else:
        optimizer = get_optimizer(cfg.TRAIN.OPTIMIZER, victim_model,
                                lr=cfg.TRAIN.LR, weight_decay=cfg.TRAIN.WEIGHT_DECAY)
        criteria = get_loss_criterion(cfg.TRAIN.LOSS_CRITERION)
        for i in range(min(10, cfg.TRAIN.EPOCH)):
            train_one_epoch(cfg, victim_model, victim_test_loader, optimizer, criteria, verbose=False)
        
    # Measure performance of victim model on victim dataset
    metrics = accuracy_f1_precision_recall(cfg, victim_model, victim_test_loader, cfg.DEVICE, is_victim_loader=True)

    log_victim_metrics(cfg.LOG_PATH, metrics)
    log_victim_metrics(cfg.INTERNAL_LOG_PATH, metrics)

    return (victim_data, num_class), victim_test_loader, victim_model



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



def train_one_epoch(cfg: CfgNode, model: nn.Module, dataloader: DataLoader, optimizer: Optimizer,
                    criterion: _Loss, verbose: bool = True):
    train_loss = 0
    correct = 0
    total = 0
    model.train()
    model = model.to(cfg.DEVICE)
    for images, labels, index in dataloader:
        images, labels = images.to(cfg.DEVICE), labels.to(cfg.DEVICE)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        predicted = torch.argmax(outputs.data, dim=1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    return train_loss / len(dataloader.dataset), 100. * correct / total