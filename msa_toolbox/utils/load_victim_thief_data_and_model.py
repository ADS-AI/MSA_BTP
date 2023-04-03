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
from . train_model import train_one_epoch


def load_victim_data_and_model(cfg: CfgNode):
    '''
    Loads the victim dataset and model
    '''
    if cfg.VICTIM.DATASET.lower() == 'custom_dataset':
        transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor()
        ])
        victim_data = load_custom_dataset(
            root_dir=cfg.VICTIM.DATA_ROOT, transform=transform)
    else:
        victim_data = load_victim_dataset(
            cfg.VICTIM.DATASET, train=False, transform=True, download=True)
    num_class = len(victim_data.classes)
    print(
        f"Loaded Victim Datset of size {len(victim_data)} with {num_class} classes")

    victim_data_loader = get_data_loader(
        victim_data, batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=True, num_workers=cfg.NUM_WORKERS)

    if len(cfg.VICTIM.ARCHITECTURE.split('/')) > 1 or len(cfg.VICTIM.ARCHITECTURE.split('\\'[0])) > 1:
        victim_model = torch.load(cfg.VICTIM.ARCHITECTURE)
    else:
        victim_model = load_victim_model(
            cfg.VICTIM.ARCHITECTURE, num_classes=num_class, weights=cfg.VICTIM.WEIGHTS, progress=False)
    if cfg.VICTIM.WEIGHTS is not None and cfg.VICTIM.WEIGHTS != 'None' and (len(cfg.VICTIM.WEIGHTS.split('\\'[0])) > 1 or len(cfg.VICTIM.WEIGHTS.split('/')) > 1):
        victim_model.load_state_dict(torch.load(cfg.VICTIM.WEIGHTS))

    # '''
    optimizer = get_optimizer(cfg.TRAIN.OPTIMIZER, victim_model,
                              lr=cfg.TRAIN.LR, weight_decay=cfg.TRAIN.WEIGHT_DECAY)
    criteria = get_loss_criterion(cfg.TRAIN.LOSS_CRITERION)
    train_one_epoch(
        victim_model, victim_data_loader, 0, cfg.TRAIN.BATCH_SIZE, optimizer,
        criteria, cfg.DEVICE, 100, verbose=False)
    # '''
    metrics = accuracy_f1_precision_recall(
        victim_model, victim_data_loader, cfg.DEVICE)
    print("Metrics of Victim Model on Victim Dataset:", metrics)
    return (victim_data, num_class), victim_data_loader, victim_model


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
    train_loader = change_thief_loader_labels(cfg, train_loader, victim_model)
    val_loader = change_thief_loader_labels(cfg, val_loader, victim_model)
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
        data_loader.dataset.labels = new_labels
    return data_loader
