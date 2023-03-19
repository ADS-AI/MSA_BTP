import torch
import random
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from . loss_criterion import get_loss_criterion
from . optimizer import get_optimizer
from . load_data_and_models import load_thief_dataset, load_victim_dataset, get_data_loader
from . load_data_and_models import load_thief_model, load_victim_model
from . cfg_reader import load_cfg, CfgNode
from . train_utils import get_accuracy_f1_precision_recall


def one_trial(cfg: CfgNode, victim_data_loader: DataLoader, num_class: int, victim_model: nn.Module):
    thief_data = load_thief_dataset(
        cfg.THIEF.DATASET, train=True, transform=True, download=True)
    indices = np.arange(min(len(thief_data), cfg.THIEF.MAX_SAMPLES))
    random.shuffle(indices)
    indices = indices[:cfg.THIEF.SUBSET]
    val_indices = indices[:cfg.ACTIVE.VAL]
    labeled_indices = indices[cfg.ACTIVE.VAL:cfg.ACTIVE.VAL+cfg.ACTIVE.INITIAL]
    unlabeled_indices = indices[cfg.ACTIVE.VAL+cfg.ACTIVE.INITIAL:]

    train_loader = get_data_loader(Subset(
        thief_data, labeled_indices), batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = get_data_loader(Subset(
        thief_data, val_indices), batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=True, num_workers=4)
    unlabeled_loader = get_data_loader(Subset(
        thief_data, unlabeled_indices), batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=True, num_workers=4)
    dataloader = {'train': train_loader, 'val': val_loader,
                  'unlabeled': unlabeled_loader, 'victim': victim_data_loader}

    for cycle in range(cfg.ACTIVE.CYCLES):
        one_cycle(cfg, dataloader, num_class, victim_model)


def one_cycle(cfg: CfgNode, dataloader: dict, num_class: int, victim_model: nn.Module):
    thief_model = load_thief_model(
        cfg.THIEF.ARCHITECTURE, num_classes=num_class, weights=cfg.THIEF.WEIGHTS, progress=False)

    optimizer = get_optimizer(cfg.TRAIN.OPTIMIZER, thief_model, lr=cfg.TRAIN.LR, weight_decay=cfg.TRAIN.WEIGHT_DECAY)

    
    

def active_learning(cfg: CfgNode, victim_data_loader: DataLoader, num_class: int, victim_model: nn.Module):
    for trail in range(cfg.TRIALS):
        one_trial(cfg, victim_data_loader, victim_model)


def load_victim_data_and_model(cfg: CfgNode):
    '''
    Loads the victim dataset and model
    '''
    victim_data = load_victim_dataset(
        cfg.VICTIM.DATASET, train=True, transform=True, download=True)
    num_class = len(victim_data.classes)
    print(
        f"Loaded Victim Datset of size {len(victim_data)} with {num_class} classes")

    victim_data_loader = get_data_loader(
        victim_data, batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=True, num_workers=4)

    victim_model = load_victim_model(
        cfg.VICTIM.ARCHITECTURE, num_classes=num_class, weights=cfg.VICTIM.WEIGHTS, progress=False)

    metrics = get_accuracy_f1_precision_recall(
        victim_model, victim_data_loader, cfg.DEVICE)
    print("Metrics of Victim Model on Victim Dataset:", metrics)
    return (victim_data, num_class), victim_data_loader, victim_model
