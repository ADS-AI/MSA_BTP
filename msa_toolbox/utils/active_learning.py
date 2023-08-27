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
from typing import Any, Dict
from torchvision import transforms
from . loss_criterion import get_loss_criterion
from . optimizer import get_optimizer
from . load_data_and_models import load_thief_dataset, load_victim_dataset, get_data_loader, load_custom_dataset
from . load_data_and_models import load_thief_model, load_victim_model
from . cfg_reader import load_cfg, CfgNode
from . train_utils import accuracy_f1_precision_recall, agreement
from . active_learning_methods import train_active_learning, select_samples_active_learning
from . train_model import train_one_epoch
from . load_victim_thief_data_and_model import load_victim_data_and_model, create_thief_loaders, change_thief_loader_labels
from . all_logs import log_thief_data_model, log_new_cycle, log_metrics, log_calculating_metrics


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

        thief_model = load_thief_model(
            cfg.THIEF.ARCHITECTURE, num_classes=num_class, weights=cfg.THIEF.WEIGHTS, progress=False)

        optimizer = get_optimizer(cfg.TRAIN.OPTIMIZER, thief_model,
                                  lr=cfg.TRAIN.LR, weight_decay=cfg.TRAIN.WEIGHT_DECAY)
        criteria = get_loss_criterion(cfg.TRAIN.LOSS_CRITERION)

        train_active_learning(cfg, thief_model, victim_model, criteria, optimizer,
              dataloader, trial_num, cycle, log_interval=1000)

        best_model_path = os.path.join(
            cfg.OUT_DIR_MODEL, f"thief_model__trial_{trial_num+1}_cycle_{cycle+1}.pth")
        best_state = torch.load(best_model_path)['state_dict']
        thief_model.load_state_dict(best_state)

        log_calculating_metrics(cfg.LOG_PATH)
        log_calculating_metrics(cfg.INTERNAL_LOG_PATH)
        
        metrics_victim = accuracy_f1_precision_recall(
            thief_model, victim_model, dataloader['victim'], cfg.DEVICE, is_thief_set=False)
        agree_victim = agreement(thief_model, victim_model,
                          dataloader['victim'], cfg.DEVICE)
        metrics_thief = accuracy_f1_precision_recall(
            thief_model, victim_model, dataloader['val'], cfg.DEVICE, is_thief_set=True)
        agree_thief = agreement(thief_model, victim_model,
                          dataloader['val'], cfg.DEVICE)
    
        log_metrics(cfg.LOG_PATH, cycle, metrics_victim, agree_victim, metrics_thief, agree_thief)
        log_metrics(cfg.INTERNAL_LOG_PATH, cycle, metrics_victim, agree_victim, metrics_thief, agree_thief)

        if cycle != cfg.ACTIVE.CYCLES-1:
            new_training_samples = select_samples_active_learning(
                cfg, thief_model, dataloader['unlabeled'])
            if len(new_training_samples) == 0:
                return
            new_training_samples = unlabeled_indices[new_training_samples]
            labeled_indices = np.concatenate(
                [labeled_indices, new_training_samples])
            labeled_indices = np.array(list(set(labeled_indices)))
            unlabeled_indices = np.array(
                list(set(unlabeled_indices) - set(new_training_samples)))
            dataloader['train'] = get_data_loader(Subset(
                thief_data, labeled_indices), batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=False, num_workers=cfg.NUM_WORKERS)
            dataloader['unlabeled'] = get_data_loader(Subset(
                thief_data, unlabeled_indices), batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=False, num_workers=cfg.NUM_WORKERS)


def active_learning(cfg: CfgNode, victim_data_loader: DataLoader, num_class: int, victim_model: nn.Module):
    '''
    Performs active learning on the victim dataset according to the user configuration
    '''
    model = load_thief_model(cfg.THIEF.ARCHITECTURE, num_classes=num_class, weights=cfg.THIEF.WEIGHTS, progress=False)

    if cfg.THIEF.DATASET.lower() == 'custom_dataset':
        transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor()
        ])
        thief_data = load_custom_dataset(
            root_dir=cfg.THIEF.DATASET_ROOT, transform=model.transforms)
    else:
        thief_data = load_thief_dataset(
            cfg.THIEF.DATASET, cfg, train=True, transform=model.transforms, download=True)

    log_thief_data_model(cfg.LOG_PATH, thief_data, model, cfg.THIEF.ARCHITECTURE)
    log_thief_data_model(cfg.INTERNAL_LOG_PATH, thief_data, model, cfg.THIEF.ARCHITECTURE)

    '''
    thief_data_loader = get_data_loader(thief_data, batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=False, num_workers=cfg.NUM_WORKERS)
    new_labels = change_thief_loader_labels(cfg, thief_data_loader, victim_model)
    
    thief_data.labels = new_labels
    thief_data.targets = new_labels
    thief_data.classes = victim_data_loader.dataset.classes
    data_name = cfg.THIEF.DATASET.lower()
    if data_name == 'custom_dataset' or data_name == 'caltech256' or data_name == 'cubs200' or data_name == 'diabetic5' or data_name == 'imagenet' or data_name == 'indoor67' or data_name == 'tinyimagesubset' or data_name == 'tinyimagenet200':
        samples = np.array(thief_data.samples, dtype=object)
        print(samples[:,1])
        samples[:,1] = torch.tensor(new_labels, dtype=torch.int64)
        print(samples[:,1])
        thief_data.samples = samples.tolist()
    '''
    for trial in range(cfg.TRIALS):
        one_trial(cfg, trial, num_class, victim_data_loader, victim_model, thief_data)
