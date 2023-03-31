import os
import torch
import random
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from torch.nn.modules.loss import _Loss
from torch.optim import Optimizer
from typing import Any, Dict
from . loss_criterion import get_loss_criterion
from . optimizer import get_optimizer
from . load_data_and_models import load_thief_dataset, load_victim_dataset, get_data_loader
from . load_data_and_models import load_thief_model, load_victim_model
from . cfg_reader import load_cfg, CfgNode
from . train_utils import accuracy_f1_precision_recall, agreement
from . active_learning_methods import active_learning_technique
from . train_model import train_one_epoch
from . load_victim_thief_data_and_model import load_victim_data_and_model, create_thief_loaders, change_thief_loader_labels
from . active_learning_train import train


def one_trial(cfg: CfgNode, trial_num: int, num_class: int, victim_data_loader: DataLoader, victim_model: nn.Module):
    thief_data = load_thief_dataset(
        cfg.THIEF.DATASET, train=False, transform=True, download=True)
    indices = np.arange(min(len(thief_data), cfg.THIEF.NUM_TRAIN))
    random.shuffle(indices)

    indices = indices[:cfg.THIEF.SUBSET]
    val_indices = indices[:cfg.ACTIVE.VAL]
    labeled_indices = indices[cfg.ACTIVE.VAL:cfg.ACTIVE.VAL+cfg.ACTIVE.INITIAL]
    unlabeled_indices = indices[cfg.ACTIVE.VAL+cfg.ACTIVE.INITIAL:]
    # print("Length of Dataloaders: ", len(labeled_indices), len(val_indices), len(unlabeled_indices), len(thief_data))

    dataloader = create_thief_loaders(
        cfg, victim_model, thief_data, labeled_indices, unlabeled_indices, val_indices)
    dataloader['victim'] = victim_data_loader
    # print("Length of Dataloaders: ", len(dataloader['train']), len(dataloader['val']), len(dataloader['unlabeled']), len(dataloader['victim']))

    for cycle in range(cfg.ACTIVE.CYCLES):
        print("\nCycle: ", cycle)
        print('Length of Datasets: ', 'labeled:', len(dataloader['train'].dataset), 'val:', len(
            dataloader['val'].dataset), 'unlabeled:', len(dataloader['unlabeled'].dataset))
        thief_model = load_thief_model(
            cfg.THIEF.ARCHITECTURE, num_classes=num_class, weights=cfg.THIEF.WEIGHTS, progress=False)

        optimizer = get_optimizer(cfg.TRAIN.OPTIMIZER, thief_model,
                                  lr=cfg.TRAIN.LR, weight_decay=cfg.TRAIN.WEIGHT_DECAY)
        criteria = get_loss_criterion(cfg.TRAIN.LOSS_CRITERION)

        # Train the model
        train(cfg, thief_model, criteria, optimizer,
              dataloader, trial_num, cycle, log_interval=1000)

        # print("Loading best checkpoint for Thief model")
        best_model_path = os.path.join(
            cfg.OUT_DIR, f"thief_model__trial_{trial_num+1}_cycle_{cycle+1}.pth")
        best_state = torch.load(best_model_path)['state_dict']
        thief_model.load_state_dict(best_state)

        # '''
        metrics = accuracy_f1_precision_recall(
            thief_model, dataloader['victim'], cfg.DEVICE)
        agree = agreement(thief_model, victim_model,
                          dataloader['victim'], cfg.DEVICE)
        print("Metrics and Agreement of Thief Model on Victim Dataset:", metrics, agree)
        metrics = accuracy_f1_precision_recall(
            thief_model, dataloader['val'], cfg.DEVICE)
        agree = agreement(thief_model, victim_model,
                          dataloader['val'], cfg.DEVICE)
        print("Metrics and Agreement of Thief Model on Validation Dataset:", metrics, agree)
        # '''

        if cycle == cfg.ACTIVE.CYCLES-1 or True:
            new_training_samples = active_learning_technique(
                cfg, thief_model, dataloader['unlabeled'])
            if len(new_training_samples) == 0:
                return
            new_training_samples = unlabeled_indices[new_training_samples]
            # print(len(new_training_samples), len(labeled_indices), len(unlabeled_indices))
            labeled_indices = np.concatenate(
                [labeled_indices, new_training_samples])
            labeled_indices = np.array(list(set(labeled_indices)))
            unlabeled_indices = np.array(
                list(set(unlabeled_indices) - set(new_training_samples)))
            # print(len(new_training_samples), len(labeled_indices), len(unlabeled_indices))
            # arr, cnt = np.unique(labeled_indices, return_counts=True)
            # print(len(cnt[cnt > 1]), arr[cnt > 1])
            # arr, cnt = np.unique(unlabeled_indices, return_counts=True)
            # print(len(cnt[cnt > 1]), arr[cnt > 1])
            dataloader['train'] = get_data_loader(Subset(
                thief_data, labeled_indices), batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=False, num_workers=cfg.NUM_WORKERS)
            dataloader['train'] = change_thief_loader_labels(
                cfg, dataloader['train'], victim_model)
            dataloader['unlabeled'] = get_data_loader(Subset(
                thief_data, unlabeled_indices), batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=False, num_workers=cfg.NUM_WORKERS)


def active_learning(cfg: CfgNode, victim_data_loader: DataLoader, num_class: int, victim_model: nn.Module):
    '''
    Performs active learning on the victim dataset according to the user configuration 
    '''
    for trial in range(cfg.TRIALS):
        one_trial(cfg, trial, num_class, victim_data_loader, victim_model)
