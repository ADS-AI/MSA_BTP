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


def one_trial(cfg: CfgNode, trial_num: int, num_class: int, victim_data_loader: DataLoader, victim_model: nn.Module):
    thief_data = load_thief_dataset(
        cfg.THIEF.DATASET, train=False, transform=True, download=True)
    indices = np.arange(min(len(thief_data), cfg.THIEF.NUM_TRAIN))
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
    print("Length of Dataloaders: ", len(dataloader['train']), len(
        dataloader['val']), len(dataloader['unlabeled']), len(dataloader['victim']))

    for cycle in range(cfg.ACTIVE.CYCLES):
        print("Cycle: ", cycle)
        thief_model = load_thief_model(
            cfg.THIEF.ARCHITECTURE, num_classes=num_class, weights=cfg.THIEF.WEIGHTS, progress=False)

        '''
        metrics = accuracy_f1_precision_recall(
            thief_model, dataloader['val'], cfg.DEVICE)
        agree = agreement(thief_model, victim_model,
                          dataloader['val'], cfg.DEVICE)
        print("Metrics and Agreement of Thief Model on Validation Dataset:", metrics, agree)

        metrics = accuracy_f1_precision_recall(
            thief_model, dataloader['victim'], cfg.DEVICE)
        agree = agreement(thief_model, victim_model,
                          dataloader['victim'], cfg.DEVICE)
        print("Metrics and Agreement of Thief Model on Victim Dataset:", metrics, agree)
        '''

        optimizer = get_optimizer(cfg.TRAIN.OPTIMIZER, thief_model,
                                  lr=cfg.TRAIN.LR, weight_decay=cfg.TRAIN.WEIGHT_DECAY)
        criteria = get_loss_criterion(cfg.TRAIN.LOSS_CRITERION)

        # Train the model
        train(cfg, thief_model, criteria, optimizer,
              dataloader, trial_num, cycle, log_interval=1000)

        print("Loading best checkpoint for Thief model")
        best_model_path = os.path.join(
            cfg.OUT_DIR, f"thief_model_{trial_num+1}_{cycle+1}.pth")
        best_state = torch.load(best_model_path)['state_dict']
        thief_model.load_state_dict(best_state)

        '''
        metrics = accuracy_f1_precision_recall(
            thief_model, dataloader['victim'], cfg.DEVICE)
        agree = agreement(thief_model, victim_model,
                          dataloader['victim'], cfg.DEVICE)
        print("Metrics and Agreement of Thief Model on Victim Dataset:", metrics, agree)
        '''
        if cycle == cfg.ACTIVE.CYCLES-1:
            new_training_samples = active_learning_technique(
                cfg, thief_model, dataloader['unlabeled'])
            print(len(new_training_samples), len(
                labeled_indices), len(unlabeled_indices))
            labeled_indices = np.concatenate(
                [labeled_indices, new_training_samples])
            print(len(new_training_samples), len(
                labeled_indices), len(unlabeled_indices))

            arr, cnt = np.unique(labeled_indices, return_counts=True)
            print(len(cnt[cnt > 1]), arr[cnt > 1])
            arr, cnt = np.unique(unlabeled_indices, return_counts=True)
            print(len(cnt[cnt > 1]), arr[cnt > 1])

            # unlabeled_indices = [i for i in unlabeled_indices if i not in new_training_samples]
            unlabeled_indices = list(
                set(unlabeled_indices) - set(new_training_samples))
            labeled_indices = list(set(labeled_indices))

            print(len(new_training_samples), len(
                labeled_indices), len(unlabeled_indices))

            dataloader['train'] = get_data_loader(Subset(
                thief_data, labeled_indices), batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=True, num_workers=4)
            dataloader['unlabeled'] = get_data_loader(Subset(
                thief_data, unlabeled_indices), batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=True, num_workers=4)


def active_learning(cfg: CfgNode, victim_data_loader: DataLoader, num_class: int, victim_model: nn.Module):
    '''
    Performs active learning on the victim dataset according to the user configuration 
    '''
    for trial in range(cfg.TRIALS):
        one_trial(cfg, trial, num_class, victim_data_loader, victim_model)


def load_victim_data_and_model(cfg: CfgNode):
    '''
    Loads the victim dataset and model
    '''
    victim_data = load_victim_dataset(
        cfg.VICTIM.DATASET, train=False, transform=True, download=True)
    num_class = len(victim_data.classes)
    print(
        f"Loaded Victim Datset of size {len(victim_data)} with {num_class} classes")

    victim_data_loader = get_data_loader(
        victim_data, batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=True, num_workers=4)

    victim_model = load_victim_model(
        cfg.VICTIM.ARCHITECTURE, num_classes=num_class, weights=cfg.VICTIM.WEIGHTS, progress=False)

    metrics = accuracy_f1_precision_recall(
        victim_model, victim_data_loader, cfg.DEVICE)
    print("Metrics of Victim Model on Victim Dataset:", metrics)
    return (victim_data, num_class), victim_data_loader, victim_model


def train(cfg: CfgNode, thief_model: nn.Module, criterion: _Loss, optimizer: Optimizer,
          dataloader: Dict[str, DataLoader], trail_num: int, cycle_num: int, log_interval=1000):
    '''
    Trains the Thief Model on the Victim Dataset
    '''
    print("Training Thief Model on Thief Dataset")
    exit = False
    curr_loss = None
    best_f1 = None
    no_improvement = 0
    for epoch in range(cfg.TRAIN.EPOCH):
        train_epoch_loss, train_epoch_acc = train_one_epoch(
            thief_model, dataloader['train'], epoch, cfg.TRAIN.BATCH_SIZE, optimizer,
            criterion, cfg.DEVICE, log_interval, verbose=False)

        metrics_val = accuracy_f1_precision_recall(
            thief_model, dataloader['val'], cfg.DEVICE)
        if best_f1 is None or metrics_val['f1'] > best_f1:
            if os.path.isdir(cfg.OUT_DIR) is False:
                os.makedirs(cfg.OUT_DIR, exist_ok=True)
            best_f1 = metrics_val['f1']
            torch.save({'trail': trail_num, 'cycle': cycle_num, 'epoch': epoch, 'state_dict': thief_model.state_dict(
            )}, f"{cfg.OUT_DIR}/thief_model_{trail_num+1}_{cycle_num+1}.pth")
            no_improvement = 0
        else:
            no_improvement += 1
            if no_improvement == cfg.TRAIN.PATIENCE:
                exit = True
        if (epoch+1) % log_interval == 0:
            metrics_train = accuracy_f1_precision_recall(
                thief_model, dataloader['train'], cfg.DEVICE)
            metrics_victim = accuracy_f1_precision_recall(
                thief_model, dataloader['victim'], cfg.DEVICE)
            print(
                f"Epoch: {epoch+1}, Train Loss: {train_epoch_loss:.4f}, Train Acc: {train_epoch_acc:.4f}, Val Acc: {metrics_val['accuracy']:.4f}, Val F1: {metrics_val['f1']:.4f}, Victim Acc: {metrics_victim['accuracy']:.4f}, Victim F1: {metrics_victim['f1']:.4f}")
        if exit:
            break
    print("Training Completed")
