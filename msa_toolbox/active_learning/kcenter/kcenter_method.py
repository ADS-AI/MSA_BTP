import os
import torch
import random
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import sys
from torch.utils.data import Dataset, DataLoader, Subset
from torch.nn.modules.loss import _Loss
from torch.optim.lr_scheduler import MultiStepLR
from torch.optim import Optimizer
import torch.nn.functional as F
from typing import Any, Dict, List
from ...utils.image.cfg_reader import load_cfg, CfgNode
from ...utils.image.train_utils import accuracy_f1_precision_recall, agreement
from ...utils.image.all_logs import log_training, log_finish_training, log_epoch, log_metrics_intervals, log_metrics_intervals_api
from .kcenter_greedy import KCenterGreedy


def train_kcenter(cfg: CfgNode, thief_model: nn.Module, criterion: _Loss, optimizer: Optimizer,
            scheduler:MultiStepLR, dataloader: Dict[str, DataLoader], trail_num: int, cycle_num: int, log_interval:int, *args, **kwargs):
    '''
    Trains the Thief Model on the Thief Dataset
    '''
    log_training(cfg.LOG_PATH, cfg.TRAIN.EPOCH)
    log_training(cfg.INTERNAL_LOG_PATH, cfg.TRAIN.EPOCH)
    
    exit = False
    curr_loss = None
    best_acc = None
    no_improvement = 0
    for epoch in range(cfg.TRAIN.EPOCH):
        log_epoch(cfg.LOG_PATH, epoch)
        log_epoch(cfg.INTERNAL_LOG_PATH, epoch)
        
        train_epoch_loss, train_epoch_acc = train_one_epoch(
            cfg, thief_model, dataloader['train'], epoch, optimizer, criterion)
        scheduler.step()

        metrics_thief_val = accuracy_f1_precision_recall(cfg, thief_model, dataloader['val'], cfg.DEVICE)

        if best_acc is None or metrics_thief_val['accuracy'] > best_acc:
            if os.path.isdir(cfg.OUT_DIR_MODEL) is False:
                os.makedirs(cfg.OUT_DIR_MODEL, exist_ok=True)
            best_acc = metrics_thief_val['accuracy']
            torch.save({'trail': trail_num+1, 'cycle': cycle_num+1, 'epoch': epoch+1, 'state_dict': thief_model.state_dict(
            )}, f"{cfg.OUT_DIR_MODEL}/thief_model__trial_{trail_num+1}_cycle_{cycle_num+1}.pth")
            no_improvement = 0
        else:
            no_improvement += 1
            if no_improvement == cfg.TRAIN.PATIENCE:
                exit = True
        if exit:
            break
        if (epoch + 1) % log_interval == 0:
            metrics_thief_train = accuracy_f1_precision_recall(cfg, thief_model, dataloader['train'], cfg.DEVICE)
            if cfg.VICTIM.IS_API:
                log_metrics_intervals_api(cfg.LOG_PATH, metrics_thief_train, metrics_thief_val)
                log_metrics_intervals_api(cfg.INTERNAL_LOG_PATH, metrics_thief_train, metrics_thief_val)
            else:
                metrics_victim_test = accuracy_f1_precision_recall(cfg, thief_model, dataloader['victim'], cfg.DEVICE, is_victim_loader=True)
                log_metrics_intervals(cfg.LOG_PATH, metrics_thief_train, metrics_thief_val, metrics_victim_test)
                log_metrics_intervals(cfg.INTERNAL_LOG_PATH, metrics_thief_train, metrics_thief_val, metrics_victim_test)

    log_finish_training(cfg.LOG_PATH)
    log_finish_training(cfg.INTERNAL_LOG_PATH)



def train_one_epoch(cfg: CfgNode, thief_model: nn.Module, dataloader: DataLoader, epoch: int, optimizer: Optimizer,
                    criterion: _Loss, verbose: bool = True):
    train_loss = 0
    correct = 0
    total = 0
    thief_model.train()
    thief_model = thief_model.to(cfg.DEVICE)
    f1 = open(os.path.join(cfg.LOG_PATH, 'log_tqdm.txt'), 'a', encoding="utf-8")
    f2 = open(os.path.join(cfg.INTERNAL_LOG_PATH, 'log_tqdm.txt'), 'a', encoding="utf-8")

    with tqdm(dataloader, file=sys.stdout, leave=False) as pbar:
        for images, labels, index in pbar:
            images, labels = images.to(cfg.DEVICE), labels.to(cfg.DEVICE)
            
            optimizer.zero_grad()
            outputs = thief_model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            predicted = torch.argmax(outputs.data, dim=1)
            total += labels.size(0)
            if cfg.TRAIN.BLACKBOX_TRAINING == False:
                # _, labels = torch.max(labels.data, 1)
                labels = torch.argmax(labels, dim=1)
            correct += predicted.eq(labels).sum().item()
    
            if verbose:
                f1.write(str(pbar) + '\n')
                f2.write(str(pbar) + '\n')
    f1.close()
    f2.close()
    return train_loss / len(dataloader.dataset), 100. * correct / total



def select_samples_kcenter(cfg: CfgNode, thief_model: nn.Module, thief_data:Dataset, 
            labeled_indices:List, unlabeled_indices:List, *args, **kwargs):
    '''
    Selects the samples from the unlabeled dataset to be labeled
    '''
    labeled_indices, unlabeled_indices = np.array(labeled_indices), np.array(unlabeled_indices)
    labeled_indices, unlabeled_indices = labeled_indices.tolist(), unlabeled_indices.tolist()

    # Initialize the K-Center Greedy Sampler
    sampler = KCenterGreedy(thief_model, thief_data, feature=cfg.ACTIVE.FEATURE, metric=cfg.ACTIVE.METRIC)
    # Select the samples to be labeled
    selected_index_list = sampler.select_batch(cfg, labeled_indices, unlabeled_indices, cfg.ACTIVE.ADDENDUM)
    return selected_index_list
    