import os
import torch
import random
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import sys
from torch.utils.data import Dataset, DataLoader, Subset
from torch.nn.modules.loss import _Loss
from torch.optim import Optimizer
import torch.nn.functional as F
from typing import Any, Dict
from ...utils.image.cfg_reader import load_cfg, CfgNode
from ...utils.image.train_utils import accuracy_f1_precision_recall, agreement
from ...utils.image.all_logs import log_training, log_finish_training, log_epoch, log_metrics_intervals


def train_entropy(cfg: CfgNode, thief_model: nn.Module, victim_model: nn.Module, criterion: _Loss, optimizer: Optimizer,
          dataloader: Dict[str, DataLoader], trail_num: int, cycle_num: int, log_interval=1000):
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
            cfg, thief_model, victim_model, dataloader['train'], epoch, optimizer, criterion)

        metrics_thief_val = accuracy_f1_precision_recall(
            thief_model, victim_model, dataloader['val'], cfg.DEVICE, is_thief_set=True)

        if best_acc is None or metrics_thief_val['accuracy'] > best_acc:
            if os.path.isdir(cfg.OUT_DIR_MODEL) is False:
                os.makedirs(cfg.OUT_DIR_MODEL, exist_ok=True)
            best_acc = metrics_thief_val['accuracy']
            torch.save({'trail': trail_num, 'cycle': cycle_num, 'epoch': epoch, 'state_dict': thief_model.state_dict(
            )}, f"{cfg.OUT_DIR_MODEL}/thief_model__trial_{trail_num+1}_cycle_{cycle_num+1}.pth")
            no_improvement = 0
        else:
            no_improvement += 1
            if no_improvement == cfg.TRAIN.PATIENCE:
                exit = True
        if exit:
            break
        if (epoch + 1) % log_interval == 0:
            metrics_victim_test = accuracy_f1_precision_recall(
                    thief_model, victim_model, dataloader['victim'], cfg.DEVICE, is_thief_set=False)
            agree_victim_test = agreement(thief_model, victim_model,
                              dataloader['victim'], cfg.DEVICE)
            agree_thief_val = agreement(thief_model, victim_model,
                              dataloader['val'], cfg.DEVICE)
            log_metrics_intervals(cfg.LOG_PATH, metrics_victim_test, agree_victim_test, metrics_thief_val, agree_thief_val)
            log_metrics_intervals(cfg.INTERNAL_LOG_PATH, metrics_victim_test, agree_victim_test, metrics_thief_val, agree_thief_val)

    log_finish_training(cfg.LOG_PATH)
    log_finish_training(cfg.INTERNAL_LOG_PATH)



def train_one_epoch(cfg: CfgNode, thief_model: nn.Module, victim_model: nn.Module, dataloader: DataLoader, epoch: int, optimizer: Optimizer,
                    criterion: _Loss, verbose: bool = True):
    train_loss = 0
    correct = 0
    total = 0
    thief_model.train()
    thief_model = thief_model.to(cfg.DEVICE)
    if victim_model is not None:
        victim_model.eval()
        victim_model = victim_model.to(cfg.DEVICE)
        
    f1 = open(os.path.join(cfg.LOG_PATH, 'log_tqdm.txt'), 'a', encoding="utf-8")
    f2 = open(os.path.join(cfg.INTERNAL_LOG_PATH, 'log_tqdm.txt'), 'a', encoding="utf-8")

    with tqdm(dataloader, file=sys.stdout, leave=False) as pbar:
        for inputs, targets in pbar:
            inputs, targets = inputs.to(cfg.DEVICE), targets.to(cfg.DEVICE)
            if victim_model is not None:
                targets = victim_model(inputs)
                if cfg.TRAIN.BLACKBOX_TRAINING == True:
                    _, targets = torch.max(targets.data, 1)
                else:
                    targets = F.softmax(targets, dim=1)    
            optimizer.zero_grad()
            outputs = thief_model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            if cfg.TRAIN.BLACKBOX_TRAINING == False:
                _, targets = torch.max(targets.data, 1)
            correct += predicted.eq(targets).sum().item()
    
            if verbose:
                f1.write(str(pbar) + '\n')
                f2.write(str(pbar) + '\n')
    f1.close()
    f2.close()
    return train_loss / len(dataloader.dataset), 100. * correct / total



def select_samples_entropy(cfg: CfgNode, theif_model: nn.Module, unlabeled_loader: DataLoader):
    theif_model.eval()
    theif_model = theif_model.to(cfg.DEVICE)
    uncertainty = torch.tensor([])
    indices = torch.tensor([])
    with torch.no_grad():
        for i, (images, _) in enumerate(unlabeled_loader):
            images = images.to(cfg.DEVICE)
            outputs = theif_model(images)
            prob = F.softmax(outputs, dim=1)
            entropy = -torch.sum(prob * torch.log(prob), dim=1)
            uncertainty = torch.cat(
                (uncertainty, entropy.clone().detach().cpu()), dim=0)
            indices = torch.cat((indices, torch.tensor(
                np.arange(i*cfg.TRAIN.BATCH_SIZE, i*cfg.TRAIN.BATCH_SIZE + images.shape[0]))), dim=0)

    arg = np.argsort(uncertainty)
    selected_index_list = indices[arg][-(cfg.ACTIVE.ADDENDUM):].numpy().astype('int')
    return selected_index_list

    # indices = [0, 1, 2, 3, 4]
    # uncertainty = [6.6, 5.6, 9.9, 2.2, 8.3]
    # arg = [3, 1, 0, 4, 2]
    # indices[arg] = [3, 1, 0, 4, 2]