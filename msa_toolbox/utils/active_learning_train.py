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
from typing import Any, Dict
from . cfg_reader import load_cfg, CfgNode
from . train_utils import accuracy_f1_precision_recall, agreement


def train(cfg: CfgNode, thief_model: nn.Module, criterion: _Loss, optimizer: Optimizer,
          dataloader: Dict[str, DataLoader], trail_num: int, cycle_num: int, log_interval=1000):
    '''
    Trains the Thief Model on the Thief Dataset
    '''
    log_training(cfg.LOG_DEST, cfg.TRAIN.EPOCH)
    log_training(cfg.INTERNAL_LOG_PATH, cfg.TRAIN.EPOCH)
    
    exit = False
    curr_loss = None
    best_f1 = None
    no_improvement = 0
    for epoch in range(cfg.TRAIN.EPOCH):
        log_epoch(cfg.LOG_DEST, epoch)
        log_epoch(cfg.INTERNAL_LOG_PATH, epoch)
        
        train_epoch_loss, train_epoch_acc = train_one_epoch(
            cfg, thief_model, dataloader['train'], epoch, optimizer, criterion)

        metrics_val = accuracy_f1_precision_recall(
            thief_model, dataloader['val'], cfg.DEVICE)

        if best_f1 is None or metrics_val['f1'] > best_f1:
            if os.path.isdir(cfg.OUT_DIR) is False:
                os.makedirs(cfg.OUT_DIR, exist_ok=True)
            best_f1 = metrics_val['f1']
            torch.save({'trail': trail_num, 'cycle': cycle_num, 'epoch': epoch, 'state_dict': thief_model.state_dict(
            )}, f"{cfg.OUT_DIR}/thief_model__trial_{trail_num+1}_cycle_{cycle_num+1}.pth")
            no_improvement = 0
        else:
            no_improvement += 1
            if no_improvement == cfg.TRAIN.PATIENCE:
                exit = True
        if exit:
            break
    log_finish_training(cfg.LOG_DEST)
    log_finish_training(cfg.INTERNAL_LOG_PATH)


def train_one_epoch(cfg: CfgNode, model: nn.Module, dataloader: DataLoader, epoch: int, optimizer: Optimizer,
                    criterion: _Loss, verbose: bool = True):
    train_loss = 0
    correct = 0
    total = 0
    model.train()
    model = model.to(cfg.DEVICE)
    f1 = open(os.path.join(cfg.LOG_DEST, 'log_tqdm.txt'), 'a', encoding="utf-8")
    f2 = open(os.path.join(cfg.INTERNAL_LOG_PATH, 'log_tqdm.txt'), 'a', encoding="utf-8")

    with tqdm(dataloader, file=sys.stdout, leave=False) as pbar:
        for inputs, targets in pbar:
            inputs, targets = inputs.to(cfg.DEVICE), targets.to(cfg.DEVICE)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
            if verbose:
                f1.write(str(pbar) + '\n')
                f2.write(str(pbar) + '\n')
    f1.close()
    f2.close()
    return train_loss / len(dataloader.dataset), 100. * correct / total


def log_training(path: str, total_epoch:int):
    with open(os.path.join(path, 'log.txt'), 'a') as f:
        f.write(f"Training Thief Model on Thief Dataset with {total_epoch} epochs\n")
        

def log_finish_training(path:str):
    with open(os.path.join(path, 'log.txt'), 'a') as f:
        f.write("Training Completed\n\n")
    
def log_epoch(path:str, epoch:int):
    with open(os.path.join(path, 'log.txt'), 'a') as f:
        f.write(f"Epoch {epoch+1} started\n")
    with open(os.path.join(path, 'log_tqdm.txt'), 'a') as f:
        f.write(f"Epoch:{epoch+1}\n")