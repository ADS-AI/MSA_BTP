import os
import torch
import random
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader, Subset
from torch.nn.modules.loss import _Loss
from torch.optim import Optimizer
from typing import Any, Dict
from . cfg_reader import load_cfg, CfgNode
from . train_utils import accuracy_f1_precision_recall, agreement
from . train_model import train_one_epoch


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
            )}, f"{cfg.OUT_DIR}/thief_model__trial_{trail_num+1}_cycle_{cycle_num+1}.pth")
            no_improvement = 0
        else:
            no_improvement += 1
            if no_improvement == cfg.TRAIN.PATIENCE:
                exit = True
        if exit:
            break
    print("Training Completed")
