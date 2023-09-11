import torch
import torch.nn as nn
import torch.optim as optim
import os
import sys
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from torch.utils.data import Dataset, DataLoader, Subset
from torch.nn.modules.loss import _Loss
from torch.optim import Optimizer
import torch.nn.functional as F
from typing import Any, Dict
from scipy.stats import entropy
from ...utils.image.cfg_reader import CfgNode
from ...utils.image.all_logs import log_training, log_finish_training, log_epoch, log_metrics_intervals
from ...utils.image.train_utils import accuracy_f1_precision_recall, agreement



def train_montecarlo(cfg: CfgNode, thief_model: nn.Module, victim_model: nn.Module, criterion: _Loss, optimizer: Optimizer,
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
                _, targets = torch.max(targets.data, 1)
            
            optimizer.zero_grad()
            outputs = thief_model(inputs)
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



def select_samples_montecarlo(cfg: CfgNode, theif_model: nn.Module, unlabeled_loader: DataLoader):
    theif_model.eval()
    theif_model = theif_model.to(cfg.DEVICE)
    for m in theif_model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.train()
            
    uncertainty = torch.tensor([])
    indices = torch.tensor([])
    with torch.no_grad():
        for i, (images, _) in enumerate(unlabeled_loader):
            images = images.to(cfg.DEVICE)
            z=np.zeros((int(images.shape[0]), cfg.ACTIVE.K, cfg.VICTIM.NUM_CLASSES))
            
            for i in range(cfg.ACTIVE.K):
                scores = theif_model(images)
                scores = F.softmax(scores)
                z[:,i,:] = scores.detach().cpu().numpy()

            pred_sum = np.zeros((int(images.shape[0]), cfg.VICTIM.NUM_CLASSES))
            for index in range(len(z)):
                pred_sum[index,:] = np.sum(z[index],axis=0)
           
            entropies = np.zeros((int(images.shape[0])))
            for index in range(len(pred_sum)):
                entropies[index] = entropy(pred_sum[index])

            uncertainty = torch.cat((uncertainty, torch.tensor(entropies).float()), dim=0)
            indices = torch.cat((indices, torch.tensor(
                np.arange(i*cfg.TRAIN.BATCH_SIZE, i*cfg.TRAIN.BATCH_SIZE + images.shape[0]))), dim=0)

    arg = np.argsort(uncertainty)
    selected_index_list = indices[arg][-(cfg.ACTIVE.ADDENDUM):].numpy().astype('int')
    with open(os.path.join(cfg.LOG_PATH, 'log.txt'), 'a') as f:
        f.write("Length of samples selected: " + len(str(selected_index_list)) + '\n')
    return selected_index_list
