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
from torch.optim.lr_scheduler import MultiStepLR
import torch.nn.functional as F
from typing import Any, Dict
from scipy.stats import entropy
from ...utils.image.cfg_reader import CfgNode
from ...utils.image.all_logs import log_training, log_finish_training, log_epoch, log_metrics_intervals, log_metrics_intervals_api
from ...utils.image.train_utils import accuracy_f1_precision_recall, agreement



def train_montecarlo(cfg: CfgNode, thief_model: nn.Module, criterion: _Loss, optimizer: Optimizer,
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



# def select_samples_montecarlo(cfg: CfgNode, theif_model: nn.Module, unlabeled_loader: DataLoader, *args, **kwargs):
#     theif_model.eval()
#     theif_model = theif_model.to(cfg.DEVICE)
#     for m in theif_model.modules():
#         if m.__class__.__name__.startswith('Dropout'):
#             m.train()
            
#     uncertainty = torch.tensor([])
#     indices = torch.tensor([])
#     with torch.no_grad():
#         for ind, (images, _, _) in enumerate(unlabeled_loader):
#             images = images.to(cfg.DEVICE)
#             z=np.zeros((int(images.shape[0]), cfg.ACTIVE.K, cfg.VICTIM.NUM_CLASSES))
            
#             for i in range(cfg.ACTIVE.K):
#                 scores = theif_model(images)
#                 scores = F.softmax(scores, dim=1)
#                 z[:,i,:] = scores.detach().cpu().numpy()

#             pred_sum = np.zeros((int(images.shape[0]), cfg.VICTIM.NUM_CLASSES))
#             for i0 in range(len(z)):
#                 pred_sum[i0,:] = np.sum(z[i0],axis=0)
           
#             entropies = np.zeros((int(images.shape[0])))
#             for i0 in range(len(pred_sum)):
#                 entropies[i0] = entropy(pred_sum[i0])

#             uncertainty = torch.cat((uncertainty, torch.tensor(entropies).float()), dim=0)
#             indices = torch.cat((indices, torch.tensor(
#                 np.arange(ind*cfg.TRAIN.BATCH_SIZE, ind*cfg.TRAIN.BATCH_SIZE + images.shape[0]))), dim=0)

#     arg = np.argsort(uncertainty)
#     selected_index_list = indices[arg][-(cfg.ACTIVE.ADDENDUM):].numpy().astype('int')
#     return selected_index_list


def select_samples_montecarlo(cfg: CfgNode, theif_model: nn.Module, unlabeled_loader: DataLoader, *args, **kwargs):
    theif_model.eval()
    theif_model = theif_model.to(cfg.DEVICE)
    for m in theif_model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.train()
            
    uncertainty = []
    indices = []
    with torch.no_grad():
        for ind, (images, _, index) in enumerate(unlabeled_loader):
            images = images.to(cfg.DEVICE)
            z=np.zeros((int(images.shape[0]), cfg.ACTIVE.K, cfg.VICTIM.NUM_CLASSES))
            
            for i in range(cfg.ACTIVE.K):
                scores = theif_model(images)
                scores = F.softmax(scores, dim=1)
                z[:,i,:] = scores.detach().cpu().numpy()

            pred_sum = np.zeros((int(images.shape[0]), cfg.VICTIM.NUM_CLASSES))
            for i0 in range(len(z)):
                pred_sum[i0,:] = np.sum(z[i0],axis=0)
           
            entropies = np.zeros((int(images.shape[0])))
            for i0 in range(len(pred_sum)):
                entropies[i0] = entropy(pred_sum[i0])

            uncertainty.extend(torch.tensor(entropies).clone().detach().cpu().numpy())
            indices.extend(index.numpy())
            
    arg = np.argsort(uncertainty)
    selected_index_list = np.array(indices)[arg][-(cfg.ACTIVE.ADDENDUM):].astype('int')    
    return selected_index_list
