import torch
import torch.nn as nn
import torch.optim as optim
import os
import sys
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from .vaal_models import VAE, Discriminator
from torch.utils.data import Dataset, DataLoader, Subset
from torch.nn.modules.loss import _Loss
from torch.optim import Optimizer
import torch.nn.functional as F
from typing import Any, Dict
from ...utils.cfg_reader import CfgNode
from ...utils.all_logs import log_training, log_finish_training, log_epoch
from ...utils.train_utils import accuracy_f1_precision_recall, agreement
from ...utils.optimizer import get_optimizer
from ...utils.loss_criterion import get_loss_criterion


def train_vaal(cfg: CfgNode, thief_model: nn.Module, victim_model: nn.Module, criterion: _Loss, optimizer: Optimizer,
          dataloader: Dict[str, DataLoader], trail_num: int, cycle_num: int, log_interval=1000):
    '''
    Trains the Thief Model on the Thief Dataset
    '''
    log_training(cfg.LOG_PATH, cfg.TRAIN.EPOCH)
    log_training(cfg.INTERNAL_LOG_PATH, cfg.TRAIN.EPOCH)
    
    vae = VAE(cfg.ACTIVE.VAE_LATENT_DIM)
    discriminator = Discriminator(cfg.ACTIVE.DISCRIMINATOR_LATENT_DIM)
    optimizer_vae = get_optimizer(cfg.ACTIVE.VAE_OPTIMIZER, vae,
                                  lr=cfg.TRAIN.LR, weight_decay=cfg.TRAIN.WEIGHT_DECAY)
    optimizer_discriminator = get_optimizer(cfg.ACTIVE.DISCRIMINATOR_OPTIMIZER, discriminator,
                                  lr=cfg.TRAIN.LR, weight_decay=cfg.TRAIN.WEIGHT_DECAY)
    criterion_vae = get_loss_criterion(cfg.ACTIVE.VAE_LOSS_CRITERION)
    criterion_discriminator = get_loss_criterion(cfg.ACTIVE.DISCRIMINATOR_LOSS_CRITERION)
    vaal = {
        'vae': vae,
        'discriminator': discriminator,
        'optimizer_vae': optimizer_vae,
        'optimizer_discriminator': optimizer_discriminator,
        'criterion_vae': criterion_vae,
        'criterion_discriminator': criterion_discriminator
    }
        
    exit = False
    curr_loss = None
    best_f1 = None
    no_improvement = 0
    for epoch in range(cfg.TRAIN.EPOCH):
        log_epoch(cfg.LOG_PATH, epoch)
        log_epoch(cfg.INTERNAL_LOG_PATH, epoch)
        
        train_epoch_loss, train_epoch_acc = train_one_epoch(
            cfg, thief_model, victim_model, vaal, dataloader['train'], epoch, optimizer, criterion)

        metrics_val = accuracy_f1_precision_recall(
            thief_model, victim_model, dataloader['val'], cfg.DEVICE, is_thief_set=True)

        if best_f1 is None or metrics_val['f1'] > best_f1:
            if os.path.isdir(cfg.OUT_DIR_MODEL) is False:
                os.makedirs(cfg.OUT_DIR_MODEL, exist_ok=True)
            best_f1 = metrics_val['f1']
            torch.save({'trail': trail_num, 'cycle': cycle_num, 'epoch': epoch, 'state_dict': thief_model.state_dict(
            )}, f"{cfg.OUT_DIR_MODEL}/thief_model__trial_{trail_num+1}_cycle_{cycle_num+1}.pth")
            no_improvement = 0
        else:
            no_improvement += 1
            if no_improvement == cfg.TRAIN.PATIENCE:
                exit = True
        if exit:
            break
    log_finish_training(cfg.LOG_PATH)
    log_finish_training(cfg.INTERNAL_LOG_PATH)
    
    

def train_one_epoch(cfg: CfgNode, thief_model: nn.Module, victim_model: nn.Module, vaal:Dict, dataloader: DataLoader, epoch: int, optimizer: Optimizer,
                    criterion: _Loss, verbose: bool = True):
    train_loss = 0
    correct = 0
    total = 0
    thief_model.train()
    thief_model = thief_model.to(cfg.DEVICE)
    vaal['vae'].train()
    vaal['vae'] = vaal['vae'].to(cfg.DEVICE)
    vaal['discriminator'].train()
    vaal['discriminator'] = vaal['discriminator'].to(cfg.DEVICE)
    
    if victim_model is not None:
        victim_model.eval()
        victim_model = victim_model.to(cfg.DEVICE)
        
    f1 = open(os.path.join(cfg.LOG_PATH, 'log_tqdm.txt'), 'a', encoding="utf-8")
    f2 = open(os.path.join(cfg.INTERNAL_LOG_PATH, 'log_tqdm.txt'), 'a', encoding="utf-8")

    with tqdm(dataloader, file=sys.stdout, leave=False) as pbar:
        for inputs, targets in pbar:
            inputs, targets = inputs.to(cfg.DEVICE), targets.to(cfg.DEVICE)

            # model step
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
            
            
            
            # vae step
            recon, z, mu, logvar = vaal['vae'](inputs)
            unsup_loss = vae_loss(inputs, recon, mu, logvar, cfg.ACTIVE.BETA)
            unlab_recon, unlab_z, unlab_mu, unlab_logvar = vae(unlabeled_imgs)
            transductive_loss = self.vae_loss(unlabeled_imgs, 
                    unlab_recon, unlab_mu, unlab_logvar, cfg.beta)
        
            labeled_preds = discriminator(mu)
            unlabeled_preds = discriminator(unlab_mu)
            
            lab_real_preds = torch.ones(inputs.size(0))
            unlab_real_preds = torch.ones(unlabeled_imgs.size(0))
            if self.args.cuda:
                lab_real_preds = lab_real_preds.cuda()
                unlab_real_preds = unlab_real_preds.cuda()

            dsc_loss = self.bce_loss(labeled_preds, lab_real_preds) + \
                    self.bce_loss(unlabeled_preds, unlab_real_preds)
            total_vae_loss = unsup_loss + transductive_loss + self.args.adversary_param * dsc_loss
            optim_vae.zero_grad()
            total_vae_loss.backward()
            optim_vae.step()
            

            
    
            if verbose:
                f1.write(str(pbar) + '\n')
                f2.write(str(pbar) + '\n')
    f1.close()
    f2.close()
    return train_loss / len(dataloader.dataset), 100. * correct / total




def vae_loss(self, x, recon, mu, logvar, beta):
    MSE = self.mse_loss(recon, x)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    KLD = KLD * beta
    return MSE + KLD


def read_data(self, dataloader, labels=True):
    if labels:
        while True:
            for img, label in dataloader:
                yield img, label
    else:
        while True:
            for img, _ in dataloader:
                yield img