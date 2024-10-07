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
from torch.optim.lr_scheduler import MultiStepLR
import torch.nn.functional as F
from typing import Any, Dict
from ...utils.image.cfg_reader import CfgNode
from ...utils.image.all_logs import log_training, log_finish_training, log_epoch_vaal, log_metrics_intervals, log_metrics_intervals_api
from ...utils.image.train_utils import accuracy_f1_precision_recall, agreement
from ...utils.image.optimizer import get_optimizer
from ...utils.image.loss_criterion import get_loss_criterion


vae_model = [None]
discriminator_model = [None]

def train_vaal(cfg: CfgNode, thief_model: nn.Module, criterion: _Loss, optimizer: Optimizer,
            scheduler:MultiStepLR, dataloader: Dict[str, DataLoader], trail_num: int, cycle_num: int, log_interval:int, *args, **kwargs):
    '''
    Trains the Thief Model on the Thief Dataset
    '''
    if vae_model[0] is None:
        vae_model[0] = VAE(cfg.ACTIVE.LATENT_DIM)
        discriminator_model[0] = Discriminator(cfg.ACTIVE.LATENT_DIM)
        
    # Load the VAE and Discriminator models
    vae = vae_model[0]
    discriminator = discriminator_model[0]
    vae = vae.to(cfg.DEVICE)
    discriminator = discriminator.to(cfg.DEVICE)
    
    # Load the optimizer and loss criterion
    optimizer_vae = get_optimizer('adam', vae,
                                lr=5e-6, weight_decay=cfg.TRAIN.WEIGHT_DECAY)
    optimizer_discriminator = get_optimizer('sgd', discriminator,
                                lr=5e-6, weight_decay=cfg.TRAIN.WEIGHT_DECAY)
    criterion_vae = get_loss_criterion('mse_loss')
    criterion_discriminator = get_loss_criterion('bce_loss')
    
    # use cfg.THEIF.SUBSET instead of cfg.THIEF.NUM_TRAIN because we might be using a subset of the thief dataset
    TRAIN_ITERATIONS = (cfg.THIEF.SUBSET * cfg.TRAIN.EPOCH) // cfg.TRAIN.BATCH_SIZE
    LR_CHANGE = TRAIN_ITERATIONS // 4
    LABELED_DATA = read_data(dataloader['train'])
    UNLABELED_DATA = read_data(dataloader['unlabeled'], labels=False)
    
    log_training(cfg.LOG_PATH, TRAIN_ITERATIONS)
    log_training(cfg.INTERNAL_LOG_PATH, TRAIN_ITERATIONS)
    
    thief_model.train()
    thief_model = thief_model.to(cfg.DEVICE)
        
    exit = False
    curr_loss = None
    best_acc = None
    no_improvement = 0    
    
    for iter_count in range(TRAIN_ITERATIONS):
        # print(f"iter_count: {iter_count} *************")
        log_epoch_vaal(cfg.LOG_PATH, iter_count, TRAIN_ITERATIONS)
        log_epoch_vaal(cfg.INTERNAL_LOG_PATH, iter_count, TRAIN_ITERATIONS)
        
        if iter_count != 0 and iter_count % LR_CHANGE == 0:
            for param in optimizer.param_groups:
                param['lr'] = param['lr'] / 10
                
        labeled_imgs, labels = next(LABELED_DATA)
        unlabeled_imgs = next(UNLABELED_DATA)
        labeled_imgs, labels = labeled_imgs.to(cfg.DEVICE), labels.to(cfg.DEVICE)
        unlabeled_imgs = unlabeled_imgs.to(cfg.DEVICE)
        
        # Model Step
        preds = thief_model(labeled_imgs)
        task_loss = criterion(preds, labels)
        optimizer.zero_grad()
        task_loss.backward()
        optimizer.step()

        # VAE Step
        for count in range(cfg.ACTIVE.NUM_VAE_STEPS):
            # print('vae step loading ******************')
            recon, z, mu, logvar = vae(labeled_imgs)
            unsup_loss = vae_loss(criterion_vae, labeled_imgs, recon, mu, logvar, cfg.ACTIVE.BETA)
            unlab_recon, unlab_z, unlab_mu, unlab_logvar = vae(unlabeled_imgs)
            transductive_loss = vae_loss(criterion_vae, unlabeled_imgs, 
                    unlab_recon, unlab_mu, unlab_logvar, cfg.ACTIVE.BETA)

            labeled_preds = discriminator(mu)
            unlabeled_preds = discriminator(unlab_mu)
            lab_real_preds = torch.ones(labeled_imgs.size(0))
            unlab_real_preds = torch.ones(unlabeled_imgs.size(0)) 
            lab_real_preds = lab_real_preds.unsqueeze(1)
            unlab_real_preds = unlab_real_preds.unsqueeze(1)   
            lab_real_preds = lab_real_preds.to(cfg.DEVICE)
            unlab_real_preds = unlab_real_preds.to(cfg.DEVICE)
            
            # print(min(labeled_preds), min(unlabeled_preds), max(labeled_preds), max(unlabeled_preds))
            dsc_loss = criterion_discriminator(labeled_preds, lab_real_preds) + \
                    criterion_discriminator(unlabeled_preds, unlab_real_preds)
            
            total_vae_loss = unsup_loss + transductive_loss + cfg.ACTIVE.ADVERSARY_PARAM * dsc_loss
            # print('total_vae_loss', total_vae_loss)
            optimizer_vae.zero_grad()
            total_vae_loss.backward()
            optimizer_vae.step()
            
            # sample new batch if needed to train the adversarial network
            if count < (cfg.ACTIVE.NUM_VAE_STEPS - 1):
                labeled_imgs, _ = next(LABELED_DATA)
                unlabeled_imgs = next(UNLABELED_DATA)
                labeled_imgs, labels = labeled_imgs.to(cfg.DEVICE), labels.to(cfg.DEVICE)
                unlabeled_imgs = unlabeled_imgs.to(cfg.DEVICE)
                

        # Discriminator Step
        for count in range(cfg.ACTIVE.NUM_ADV_STEPS):
            # print('discriminator step loading ******************')
            with torch.no_grad():
                _, _, mu, _ = vae(labeled_imgs)
                _, _, unlab_mu, _ = vae(unlabeled_imgs)
            
            labeled_preds = discriminator(mu)
            unlabeled_preds = discriminator(unlab_mu)
            lab_real_preds = torch.ones(labeled_imgs.size(0))
            unlab_fake_preds = torch.zeros(unlabeled_imgs.size(0))
            lab_real_preds = lab_real_preds.unsqueeze(1)
            unlab_fake_preds = unlab_fake_preds.unsqueeze(1)
            lab_real_preds = lab_real_preds.to(cfg.DEVICE)
            unlab_fake_preds = unlab_fake_preds.to(cfg.DEVICE)
            
            # print(min(labeled_preds), min(unlabeled_preds), max(labeled_preds), max(unlabeled_preds))
            dsc_loss = criterion_discriminator(labeled_preds, lab_real_preds) + \
                    criterion_discriminator(unlabeled_preds, unlab_fake_preds)
            # print('discriminator loss: ', dsc_loss)
            optimizer_discriminator.zero_grad()
            dsc_loss.backward()
            optimizer_discriminator.step()
            
            # sample new batch if needed to train the adversarial network
            if count < (cfg.ACTIVE.NUM_ADV_STEPS - 1):
                labeled_imgs, _ = next(LABELED_DATA)
                unlabeled_imgs = next(UNLABELED_DATA)
                labeled_imgs, labels = labeled_imgs.to(cfg.DEVICE), labels.to(cfg.DEVICE)
                unlabeled_imgs = unlabeled_imgs.to(cfg.DEVICE)
            
        scheduler.step()

        if (iter_count + 1) % log_interval == 0:
            metrics_thief_train = accuracy_f1_precision_recall(cfg, thief_model, dataloader['train'], cfg.DEVICE)
            metrics_thief_val = accuracy_f1_precision_recall(cfg, thief_model, dataloader['val'], cfg.DEVICE)
            if cfg.VICTIM.IS_API:
                log_metrics_intervals_api(cfg.LOG_PATH, metrics_thief_train, metrics_thief_val)
                log_metrics_intervals_api(cfg.INTERNAL_LOG_PATH, metrics_thief_train, metrics_thief_val)
            else:
                metrics_victim_test = accuracy_f1_precision_recall(cfg, thief_model, dataloader['victim'], cfg.DEVICE, is_victim_loader=True)
                log_metrics_intervals(cfg.LOG_PATH, metrics_thief_train, metrics_thief_val, metrics_victim_test)
                log_metrics_intervals(cfg.INTERNAL_LOG_PATH, metrics_thief_train, metrics_thief_val, metrics_victim_test)

            if best_acc is None or metrics_thief_val['accuracy'] > best_acc:
                if os.path.isdir(cfg.OUT_DIR_MODEL) is False:
                    os.makedirs(cfg.OUT_DIR_MODEL, exist_ok=True)
                best_acc = metrics_thief_val['accuracy']
                torch.save({'trail': trail_num, 'cycle': cycle_num, 'epoch': iter_count, 'state_dict': thief_model.state_dict(
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
    
    

def vae_loss(criterion_vae, x, recon, mu, logvar, beta):
    MSE = criterion_vae(recon, x)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    KLD = KLD * beta
    return MSE + KLD


def read_data(dataloader, labels=True):
    if labels:
        while True:
            for img, label, index in dataloader:
                yield img, label
    else:
        while True:
            for img, _, _ in dataloader:
                yield img
                
                

def select_samples_vaal(cfg: CfgNode, theif_model: nn.Module, unlabeled_loader: DataLoader):
    vae = vae_model[0]
    discriminator = discriminator_model[0]
    vae.eval()
    discriminator.eval()
    
    all_preds = []
    all_indices = []
    with torch.no_grad():
        for i, (images, _, indices) in enumerate(unlabeled_loader):
            images = images.to(cfg.DEVICE)   
            _, _, mu, _ = vae(images)
            preds = discriminator(mu)
            # all_preds.extend(preds.cpu().data.numpy())
            # all_indices.extend(indices.numpy())
            all_preds.extend(preds.cpu().data)
            all_indices.extend(indices)
    # print(all_preds)
    all_preds = torch.stack(all_preds)
    all_preds = all_preds.view(-1) 
    all_preds *= -1     # need to multiply by -1 to be able to use torch.topk
    
    # select the points which the discriminator things are the most likely to be unlabeled
    _, querry_indices = torch.topk(all_preds, int(cfg.ACTIVE.ADDENDUM))
    querry_pool_indices = np.asarray(all_indices)[querry_indices]
    return querry_pool_indices
