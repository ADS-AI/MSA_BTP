import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from ...utils.image.cfg_reader import CfgNode
from ...utils.image.load_data_and_models import get_data_loader


'''
Function to be used when either:
Victim has no defense mechansim to use OR model stealing attack is not detected by the defense mechanism 
'''
def label_samples_with_no_defence(cfg:CfgNode, victim_model:nn.Module, thief_data:Dataset, 
            next_training_samples_indices:np.array, take_action:bool=False):
    '''
    Labels the new thief training samples using the victim model
    '''
    addendum_loader = get_data_loader(Subset(thief_data, next_training_samples_indices), 
                        batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=False, num_workers=cfg.NUM_WORKERS)
    victim_model.eval()
    victim_model = victim_model.to(cfg.DEVICE)
    with torch.no_grad():
        for img, label0, index in addendum_loader:
            img = img.to(cfg.DEVICE)
            label = victim_model(img)
            if cfg.TRAIN.BLACKBOX_TRAINING == True:
                label = torch.argmax(label, dim=1)
                label = label.detach().cpu().tolist()
            else:
                label = F.softmax(label, dim=1)  
                label = label.clone().detach().cpu()  
            for ii, jj in enumerate(index):
                thief_data.samples[jj] = (thief_data.samples[jj][0], label[ii])
    return
