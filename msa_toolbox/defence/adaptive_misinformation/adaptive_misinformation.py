import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from ...utils.image.cfg_reader import CfgNode
from ...utils.image.load_data_and_models import get_data_loader


'''
Function to detect model stealing attack using Adaptive Misinformation Defence
'''
def adaptive_misinformation_defence(cfg:CfgNode, victim_model:nn.Module, thief_data:Dataset, 
                next_training_samples_indices:np.array, take_action:bool=True):
    # Add your code here
    addendum_loader = get_data_loader(Subset(thief_data, next_training_samples_indices), 
                        batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=False, num_workers=cfg.NUM_WORKERS)
    victim_model.eval()
    victim_model = victim_model.to(cfg.DEVICE)
    with torch.no_grad():
        for img, label0, index in addendum_loader:
            img = img.to(cfg.DEVICE)
            output = victim_model(img)
            output_probs = output.softmax(dim=1)
            max_prob, _ = torch.max(output_probs, dim=1)
            for ii, jj in enumerate(index):
                alpha = reverse_sigmoid(max_prob[ii].item() - cfg.VICTIM.MSP_THRESHOLD)
                random = torch.randn(cfg.VICTIM.NUM_CLASSES).softmax(dim=0).to(cfg.DEVICE)
                final_probs = (1 - alpha) * output_probs[ii] + alpha * random
                if cfg.TRAIN.BLACKBOX_TRAINING == True:
                    thief_data.samples[jj] = (thief_data.samples[jj][0], torch.argmax(final_probs).item())
                else:
                    thief_data.samples[jj] = (thief_data.samples[jj][0], final_probs.clone().detach().cpu())
    return

   
def reverse_sigmoid(x):
    return 1/(1 + np.exp(1000*x))