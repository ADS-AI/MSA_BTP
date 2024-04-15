import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from ...utils.image.cfg_reader import CfgNode
from ...utils.image.load_data_and_models import get_data_loader
from .growing_distance_agent import GrowingDistanceAgent


prada_distance_agent = [None]

'''
Function to detect model stealing attack using PRADA Defence
'''
def prada_defence(cfg:CfgNode, victim_model:nn.Module, thief_data:Dataset, 
                next_training_samples_indices:np.array, take_action:bool=True):
    if prada_distance_agent[0] is None:
        prada_distance_agent[0] = GrowingDistanceAgent(shapiro_threshold=cfg.VICTIM.SHAPIRO_THRESHOLD, dist_metric=l2, thr_update_rule=mean_dif_std)
        print("================================ PRADA Defence: Distance Agent Initialized ================================")
    prada_gd_agent = prada_distance_agent[0]
    
    addendum_loader = get_data_loader(Subset(thief_data, next_training_samples_indices), 
                        batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=False, num_workers=cfg.NUM_WORKERS)
    
    victim_model.eval()
    victim_model = victim_model.to(cfg.DEVICE)
    with torch.no_grad():
        for img, label0, index in addendum_loader:
            img = img.to(cfg.DEVICE)
            label = victim_model(img)
            target_class = torch.argmax(label, dim=1)
            
            if cfg.TRAIN.BLACKBOX_TRAINING == True:
                label = torch.argmax(label, dim=1)
                label = label.detach().cpu().tolist()
            else:
                label = F.softmax(label, dim=1)  
                label = label.clone().detach().cpu()  
                
            for ii, jj in enumerate(index):
                attacker_present = prada_gd_agent.single_query(img[ii], target_class[ii].item())
                
                if (not attacker_present) or (not take_action):
                    thief_data.samples[jj] = (thief_data.samples[jj][0], label[ii])
                else:
                    print("================================ PRADA Defence: Attack Detected ================================")
                    if cfg.TRAIN.BLACKBOX_TRAINING == True:
                        thief_data.samples[jj] = (thief_data.samples[jj][0], np.random.randint(0, cfg.VICTIM.NUM_CLASSES))
                    else:
                        thief_data.samples[jj] = (thief_data.samples[jj][0], shuffle_max_logits(label[ii], cfg.VICTIM.NUM_CLASSES//2))
    return


def l2(a: np.ndarray, b: np.ndarray) -> float:
    l2_distance = np.sqrt(((a - b) ** 2).sum().cpu())
    return l2_distance.item()

def mean_dif_std(arr: np.ndarray) -> float:
	return arr.mean() - arr.std()


def shuffle_max_logits(logits: np.ndarray, n: int) -> np.ndarray:
	# simple defence mechanism that shuffles top n logits
	logits = logits.squeeze()
	idx = logits.argsort()[-n:][::-1]
	max_elems = logits[idx]
	np.random.shuffle(max_elems)
	for i, e in zip(idx, max_elems):
		logits[i] = e
	return logits