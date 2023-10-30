import numpy as np
from sklearn.metrics import pairwise_distances
from tqdm import tqdm
import sys
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from ...utils.image.cfg_reader import CfgNode
from typing import Any, Dict, List
from ...utils.image.all_logs import log_max_dist_kcenter
from .utils import WhiteBoxDataLoader


class KCenterGreedy:
    def __init__(self, thief_model:nn.Module, thief_data:Dataset, feature:str='fc', metric:str='euclidean'):
        self.name = 'k-center'
        self.thief_model = thief_model
        self.thief_data = thief_data
        self.metric = metric
        self.feature = feature        
        self.min_distances = None
        self.already_selected = []
            
            
    def update_distances(self, cluster_centers:List, only_new:bool=True, reset_dist:bool=False):
        """Update min distances given cluster centers.
        
        Args:
            cluster_centers: indices of cluster centers
            only_new: only calculate distance for newly selected points and update
                min_distances.
            rest_dist: whether to reset min_distances.
        """
        if reset_dist:
            self.min_distances = None
        if only_new:
            cluster_centers = [d for d in cluster_centers if d not in self.already_selected]
        if cluster_centers:
            feat_labeled = self.features[cluster_centers]
            dist = pairwise_distances(self.features, feat_labeled, metric=self.metric)
            if self.min_distances is None:
                self.min_distances = np.min(dist, axis=1).reshape(-1,1)
            else:
                self.min_distances = np.minimum(self.min_distances, dist)
                
                
    def select_batch(self, cfg:CfgNode, labeled_idx:List, unlabeled_idx:List, N:int, *args, **kwargs):
        """
        Diversity promoting active learning method that greedily forms a batch
        to minimize the maximum distance to a cluster center among all unlabeled
        datapoints.

        Args:
        labeled_idx: indices of labeled data points
        unlabeled_idx: indices of unlabeled data points
        N: no. of data points to be select/labeled

        Returns:
        indices of points selected to minimize distance to cluster centers
        """
        # Compute features for all data
        thief_model = self.thief_model
        thief_model.eval()
        thief_model = thief_model.to(cfg.DEVICE)
        
        # register forward hooks on the layers of choice
        if self.feature == 'avgpool':
            # a dict to store the activations
            activation = {}
            def getActivation(name):
                # the hook signature
                def hook(thief_model, input, output):
                    activation[name] = output.detach()
                return hook

            # register the hooks
            h1 = thief_model.avgpool.register_forward_hook(getActivation('avgpool'))   
        
            # compute activations for both labeled and unlabeled data
            all_idx = labeled_idx + unlabeled_idx
            if cfg.TRAIN.BLACKBOX_TRAINING == True:
                all_loader = DataLoader(Subset(self.thief_data, all_idx), batch_size=128, 
                                    pin_memory=False, num_workers=cfg.NUM_WORKERS, shuffle=True)
            else:
                all_loader = WhiteBoxDataLoader(Subset(self.thief_data, all_idx), batch_size=128, 
                                    pin_memory=False, num_workers=cfg.NUM_WORKERS, shuffle=True)
            feat = []
            already_selected = []  # local indices for labeled data points relative to full feature vector
            with torch.no_grad():
                ctr = 0
                for data in tqdm(all_loader):
                    image, index = data[0], data[2]
                    image = image.to(cfg.DEVICE)
                    out = thief_model(image)
                    # collect the activations in the correct list
                    feat.extend(activation['avgpool'].detach().cpu().numpy())
                    for j in index:
                        if j in labeled_idx:
                            already_selected.append(ctr)
                        ctr += 1       
            # detach the hooks
            h1.remove()
            self.features = np.asarray(feat)[:, :, 0, 0]
            print('here')
            
        elif self.feature == 'fc':        
            # compute activations for both labeled and unlabeled data
            all_idx = labeled_idx + unlabeled_idx
            if cfg.TRAIN.BLACKBOX_TRAINING == True:
                all_loader = DataLoader(Subset(self.thief_data, all_idx), batch_size=128, 
                                    pin_memory=True, num_workers=cfg.NUM_WORKERS, shuffle=False)
            else:
                all_loader = WhiteBoxDataLoader(Subset(self.thief_data, all_idx), batch_size=128, 
                                    pin_memory=True, num_workers=cfg.NUM_WORKERS, shuffle=False)
            feat = []
            already_selected = []  # local indices for labeled data points relative to full feature vector
            with torch.no_grad():
                ctr = 0
                for data in tqdm(all_loader):
                    image, index = data[0], data[2]
                    with torch.cuda.amp.autocast():
                        image = image.to(cfg.DEVICE)
                        out = thief_model(image) 
                        # collect the activations in the correct list
                        feat.extend(out.detach().cpu().numpy())
                    for j in index:
                        if j in labeled_idx:
                            already_selected.append(ctr)
                        ctr += 1                    
            self.features = np.asarray(feat)
                
        else:
            raise NotImplementedError(f"Feature {self.feature} not supported! Must either be 'avgpool' or 'fc'")
        
        print('Labeld idx:', labeled_idx)
        print('Already selected:', already_selected)
        # Compute distances from unlabeled points to their nearest cluster centers
        self.update_distances(already_selected, only_new=False, reset_dist=True)

        # Start greedy selection of N unlabeled data points
        new_batch = []

        for _ in range(N):
            ind_selected = np.argmax(self.min_distances)
            # true index of the selected data point in the original dataset
            true_ind = all_idx[ind_selected]
            
            # New examples should not be in already selected since those points
            # should have min_distance of zero to a cluster center.
            assert ind_selected not in already_selected

            self.update_distances([ind_selected], only_new=True, reset_dist=False)
            new_batch.append(true_ind)
        
        max_dist, mean_dist, median_dist, std_dist = np.max(self.min_distances), np.mean(self.min_distances), np.median(self.min_distances), np.std(self.min_distances)
        print(len(new_batch))
        print(new_batch)
        print(max_dist, mean_dist, median_dist, std_dist)
        log_max_dist_kcenter(cfg.LOG_PATH, max_dist, mean_dist, median_dist, std_dist)
        log_max_dist_kcenter(cfg.INTERNAL_LOG_PATH, max_dist, mean_dist, median_dist, std_dist)
        self.already_selected = already_selected
        return new_batch