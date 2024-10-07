import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
from torch.optim import Optimizer
from torch.nn.modules.loss import _Loss
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from .cfg_reader import CfgNode
from typing import Any, Dict


def accuracy_f1_precision_recall(cfg:CfgNode, thief_model: nn.Module, data_loader: DataLoader, device: torch.device, is_victim_loader:bool=False) -> Dict:
    """
    Returns accuracy, f1, precision, recall for the model on the data_loader
    """
    thief_model.eval()
    thief_model = thief_model.to(device)
    y_true = []
    y_pred = []
    with torch.no_grad():
        for image, label, index in data_loader:
            image, label = image.to(device), label.to(device)
            if (cfg.TRAIN.BLACKBOX_TRAINING == False) and (not is_victim_loader):
                label = torch.argmax(label, dim=1)
            output = thief_model(image)
            predicted = torch.argmax(output.data, dim=1)
            y_true.extend(label.detach().cpu().numpy())
            y_pred.extend(predicted.detach().cpu().numpy())
    metric = {
        'accuracy': accuracy_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred, average='macro'),
        'precision': precision_score(y_true, y_pred, average='macro'),
        'recall': recall_score(y_true, y_pred, average='macro')
    }
    return metric


def data_distribution(cfg:CfgNode, data_loader: DataLoader):
    """
    Returns the distribution of the data_loader
    """
    # print("Number of sample in Dataset: ", len(data_loader.dataset))
    # print(np.unique(data.labels, return_counts=True))
    distribution = {}
    all_labels = []
    for image, label, index in data_loader:
        image, label = image.to(cfg.DEVICE), label.to(cfg.DEVICE)
        if cfg.TRAIN.BLACKBOX_TRAINING == False:
            label = torch.argmax(label, dim=1)
        all_labels.extend(label.detach().cpu().tolist())
    all_labels = np.array(all_labels)
    for label in np.unique(all_labels):
        distribution[label] = np.sum(all_labels == label)
    return distribution


def agreement(thief_model: nn.Module, victim_model: nn.Module, data_loader: DataLoader, device: torch.device) -> float:
    '''
    Calculates the agreement between the thief and victim model on the data_loader
    '''
    thief_model.eval()
    victim_model.eval()
    thief_model = thief_model.to(device)
    victim_model = victim_model.to(device)
    correct = 0
    length = 0
    with torch.no_grad():
        for image, label, index in data_loader:
            image, label = image.to(device), label.to(device)
            thief_output = thief_model(image)
            victim_output = victim_model(image)
            thief_predicted = torch.argmax(thief_output, dim=1)
            victim_predicted = torch.argmax(victim_output, dim=1)
            correct += np.sum(thief_predicted.detach().cpu().numpy() == victim_predicted.detach().cpu().numpy())
            length += len(image)
    return correct/length

def agreement_api(thief_model: nn.Module, data_loader: DataLoader, device: torch.device) -> float:
    '''
    Calculates the agreement between the thief and victim model on the data_loader
    '''
    thief_model.eval()
    thief_model = thief_model.to(device)
    correct = 0
    length = 0
    with torch.no_grad():
        for image, label, index in data_loader:
            image, label = image.to(device), label.to(device)
            thief_output = thief_model(image)
            thief_predicted = torch.argmax(thief_output, dim=1)
            victim_predicted = label
            correct += np.sum(thief_predicted.detach().cpu().numpy() == victim_predicted.detach().cpu().numpy())
            length += len(image)
    return correct/length
