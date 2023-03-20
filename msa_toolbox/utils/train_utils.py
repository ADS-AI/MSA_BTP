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


def accuracy_f1_precision_recall(model: nn.Module, data_loader: DataLoader, device: torch.device) -> tuple:
    """
    Returns accuracy, f1, precision, recall for the model on the data_loader
    """
    model = model.to(device)
    model.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = torch.max(output.data, 1)

            y_true.extend(target.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())
    metric = {
        'accuracy': accuracy_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred, average='macro'),
        'precision': precision_score(y_true, y_pred, average='macro'),
        'recall': recall_score(y_true, y_pred, average='macro')
    }
    return metric


def data_distribution(data_loader: DataLoader) -> list:
    """
    Returns the distribution of the data_loader
    """
    print("Number of sample in Dataset: ", len(data_loader.dataset))
    # print(np.unique(data.labels, return_counts=True))
    distribution = {}
    all_labels = []
    for data, target in data_loader:
        all_labels.extend(target.cpu().detach().numpy())
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
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            thief_output = thief_model(data)
            victim_output = victim_model(data)
            _, thief_predicted = torch.max(thief_output.data, 1)
            _, victim_predicted = torch.max(victim_output.data, 1)
            correct += np.sum(thief_predicted.cpu().numpy()
                              == victim_predicted.cpu().numpy())
            length += len(target)
    return correct/length
