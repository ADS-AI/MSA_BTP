import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.nn.modules.loss import _Loss

'''
Sample file for training a model on a dataset.
'''

def train(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, epochs: int, batch_size: int, optimizer: Optimizer,
          criterion: _Loss, device: str, log_interval=100, all_data=False, verbose=True):
    '''
    This function trains a given model on a given 'train_loader' for 'epochs' number of epochs using a 
    given optimizer and criterion function. The device parameter specifies whether to use GPU or CPU
    for training.

    Parameters:
        - model (nn.Module): The neural network model to train.
        - train_loader (DataLoader): The data loader to use for training and validation.
        - epochs (int): The number of epochs to train the model for.
        - batch_size (int): The batch size to use for training.
        - optimizer (Optimizer): The optimizer to use for training the model.
        - criterion (_Loss): The loss function to use for training and validation.
        - device (str): Specifies whether to use GPU or CPU for training the model.
        - log_interval (int): The number of batches after which to print the training and validation loss and accuracy.
        - all_data (bool): If set to True, the function returns the training and validation losses and accuracies for all epochs. Otherwise, it returns only the final loss and accuracy values.
        - verbose (bool): If set to True, the function prints the training and validation loss and accuracy for each batch.

    Returns:
        The function returns the final training loss, training accuracy, validation loss, and 
        validation accuracy. If the all_data parameter is set to True, the function returns the
        training and validation losses and accuracies for all epochs.
    '''
    # Train the model
    train_loss = []
    train_acc = []
    val_loss = []
    val_acc = []
    model = model.to(device)
    for epoch in range(epochs):
        print('======================> Epoch: {} <======================='.format(epoch))
        # Train the model
        train_epoch_loss, train_epoch_acc = train_one_epoch(model, train_loader, epoch, batch_size, optimizer,
                                                            criterion, device, log_interval, verbose)
        print('Training Loss: {:.4f}\tTraining Accuracy: {:.2f}%'.format(
            train_epoch_loss, train_epoch_acc))
        train_loss.append(train_epoch_loss)
        train_acc.append(train_epoch_acc)
        # Validate the model
        val_epoch_loss, val_epoch_acc = validate_one_epoch(model, val_loader, epoch, batch_size, criterion,
                                                           device, log_interval, verbose)
        print('Validation Loss: {:.4f}\tValidation Accuracy: {:.2f}%'.format(
            val_epoch_loss, val_epoch_acc))
        val_loss.append(val_epoch_loss)
        val_acc.append(val_epoch_acc)
    if all_data:
        return train_loss, train_acc, val_loss, val_acc
    return train_loss[-1], train_acc[-1], val_loss[-1], val_acc[-1]


def train_one_epoch(model: nn.Module, dataloader: DataLoader, epoch: int, batch_size: int, optimizer: Optimizer,
                    criterion: _Loss, device: str, log_interval=20, verbose=True):
    '''
    Trains the specified 'model' on the provided 'dataloader' for one epoch.
    Args:
        model (nn.Module): The model to be trained.
        dataloader (DataLoader): The DataLoader containing the training dataset.
        epoch (int): The current epoch number.
        batch_size (int): The batch size used for training.
        optimizer (Optimizer): The optimizer to use for updating model parameters.
        criterion (_Loss): The loss function to use for computing the training loss.
        device (str): The device to run the training on.
        log_interval (int): The number of batches after which to log the training progress. Defaults to 20.
        verbose (bool): A flag indicating whether to print the training progress to the console. Defaults to True.

    Returns:
        A tuple containing the average training loss and accuracy over the entire dataset.
    '''
    train_loss = 0
    correct = 0
    total = 0
    batch_idx = 1
    model.train()
    model = model.to(device)
    for inputs, targets in tqdm(dataloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        if batch_idx % log_interval == 0 and verbose:
            print('Training Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tAccuracy: {:.2f}%'.format(
                epoch, batch_idx * len(inputs), len(dataloader.dataset),
                100. * batch_idx / len(dataloader), loss.item(),
                100. * correct / total))
        batch_idx += 1
    return train_loss / len(dataloader.dataset), 100. * correct / total


def validate_one_epoch(model: nn.Module, dataloader: DataLoader, epoch: int, batch_size: int, criterion: _Loss,
                       device: str, log_interval=20, verbose=True):
    '''
    Validates the specified 'model' on the provided 'dataloader' for one epoch.
    Args:
        model (nn.Module): The model to be validated.
        dataloader (DataLoader): The DataLoader containing the validation dataset.
        epoch (int): The current epoch number.
        batch_size (int): The batch size used for validation.
        criterion (_Loss): The loss function to use for computing the validation loss.
        device (str): The device to run the validation on.
        log_interval (int): The number of batches after which to log the validation progress. Defaults to 20.
        verbose (bool): A flag indicating whether to print the validation progress to the console. Defaults to True.

    Returns:
        A tuple containing the average validation loss and accuracy over the entire dataset.
    '''
    model.eval()
    val_loss = 0
    correct = 0
    total = 0
    batch_idx = 1
    with torch.no_grad():
        for inputs, targets in tqdm(dataloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            if batch_idx % log_interval == 0 and verbose:
                print('Validation Epoch {}: [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tAccuracy: {:.2f}%'.format(
                    epoch, batch_idx * len(inputs), len(dataloader.dataset),
                    100. * batch_idx / len(dataloader), loss.item(),
                    100. * correct / total))
            batch_idx += 1
    return val_loss / len(dataloader.dataset), 100. * correct / total


def test(model: nn.Module, test_loader: DataLoader, batch_size: int, criterion: _Loss, device: str,
         log_interval=20, verbose=True):
    '''
    Tests the specified 'model' on the provided 'dataloader'.
    Args:
        model (nn.Module): The model to be tested.
        test_loader (DataLoader): The DataLoader containing the test dataset.
        batch_size (int): The batch size used for testing.
        criterion (_Loss): The loss function to use for computing the test loss.
        device (str): The device to run the testing on.
        log_interval (int): The number of batches after which to log the testing progress. Defaults to 20.
        verbose (bool): A flag indicating whether to print the testing progress to the console. Defaults to True.

    Returns:
        A tuple containing the average test loss and accuracy over the entire dataset.
    '''
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    batch_idx = 1
    with torch.no_grad():
        for inputs, targets in tqdm(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            if batch_idx % log_interval == 0:
                print('Test Epoch: [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tAccuracy: {:.2f}%'.format(
                    batch_idx * len(inputs), len(test_loader.dataset),
                    100. * batch_idx / len(test_loader), loss.item(),
                    100. * correct / total))
            batch_idx += 1
    return test_loss / len(test_loader.dataset), 100. * correct / total
