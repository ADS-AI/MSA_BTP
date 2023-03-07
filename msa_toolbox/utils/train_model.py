import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.nn.modules.loss import _Loss


def train(model:nn.Module, dataloader:DataLoader, epochs:int, batch_size:int, optimizer:Optimizer, 
            criterion:_Loss, device:str, log_interval=100, all_data=False, verbose=True):
    # Train the model
    train_loss = []
    train_acc = []
    val_loss = []
    val_acc = []
    for epoch in range(epochs):
        print('======================> Epoch: {} <======================='.format(epoch))
        # Train the model
        train_epoch_loss, train_epoch_acc = train_one_epoch(model, dataloader, epoch, batch_size, optimizer, 
                                                            criterion, device, log_interval, verbose)
        print('Training Loss: {:.4f}\tTraining Accuracy: {:.2f}%'.format(train_epoch_loss, train_epoch_acc))
        train_loss.append(train_epoch_loss)
        train_acc.append(train_epoch_acc)
        # Validate the model
        val_epoch_loss, val_epoch_acc = validate_one_epoch(model, dataloader, epoch, batch_size, criterion,
                                                            device, log_interval, verbose)
        print('Validation Loss: {:.4f}\tValidation Accuracy: {:.2f}%'.format(val_epoch_loss, val_epoch_acc))
        val_loss.append(val_epoch_loss)
        val_acc.append(val_epoch_acc)
    if all_data:
        return train_loss, train_acc, val_loss, val_acc
    return train_loss[-1], train_acc[-1], val_loss[-1], val_acc[-1]


def train_one_epoch(model:nn.Module, dataloader:DataLoader, epoch:int, batch_size:int, optimizer:Optimizer, 
            criterion:_Loss, device:str, log_interval=20, verbose=True):
    # Train the model
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    batch_idx = 1
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


def validate_one_epoch(model:nn.Module, dataloader:DataLoader, epoch:int, batch_size:int, criterion:_Loss,
                device:str, log_interval=20, verbose=True):
    # Validate the model
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

            if batch_idx % log_interval == 0:
                print('Validation Epoch {}: [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tAccuracy: {:.2f}%'.format(
                    epoch, batch_idx * len(inputs), len(dataloader.dataset),
                    100. * batch_idx / len(dataloader), loss.item(),
                    100. * correct / total))
            batch_idx += 1

    return val_loss / len(dataloader.dataset), 100. * correct / total


def test(model:nn.Module, dataloader:DataLoader, batch_size:int, criterion:_Loss, device:str, 
            log_interval=20, verbose=True):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    batch_idx = 1
    with torch.no_grad():
        for inputs, targets in tqdm(dataloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            if batch_idx % log_interval == 0:
                print('Test Epoch: [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tAccuracy: {:.2f}%'.format(
                    batch_idx * len(inputs), len(dataloader.dataset),
                    100. * batch_idx / len(dataloader), loss.item(),
                    100. * correct / total))
            batch_idx += 1

    return test_loss / len(dataloader.dataset), 100. * correct / total