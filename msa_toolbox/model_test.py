from .utils.load_data_and_models import load_victim_model, load_victim_dataset
from torch.utils.data import DataLoader
from .utils.train_model import train, test
import torch.nn as nn
from torch import optim

model = load_victim_model('resnet18', 10, 'default', True)
# print(model)
data = load_victim_dataset('mnist', train=False, transform=None, target_transform=None, download=True)
print(data)
print(type(data))
print(data.data.shape)
print(data.targets.shape)
print(data.data.shape)
print(data.targets.shape)

dataloader = DataLoader(data, batch_size=300, shuffle=True, num_workers=2)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-3)
train_loss, train_acc, val_loss, val_acc = train(model, dataloader, 1, 300, optimizer, criterion, 'cuda:0', log_interval=100, all_data=False)
print(train_loss, train_acc, val_loss, val_acc)
