from . load_data_and_models import load_victim_dataset, load_thief_dataset, load_victim_model, load_thief_model


model = load_victim_model('resnet18', 10, 'default', True)
print(model)
data = load_victim_dataset('mnist', train=False, transform=None, target_transform=None, download=True)
print(data)
print(type(data))
print(data.train_data.shape)
print(data.train_labels.shape)
print(data.test_data.shape)
print(data.test_labels.shape)


