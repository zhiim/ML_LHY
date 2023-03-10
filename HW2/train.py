from torch.utils.data import DataLoader
import numpy as np
import gc
from utils import *

# load data
data = np.load('dataset/train_11.npy')  # load training data
print('size of dataset from train_11.npy: {}'.format(data.shape))  # (1229932, 429)

label = np.load('dataset/train_label_11.npy')  # load labels
print('size of dataset from train_label_11.npy: {}'.format(label.shape))  # (1229932,)

device = get_device()
print('your device: {}'.format(device))

# parameters for training
config = {
    'num_epochs': 10,
    'batch_size': 270,
    'optimizer': 'SGD',  # optimizer algorithm
    'optim_hparas': {
        'lr': 0.001,  # learning rate
        'momentum': 0.9  # momentum for SGD
    },
    'save_path': 'models/model.pth',
    'early_stop': 200
}

# split data for training and validation
train_data, train_label, val_data, val_label = split_data(data=data, label=label)

train_set = VoiceDataset(train_data, train_label)  # dataset for training
train_loader = DataLoader(dataset=train_set, batch_size=config['batch_size'], shuffle=True)

val_set = VoiceDataset(val_data, val_label)
val_loader = DataLoader(dataset=val_set, batch_size=config['batch_size'], shuffle=False)

# delete the data loaded to save space
del data, label, train_data, train_label, val_data, val_label
gc.collect()

model = NeuralNet(429, 39).to(device)  # create network

min_mse, loss_record = train(train_loader, val_loader, model, config, device)