# train the network for voice frame classification

from torch.utils.data import DataLoader
import numpy as np
import gc
from utils import *

myseed = 42069  # set a random seed for reproducibility
same_seeds(myseed)

device = get_device()
print('your device: {}'.format(device))

# parameters for training
config = {
    'num_epochs': 10,
    'batch_size': 60,
    'optimizer': 'Adam',  # optimizer algorithm
    'optim_hparas': {
        'lr': 0.001  # learning rate
    },
    'save_path': 'models/model.pth',
    'early_stop': 20
}

# load data
data = np.load('dataset/train_11.npy')  # load training data
print('size of dataset from train_11.npy: {}'.format(data.shape))  # (1229932, 429)

label = np.load('dataset/train_label_11.npy')  # load labels
print('size of dataset from train_label_11.npy: {}'.format(label.shape))  # (1229932,)

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

acc_record, loss_record, max_acc = train(train_loader, val_loader, model, config, device)

# plot learning curve
plot_learning_curve(loss_record, title='loss')
plot_learning_curve(acc_record, title='accuracy')

# data fot test
data_test = np.load('dataset/train_11.npy')  # load training data
print('size of dataset from test_11.npy: {}'.format(data.shape))  # (1229932, 429)

test_set = VoiceDataset(data_test, None)

test_loader = DataLoader(dataset=test_set, batch_size=config['batch_size'], shuffle=False)

predicts = test(test_loader, model, device)

# save prediction for test dataset as a .csv file
with open('prediction.csv', 'w') as f:
    f.write('Id,Class\n')
    for i, y in enumerate(predicts):
        f.write('{},{}\n'.format(i, y))
