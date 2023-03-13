# train the network for voice frame classification

from torch.utils.data import DataLoader
import gc
from utils import *


concat_nframes = 3   # the number of frames to concat with, n must be odd (total 2k+1 = n frames)
train_ratio = 0.75   # the ratio of data used for training, the rest will be used for validation

# training parameters
seed = 1213  

# parameters for training
config = {
    'num_epochs': 100,
    'batch_size': 512,
    'optimizer': 'Adam',  # optimizer algorithm
    'optim_hparas': {
        'lr': 0.0001  # learning rate
    },
    'save_path': 'models/model.pth',
    'early_stop': 20
}

same_seeds(seed)
device = get_device()

# preprocess data
train_X, train_y = preprocess_data(split='train', feat_dir='./libriphone/feat', phone_path='./libriphone', concat_nframes=concat_nframes, train_ratio=train_ratio, random_seed=seed)
val_X, val_y = preprocess_data(split='val', feat_dir='./libriphone/feat', phone_path='./libriphone', concat_nframes=concat_nframes, train_ratio=train_ratio, random_seed=seed)

train_set = VoiceDataset(train_X, train_y)  # dataset for training
train_loader = DataLoader(dataset=train_set, batch_size=config['batch_size'], shuffle=True)

val_set = VoiceDataset(val_X, val_y)
val_loader = DataLoader(dataset=val_set, batch_size=config['batch_size'], shuffle=False)

# delete the data loaded to save space
del train_X, train_y, val_X, val_y
gc.collect()

model = NeuralNet(39 * concat_nframes, 41).to(device)  # create network

acc_record, loss_record, max_acc = train(train_loader, val_loader, model, config, device)

# plot learning curve
plot_learning_curve(loss_record, title='deep model')
plot_learning_curve(acc_record, title='deep model')

# load test datasest
test_X = preprocess_data(split='test', feat_dir='./libriphone/feat', phone_path='./libriphone', concat_nframes=concat_nframes)
test_set = VoiceDataset(test_X, None)

test_loader = DataLoader(dataset=test_set, batch_size=config['batch_size'], shuffle=False)

predicts = test(test_loader, model, device)

# save prediction for test dataset as a .csv file
with open('prediction.csv', 'w') as f:
    f.write('Id,Class\n')
    for i, y in enumerate(predicts):
        f.write('{},{}\n'.format(i, y))
