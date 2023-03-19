# train the network for voice frame classification

from torch.utils.data import DataLoader
import gc
from utils import *

myseed = 6666  # set a random seed for reproducibility
same_seeds(myseed)

device = get_device()
print('your device: {}'.format(device))

# parameters for training
config = {
    'num_epochs': 10,
    'batch_size': 64,
    'optimizer': 'Adam',  # optimizer algorithm
    'optim_hparas': {
        'lr': 0.0003, # learning rate
        'weight_decay': 1e-5
    },
    'save_path': 'models/model.pth',
    'early_stop': 5
}

# load data
train_set = FoodDataset("./train", tfm=train_tfm)
val_set = FoodDataset("./valid", tfm=test_tfm)

train_loader = DataLoader(train_set, batch_size=config['batch_size'], shuffle=True)
val_loader = DataLoader(val_set, batch_size=config['batch_size'], shuffle=False)

# delete the data loaded to save space
del train_set, val_set
gc.collect()

model = Classifier().to(device)  # create network

acc_record, loss_record, max_acc = train(train_loader, val_loader, model, config, device)

# plot learning curve
plot_learning_curve(loss_record, title='loss')
plot_learning_curve(acc_record, title='accuracy')

# data fot test
test_set = FoodDataset("./test", tfm=test_tfm)

test_loader = DataLoader(dataset=test_set, batch_size=config['batch_size'], shuffle=False)

predicts = test(test_loader, model, device)

# save prediction for test dataset as a .csv file
with open('prediction.csv', 'w') as f:
    f.write('Id,Class\n')
    for i, y in enumerate(predicts):
        f.write('{},{}\n'.format(i, y))
