import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np

myseed = 42069  # set a random seed for reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(myseed)
torch.manual_seed(myseed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(myseed)


# support gpu or not
def get_device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'


# split data into training data and validation data
def split_data(data, label):
    VAL_RATIO = 0.2

    percent = int(data.shape[0] * (1 - VAL_RATIO))
    train_data, train_label, val_data, val_label = data[:percent], label[:percent], data[percent:], label[percent:]
    print('Size of training set: {}'.format(train_data.shape))
    print('Size of validation set: {}'.format(val_data.shape))
    return train_data, train_label, val_data, val_label


# a custom Dataset, load data from .npy files
class VoiceDataset(Dataset):
    def __init__(self, data_train, data_label):
        super().__init__()
        self.data_train = torch.from_numpy(data_train).float()  # data for training
        self.data_label = torch.LongTensor(data_label.astype(np.int))  # label for data used for training

    def __getitem__(self, index):
        return self.data_train[index], self.data_label[index]  # return a element in dataset according to index

    def __len__(self):
        return len(self.data_label)  # return the length of this dataset


class NeuralNet(nn.Module):
    def __init__(self, input_size, output_size) -> None:
        super(NeuralNet, self).__init__()
        # define the network
        self.net = nn.Sequential(
            nn.Linear(input_size, 1024),
            nn.Sigmoid(),
            nn.Linear(1024, 512),
            nn.Sigmoid(),
            nn.Linear(512, 128),
            nn.Sigmoid(),
            nn.Linear(128, output_size),
        )

    def forward(self, input):
        # return the output of network
        return self.net(input)
    
    def cal_loss(self, pred, target):
        self.criterion = nn.MSELoss()
        return self.criterion(pred, target)


# this function includes everything for training
def train(tr_set, dv_set, model, config, device):
    num_epochs = config['num_epochs']  # number of epochs

    # set the optimizer
    optimizer = getattr(torch.optim, config['optimizer'])(
        model.parameters(), **config['optim_hparas'])
    
    # init parameters for epochs
    min_mse = 1000.  # set the initial value for min_mse (higher than the mse after first epoch)
    loss_record = []
    
    # epochs for trianing
    for epoch in range(num_epochs):
        model.train()  # set model to trian mode
        for data, label in tr_set:
            optimizer.zero_grad()  # set gradient to zero before calculate
            data, label = data.to(device), label.to(device)  # move data to device
            print(data.dtype)
            pred = model(data)  # compute the predict from data
            loss = model.cal_loss(pred, label)  # compute the mse loss
            loss.backward()  # get the gradient
            optimizer.step()  # updata parameters in model
            loss_record.append(loss.detach().cpu().item())

            # After each epoch, test your model on the validation (development) set.
            dev_mse = dev(dv_set, model, device)
            if dev_mse < min_mse:
                # Save model if your model improved
                min_mse = dev_mse
                print('Saving model (epoch = {:4d}, loss = {:.4f})'
                    .format(epoch + 1, min_mse))
                torch.save(model.state_dict(), config['save_path'])  # Save model to specified path
                early_stop_cnt = 0
            else:
                early_stop_cnt += 1

            epoch += 1
            loss_record['dev'].append(dev_mse)
            if early_stop_cnt > config['early_stop']:
                # Stop training if your model stops improving for "config['early_stop']" epochs.
                break
    
    print('Finished training after {} epochs'.format(epoch))
    return min_mse, loss_record


# fuction to compute mse in validation
def dev(dv_set, model, device):
    model.eval()  # set model to evalutation mode
    total_loss = 0
    for x, y in dv_set:  # iterate through the dataloader
        x, y = x.to(device), y.to(device)  # move data to device (cpu/cuda)
        with torch.no_grad():  # disable gradient calculation
            pred = model(x)  # forward pass (compute output)
            mse_loss = model.cal_loss(pred, y)  # compute loss
        total_loss += mse_loss.detach().cpu().item() * len(x)  # accumulate loss
    total_loss = total_loss / len(dv_set.dataset)  # compute averaged loss

    return total_loss


# function for model test
def test(tt_set, model, device):
    model.eval()  # set model to evalutation mode
    preds = []

    for data in tt_set:
        data = data.to(device)
        # we don't need compute gradient when testing
        with torch.no_grad():
            pred = model(data)
            preds.append(pred.detach().cpu())
        preds = torch.cat(preds, dim=0).numpy()
    return preds



# load data
data = np.load('dataset/train_11.npy')  # load training data
# print(train.shape)  # (1229932, 429)

label = np.load('dataset/train_label_11.npy')  # load labels
# print(train_label.shape)  # (1229932,)

device = get_device()

# parameters for training
config = {
    'num_epochs': 1000,
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

model = NeuralNet(429, 38).to(device)  # create network

min_mse, loss_record = train(train_loader, val_loader, model, config, device)