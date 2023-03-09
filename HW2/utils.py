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


# 判断设备是否可以使用gpu
def get_device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'


# 从数据集里读取数据，形成Dataset
class VoiceDataset(Dataset):
    def __init__(self, data_train, data_label):
        super().__init__()
        self.data_train = torch.from_numpy(data_train)
        self.data_label = torch.from_numpy(data_label)

    def __getitem__(self, index):
        return self.data_train[index], self.data_label[index]

    def __len__(self):
        return len(self.data_label)
        