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
    def __init__(self, data_path, mode='train') -> None:
        super().__init__()
        