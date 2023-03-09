import numpy as np

# 读取数据
train = np.load('dataset/train_11.npy')  # 读取训练集
print(train.shape)  # (1229932, 429)

train_label = np.load('dataset/train_label_11.npy')  # 读取训练标签集
print(train_label.shape)  # (1229932,)
