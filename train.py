import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import os
from sklearn.model_selection import train_test_split
from  generate_dataset.utils import*
from models import LSTMNet


#prepare the data 
#give the actions lable_map
label_map = {label:num for num, label in enumerate(actions)}
label_map
#load the dataset in nparray
sequences, labels = [], []
for action in actions:
    for sequence in np.array(os.listdir(os.path.join(DATA_PATH, action))).astype(int):
        window = []
        for frame_num in range(sequence_length):
            res = np.load(os.path.join(DATA_PATH, action, str(sequence), "{}.npy".format(frame_num)))
            window.append(res)
        sequences.append(window)
        labels.append(label_map[action])
X=np.array(sequence)
print(labels)
Y=np.array(labels)


#train_test split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)


#load the data in data loader
train_data = TensorDataset(torch.tensor(X_train, dtype=torch.float), torch.tensor(Y_train, dtype=torch.long))
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)

# # 参数
input_size = 1662
num_classes = actions.shape[0]

# # 实例化模型
model = LSTMNet(input_size, num_classes)