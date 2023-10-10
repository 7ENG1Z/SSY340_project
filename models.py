import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

class LSTMNet(nn.Module):
    def __init__(self, input_size, num_classes):
        super(LSTMNet, self).__init__()
        self.lstm1 = nn.LSTM(input_size, 64, batch_first=True)
        self.lstm2 = nn.LSTM(64, 128, batch_first=True)
        self.lstm3 = nn.LSTM(128, 64, batch_first=True)
        self.fc1 = nn.Linear(64, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        x, _ = self.lstm3(x)
        x = x[:, -1, :]
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    



#the transformer

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=30):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]

class TransformerClassifier(nn.Module):
    def __init__(self, input_dim, nhead, num_encoder_layers, num_classes, dropout=0.5):
        super(TransformerClassifier, self).__init__()
        self.embedding_size = input_dim
        self.positional_encoder = PositionalEncoding(self.embedding_size)
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.embedding_size, nhead=nhead, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        self.decoder = nn.Linear(self.embedding_size, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.positional_encoder(x)
        x = self.transformer_encoder(x)
        x = self.dropout(x[:, -1, :])  # 在解码前使用Dropout
        x = self.decoder(x)
        return x

# # 实例化模型
# model = TransformerClassifier(input_dim=1662, nhead=8, num_encoder_layers=3, num_classes=3, dropout=0.5)

# # 输入尺寸为(batch_size, sequence_length, feature_dim)
# input_tensor = torch.rand(90, 30, 1662)
# output = model(input_tensor)
# print(output.shape)  # 应该输出 torch.Size([90, 3])




