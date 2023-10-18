import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# class LSTMNet(nn.Module):
#     def __init__(self, input_size, num_classes):
#         super(LSTMNet, self).__init__()
#         self.lstm1 = nn.LSTM(input_size, 64, batch_first=True)
#         self.lstm2 = nn.LSTM(64, 128, batch_first=True)
#         self.lstm3 = nn.LSTM(128, 64, batch_first=True)
#         self.fc1 = nn.Linear(64, 64)
#         self.fc2 = nn.Linear(64, 32)
#         self.fc3 = nn.Linear(32, num_classes)
#         self.relu = nn.ReLU()

#     def forward(self, x):
#         x, _ = self.lstm1(x)
#         x, _ = self.lstm2(x)
#         x, _ = self.lstm3(x)
#         x = x[:, -1, :]
#         x = self.relu(self.fc1(x))
#         x = self.relu(self.fc2(x))
#         x = self.fc3(x)
#         return x
    
#     def freeze(self, layer_name):
#         # 冻结指定层
#         for param in getattr(self, layer_name).parameters():
#             param.requires_grad = False

#     def unfreeze(self, layer_name):
#         # 解冻指定层
#         for param in getattr(self, layer_name).parameters():
#             param.requires_grad = True



class LSTMNet(nn.Module):
    def __init__(self, input_size, num_classes, dropout_prob=0.5):
        super(LSTMNet, self).__init__()
        
        self.lstm1 = nn.LSTM(input_size, 64, batch_first=True)
        self.lstm2 = nn.LSTM(64, 128, batch_first=True)
        self.lstm3 = nn.LSTM(128, 64, batch_first=True)
        
        self.dropout1 = nn.Dropout(dropout_prob)
        self.dropout2 = nn.Dropout(dropout_prob)
        self.dropout3 = nn.Dropout(dropout_prob)
        
        self.fc1 = nn.Linear(64, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x, _ = self.lstm1(x)
        x = self.dropout1(x)
        
        x, _ = self.lstm2(x)
        x = self.dropout2(x)
        
        x, _ = self.lstm3(x)
        x = self.dropout3(x)
        
        x = x[:, -1, :]
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
    def freeze(self, layer_name):
        # 冻结指定层
        for param in getattr(self, layer_name).parameters():
            param.requires_grad = False

    def unfreeze(self, layer_name):
        # 解冻指定层
        for param in getattr(self, layer_name).parameters():
            param.requires_grad = True
    


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
class SimpleTransformerClassifier(nn.Module):
    def __init__(self, input_dim, num_classes, dropout=0.3):  # 将dropout调整为0.3
        super(SimpleTransformerClassifier, self).__init__()
        self.embedding_size = input_dim
        self.positional_encoder = PositionalEncoding(self.embedding_size)
        
        # 使用较小的embedding size和少量的attention heads
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.embedding_size, nhead=1, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)  # 使用一个encoder层
        self.decoder = nn.Linear(self.embedding_size, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.positional_encoder(x)
        x = self.transformer_encoder(x)
        x = self.dropout(x[:, -1, :])  # 在解码前使用Dropout
        x = self.decoder(x)
        return x
    


    def freeze(self, layer_name):
        # 冻结指定层
        for param in getattr(self, layer_name).parameters():
            param.requires_grad = False

    def unfreeze(self, layer_name):
        # 解冻指定层
        for param in getattr(self, layer_name).parameters():
            param.requires_grad = True



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
class TransformerLSTMClassifier(nn.Module):
    def __init__(self, input_dim, num_classes, dropout=0.3):
        super(TransformerLSTMClassifier, self).__init__()
        
        self.embedding_size = input_dim
        self.positional_encoder = PositionalEncoding(self.embedding_size)
        
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.embedding_size, nhead=1, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)
        
        # LSTM Layer
        self.lstm = nn.LSTM(self.embedding_size, 64, batch_first=True)  # 输出维度为64
        
        self.decoder = nn.Linear(64, num_classes)  # 因为LSTM的输出维度是64
        self.dropout = nn.Dropout(dropout)num_classes
        x = self.transformer_encoder(x)
        
        # LSTM Layer
        x, _ = self.lstm(x)
        x = self.dropout(x[:, -1, :])
        
        x = self.decoder(x)
        return x






# # 实例化模型
# model = TransformerClassifier(input_dim=1662, nhead=8, num_encoder_layers=3, num_classes=3, dropout=0.5)

# # 输入尺寸为(batch_size, sequence_length, feature_dim)
# input_tensor = torch.rand(90, 30, 1662)
# output = model(input_tensor)
# print(output.shape)  # 应该输出 torch.Size([90, 3])




