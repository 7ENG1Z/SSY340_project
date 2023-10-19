import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

class LSTMNet(nn.Module):
    def __init__(self, input_size, num_classes, dropout_prob=0.5):
        super(LSTMNet, self).__init__()
        
        self.lstm1 = nn.LSTM(input_size, 64, batch_first=True)
        self.lstm2 = nn.LSTM(64, 64, batch_first=True)


        
        self.dropout = nn.Dropout(dropout_prob)
        
        self.fc1 = nn.Linear(64, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, num_classes)
        self.relu = nn.ReLU()
        for param in self.parameters():
            param.requires_grad = True

    def forward(self, x):
        x, _ = self.lstm1(x)
        
        # x, _ = self.lstm2(x)
        
        x, _ = self.lstm2(x)
        x = self.dropout(x)
        
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
class TransformerLSTMClassifier1(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(TransformerLSTMClassifier1, self).__init__()
        
        self.embedding_size = input_dim
        self.positional_encoder = PositionalEncoding(self.embedding_size)
        
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.embedding_size, nhead=1, dropout=0.7)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)
        
        # LSTM Layer
        self.lstm = nn.LSTM(self.embedding_size, 32, batch_first=True)  # 输出维度为64
        
        self.decoder = nn.Linear(32, num_classes)
          # 因为LSTM的输出维度是64
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # Positional Encoding and Transformer Encoding
        x = self.positional_encoder(x)
        x = self.transformer_encoder(x)
        
        # LSTM Layer
        x, _ = self.lstm(x)
        x = self.dropout(x[:, -1, :])
        
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

class LSTM_FCN(nn.Module):
    def __init__(self, input_size, num_classes, dropout_rate=0.5):
        super(LSTM_FCN, self).__init__()

        # LSTM branch
        self.lstm = nn.LSTM(input_size=input_size,
                            hidden_size=64,  
                            num_layers=1,
                            batch_first=True)
        self.lstm_fc = nn.Linear(64, 64)
        self.lstm_dropout = nn.Dropout(dropout_rate)

        # FCN branch
        self.conv1 = nn.Conv1d(input_size, 32, 8, 1)
        self.conv2 = nn.Conv1d(32, 64, 5, 1)
        self.fcn_fc = nn.Linear(64*19, 64)
        self.fcn_dropout = nn.Dropout(dropout_rate)  # Dropout after FC layer

        # Final classification layer
        self.classifier = nn.Linear(64 * 2, num_classes)

    def forward(self, x):
        # LSTM branch
        h, _ = self.lstm(x)
        h = self.lstm_fc(h[:, -1, :])
        h = self.lstm_dropout(h)

        # FCN branch
        x = x.transpose(1, 2)
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = self.fcn_fc(x)
        x = self.fcn_dropout(x)  # Dropout added here, after FC layer

        # Concatenate and classify
        combined = torch.cat((h, x), 1)
        output = self.classifier(combined)
        return output

    def freeze(self, layer_name):
        for param in getattr(self, layer_name).parameters():
            param.requires_grad = False

    def unfreeze(self, layer_name):
        for param in getattr(self, layer_name).parameters():
            param.requires_grad = True