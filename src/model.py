# importing libraries 
import torch
import torch.nn as nn
# model class f
class LSTMRULRegressor(nn.Module): 
    def __init__(self, input_dim, hidden_dim=64, num_layers=2, output_dim=1, dropout=0.2): # We are taking inputs 
        super(LSTMRULRegressor, self).__init__() # Inheriting the model class
        # Using LSTM
        self.lstm = nn.LSTM(  
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        # Our workflow will be sequence wise
        self.fc = nn.Sequential(    
            nn.Linear(hidden_dim, 32),  # Linear NN
            nn.ReLU(),    # Activation function
            nn.Dropout(dropout),
            nn.Linear(32, output_dim)
        )
    # forward 
    def forward(self, x):  
        # x shape: [batch_size, sequence_length, input_dim]
        lstm_out, _ = self.lstm(x)
        # Gathering the final hidden state slice from the sequence lookback window
        last_time_step = lstm_out[:, -1, :]
        predictions = self.fc(last_time_step)
        return predictions.squeeze(-1)