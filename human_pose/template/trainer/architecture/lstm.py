from unicodedata import bidirectional
import torch.nn as nn
import torch

class LSTM(nn.Module):
  def __init__(self, bidirection = False):
    super(LSTM, self).__init__()
    n_class = 3
    n_hidden = 32
    self.lstm1 = nn.LSTM(input_size=n_class, hidden_size=n_hidden, dropout=0.3, bidirectional=bidirection)
    self.lstm2 = nn.LSTM(input_size=n_hidden, hidden_size=n_hidden * 2, dropout=0.3, bidirectional = bidirection)
    self.lstm3 = nn.LSTM(input_size=n_hidden *2, hidden_size=n_hidden, dropout=0.3, bidirectional = bidirection)
    self.dense1 = nn.Linear(n_hidden, int(n_hidden/2))
    self.dense2 = nn.Linear(n_hidden, n_class)
    self.Softmax = nn.Softmax(dim=1)

  def forward(self, X):
    outputs, hidden = self.lstm(X)
    outputs = outputs[-1]  # 최종 예측 Hidden Layer
    output = self.dense1(outputs)
    output = self.dense2(output)
    return output
	