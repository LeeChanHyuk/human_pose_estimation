from unicodedata import bidirectional
import torch.nn as nn
import torch

class LSTM(nn.Module):
  def __init__(self, bidirection = False):
    super(LSTM, self).__init__()
    input_dim = 3
    n_class = 7
    n_hidden = 32
    self.lstm1 = nn.LSTM(input_size=input_dim, hidden_size=n_hidden, dropout=0.3, bidirectional=bidirection, batch_first = True, num_layers = 3)
    #self.lstm2 = nn.LSTM(input_size=n_hidden, hidden_size=n_hidden * 2, dropout=0.3, bidirectional = bidirection, batch_first=True)
    #self.lstm3 = nn.LSTM(input_size=n_hidden *2, hidden_size=n_hidden, dropout=0.3, bidirectional = bidirection, batch_first = True)
    self.dense1 = nn.Linear(n_hidden, int(n_hidden/2))
    self.dense2 = nn.Linear(int(n_hidden / 2), n_class)
    self.Softmax = nn.Softmax(dim=1)

  def forward(self, X):
    sequence = 30
    outputs, (hidden, cell) = self.lstm1(X)
    #outputs, (hidden, cell) = self.lstm1(outputs)
    #outputs, (hidden, cell) = self.lstm1(outputs)
    #outputs = outputs[:,-1,:]  # batch_size, hidden_state, class_num
    output = outputs[:, sequence-1, :]
    output = self.dense1(output)
    output = self.dense2(output)
    return output