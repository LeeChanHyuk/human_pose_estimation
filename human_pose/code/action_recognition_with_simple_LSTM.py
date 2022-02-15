from lib2to3.pytree import BasePattern
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
from torch.utils.data import Dataset, dataloader
from sklearn.model_selection import train_test_split


import os

base_path = os.getcwd()
yaw_data_name = '/head_pose_yaw.txt'
pitch_data_name = '/head_pose_pitch.txt'

yaw_file = open(base_path + yaw_data_name, 'r')
pitch_file = open(base_path + pitch_data_name, 'r')
x_data = []
labels = []

### Yaw text data processing
while True:
	line = yaw_file.readline()
	if not line: break
	yaw, pitch, roll, label = line.strip().split(' ')
	yaw, pitch, roll = float(yaw), float(pitch), float(roll)
	x_data.append([yaw, pitch, roll])
	if label == 'Y':
		labels.append(1)
	elif label == 'N':
		labels.append(0)

### Pitch text data processing
while True:
	line = pitch_file.readline()
	if not line: break
	yaw, pitch, roll, label = line.strip().split(' ')
	yaw, pitch, roll = float(yaw), float(pitch), float(roll)
	x_data.append([yaw, pitch, roll])
	if label == 'P':
		labels.append(2)
	elif label == 'N':
		labels.append(0)

x_data = np.array(x_data)
labels = np.array(labels)

x_train, x_test, y_train, y_test = train_test_split(x_data, labels, test_size=0.2, shuffle=True, stratify=labels, random_state=34)

x_train_tensors = Variable(torch.Tensor(x_train))
y_train_tensors = Variable(torch.Tensor(y_train))
x_test_tensors = Variable(torch.Tensor(x_test))
y_test_tensors = Variable(torch.Tensor(y_test))

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # device print(torch.cuda.get_device_name(0))

class LSTM1(nn.Module): 
	def __init__(self, num_classes, input_size, hidden_size, num_layers, seq_length): 
		super(LSTM1, self).__init__() 
		self.num_classes = num_classes #number of classes 
		self.num_layers = num_layers #number of layers 
		self.input_size = input_size #input size 
		self.hidden_size = hidden_size #hidden state 
		self.seq_length = seq_length #sequence length 
		self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True) #lstm 
		self.fc_1 = nn.Linear(hidden_size, 128) #fully connected 1 
		self.fc = nn.Linear(128, num_classes) #fully connected last layer 
		self.relu = nn.ReLU() 
	def forward(self,x): 
		h_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).to(device) #hidden state 
		c_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).to(device) #internal state 
		
		# Propagate input through LSTM 
		
		output, (hn, cn) = self.lstm(x, (h_0, c_0)) #lstm with input, hidden, and internal state 
		hn = hn.view(-1, self.hidden_size) #reshaping the data for Dense layer next 
		out = self.relu(hn) 
		out = self.fc_1(out) #first Dense 
		out = self.relu(out) #relu 
		out = self.fc(out) #Final Output 

		return out

num_epochs = 1000 #1000 epochs
learning_rate = 0.0001 #0.001 lr

input_size = 30 #number of features
hidden_size = 64 #number of features in hidden state
num_layers = 1 #number of stacked lstm layers

num_classes = 1 #number of output classes 
lstm1 = LSTM1(num_classes, input_size, hidden_size, num_layers, x_train_tensors.shape[1]).to(device)

loss_function = torch.nn.CrossEntropyLoss()    # mean-squared error for regression
optimizer = torch.optim.Adam(lstm1.parameters(), lr=learning_rate)  # adam optimizer

for epoch in range(num_epochs):
  outputs = lstm1.forward(x_train_tensors.to(device)) #forward pass
  optimizer.zero_grad() #caluclate the gradient, manually setting to 0

  # obtain the loss function
  loss = loss_function(outputs, y_train_tensors.to(device))

  loss.backward() #calculates the loss of the loss function
 
  optimizer.step() #improve from loss, i.e backprop
  if epoch % 100 == 0:
    print("Epoch: %d, loss: %1.5f" % (epoch, loss.item())) 
 