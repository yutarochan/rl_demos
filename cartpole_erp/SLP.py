'''
Single Layer Perceptron
Author: Yuya Jeremy Ong (yuyajeremyong@gmail.com)
'''
from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F

class SLP(nn.Module):
    def __init__(self, input_shape, output_shape, hidden_shape=40, device=torch.device('cpu')):
        super(SLP, self).__init__()

        self.device = device
        self.input_shape = input_shape[0]
        self.hidden_shape = hidden_shape
        self.output_shape = output_shape

        self.linear = nn.Linear(self.input_shape, self.hidden_shape)
        self.out = nn.Linear(self.hidden_shape, self.output_shape)

    def forward(self, x):
        x = torch.from_numpy(x).float().to(self.device)
        x = F.relu(self.linear(x))
        x = self.out(x)

        return x
