#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  7 20:33:03 2021

@author: wenminggong

deterministic policy
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

from collections import OrderedDict


# initialize the parameters of neural network (how?)
def weight_init(module):
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight, gain = math.sqrt(2))
        module.bias.data.zero_()


class DeterministicDNNPolicy(nn.Module):
    def __init__(self, input_size, output_size, max_action, hidden_sizes=(),
                 nonlinearity=F.relu):
        super(DeterministicDNNPolicy, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.max_action = max_action
        self.hidden_sizes = hidden_sizes
        self.nonlinearity = nonlinearity
        self.num_layers = len(hidden_sizes) + 1

        layer_sizes = (input_size,) + hidden_sizes
        # all layers are connected with Linear function
        for i in range(1, self.num_layers):
            self.add_module('layer{0}'.format(i), nn.Linear(layer_sizes[i - 1], layer_sizes[i]))
        self.action = nn.Linear(layer_sizes[-1], output_size)
        
        # initilize the parameters of DNN
        self.apply(weight_init)

    def forward(self, input, params=None):
        if params is None:
            params = OrderedDict(self.named_parameters())
        output = torch.tensor(input, dtype = torch.float32)
        # print(output.type())
        for i in range(1, self.num_layers):
            output = F.linear(output,
                weight=params['layer{0}.weight'.format(i)],
                bias= params['layer{0}.bias'.format(i)])
            output = self.nonlinearity(output)
        action = F.linear(output, weight=params['action.weight'],
            bias=params['action.bias'])
        action = torch.tanh(action) * self.max_action
        # action = torch.tanh(action)
        return action
    
    
    def get_exploration_action(self, action, sigma):
        sigma = self.max_action * sigma
        new_action = Normal(loc=action, scale=sigma).sample()
        return new_action
