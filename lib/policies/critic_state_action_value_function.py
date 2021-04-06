#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  7 20:40:35 2021

@author: wenminggong

the critic network for approximating the state-action value function
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict


# initialize the parameters of neural network (how?)
def weight_init(module):
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight, gain = math.sqrt(2))
        module.bias.data.zero_()


class StateActionValueCritic(nn.Module):
    def __init__(self, state_size, action_size, hidden_sizes=(),
                 nonlinearity=F.relu):
        super(StateActionValueCritic, self).__init__()
        self.input_size = state_size + action_size
        self.hidden_sizes = hidden_sizes
        self.nonlinearity = nonlinearity
        self.num_layers = len(hidden_sizes) + 1

        layer_sizes = (self.input_size,) + hidden_sizes
        # all layers are connected with Linear function
        for i in range(1, self.num_layers):
            self.add_module('layer{0}'.format(i), nn.Linear(layer_sizes[i - 1], layer_sizes[i]))
        self.q = nn.Linear(layer_sizes[-1], 1)
        
        # initilize the parameters of DNN
        self.apply(weight_init)

    def forward(self, state, action, params=None):
        if params is None:
            params = OrderedDict(self.named_parameters())
        input = torch.cat([state, action], dim = 1)
        output = torch.tensor(input, dtype = torch.float32)
        # print(output.type())
        for i in range(1, self.num_layers):
            output = F.linear(output,
                weight=params['layer{0}.weight'.format(i)],
                bias= params['layer{0}.bias'.format(i)])
            output = self.nonlinearity(output)
        q = F.linear(output, weight=params['q.weight'],
            bias=params['q.bias'])
        return q
