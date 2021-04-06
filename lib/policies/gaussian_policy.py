#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 25 19:41:41 2020

@author: wenminggong

the policy network with Gaussian continuous actions
this policy network can be used on tasks with continuous action spaces
adapted from https://github.com/HeyuanMingong/irl_cs/blob/master/myrllib/policies/normal_mlp.py
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


class GaussianDNNPolicy(nn.Module):
    def __init__(self, input_size, output_size, hidden_sizes=(),
                 nonlinearity=F.relu, init_std=1.0, min_std=1e-6, max_std=1e6):
        super(GaussianDNNPolicy, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        # self.max_action = max_action
        self.hidden_sizes = hidden_sizes
        self.nonlinearity = nonlinearity
        self.num_layers = len(hidden_sizes) + 1
        self.min_log_std = math.log(min_std)
        self.max_log_std = math.log(max_std) 

        layer_sizes = (input_size,) + hidden_sizes
        # all layers are connected with Linear function
        for i in range(1, self.num_layers):
            self.add_module('layer{0}'.format(i), nn.Linear(layer_sizes[i - 1], layer_sizes[i]))
        self.mu = nn.Linear(layer_sizes[-1], output_size)
        
        # view the sigma as a nn parameter
        self.sigma = nn.Parameter(torch.Tensor(output_size))
        self.sigma.data.fill_(math.log(init_std))
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
        mu = F.linear(output, weight=params['mu.weight'],
            bias=params['mu.bias'])
        # select tanh active function in the output [-1, 1]
        mu = torch.tanh(mu)
        # set mu in [-0.1, 0.1]
        # mu = torch.true_divide(mu, torch.ones_like(mu) * 10)
        scale = torch.exp(torch.clamp(params['sigma'], min=self.min_log_std,
                max=self.max_log_std))
        # output is a gaussian function
        # Normal(loc=mu, scale=scale).sample((2,)):sample 2 elements from the gaussian distribution
        return Normal(loc=mu, scale=scale)
