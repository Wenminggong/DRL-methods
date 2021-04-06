#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 22 11:24:46 2020

@author: wenminggong

The vanilla policy gradient method, REINFORCE 
adapted from https://github.com/HeyuanMingong/irl_cs/blob/master/myrllib/algorithms/reinforce.py
"""

import torch
import numpy as np
from torch.optim import Adam, SGD


class REINFORCE(object):
    def __init__(self, policy, lr=1e-2, device='cpu'):
        # policy: DNN model; lr:learning rate; device = 'cpu' or 'cuda'
        self.policy = policy 
        self.lr = lr
        self.opt = SGD(policy.parameters(), lr=lr)
        self.to(device)

    def inner_loss(self, episodes, return_type):
        returns = episodes.returns
        pi = self.policy(episodes.observations)
        log_probs = pi.log_prob(episodes.actions)
        if log_probs.dim() > 2:
            # if log_probs dimension > 2, we should sum different dimension of log_probs of dnn output
            log_probs = torch.sum(log_probs, dim=2)
        # calcute the mean of all elements
        # loss is scalar
        if return_type == 'causality':
            loss = - torch.mean(log_probs * returns)
        elif return_type == 'total_rewards':
            loss = - torch.mean(log_probs * returns[0])
        else:
            # baseline
            loss = - torch.mean(log_probs * (returns[0] - torch.mean(returns[0])))
        return loss 


    def step(self, episodes, return_type = 'causality'):
        # return_type = 'causality' or 'total_rewards' or 'baseline'
        loss = self.inner_loss(episodes, return_type)
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()


    def to(self, device, **kwargs):
        self.policy.to(device, **kwargs)
        self.device = device