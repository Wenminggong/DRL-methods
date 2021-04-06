#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  5 13:59:57 2021

@author: wenminggong

basic batch actor-critic algorithm, A2C
"""

import torch
import numpy as np
from torch.optim import Adam, SGD


class A2C(object):
    def __init__(self, policy, critic, lr=1e-2, gamma=0.95, device='cpu'):
        # policy: DNN model; critic: state-value function critic; lr:learning rate; device = 'cpu' or 'cuda'
        self.policy = policy
        self.critic = critic
        self.lr = lr
        self.gamma = gamma
        self.policy_opt = SGD(self.policy.parameters(), lr=0.01)
        self.critic_opt = SGD(self.critic.parameters(), lr=0.001)
        self.to(device)


    def critic_loss(self, episodes):
        v = self.critic(episodes.observations)
        if v.dim() > 2:
            # if v dimension > 2, we should sum different dimension of v of dnn output
            v = torch.sum(v, dim=2)
        # print(v.shape)
        with torch.set_grad_enabled(False):
            v_next_state = self.critic(episodes.observations[1:])
            v_terminal = torch.zeros((1,) + v_next_state.shape[1:])
            v_next_state = torch.cat([v_next_state, v_terminal], dim = 0)
            if v_next_state.dim() > 2:
            # if v_next_state dimension > 2, we should sum different dimension of v_next_state of dnn output
                v_next_state = torch.sum(v_next_state, dim=2)
            v_label = episodes.rewards + self.gamma * v_next_state
        # print(v_next_state.shape)
        # print(episodes.rewards.shape)
        loss = torch.nn.MSELoss() #loss是一个类的实例
        output = loss(v, v_label)
        # print(v.requires_grad)
        # print(v_label.requires_grad)
        return output


    def policy_loss(self, episodes, return_type):
        returns = episodes.returns
        pi = self.policy(episodes.observations)
        log_probs = pi.log_prob(episodes.actions)
        if log_probs.dim() > 2:
            # if log_probs dimension > 2, we should sum different dimension of log_probs of dnn output
            log_probs = torch.sum(log_probs, dim=2)
        # compute advantage function
        with torch.set_grad_enabled(False):
            v = self.critic(episodes.observations)
            v_terminal = torch.zeros((1,) + v.shape[1:])
            v_next_state = torch.cat([v[1:], v_terminal])
            if v.dim() > 2:
            # if v dimension > 2, we should sum different dimension of v of dnn output
                v = torch.sum(v, dim=2)
            if v_next_state.dim() > 2:
            # if v_next_state dimension > 2, we should sum different dimension of v_next_state of dnn output
                v_next_state = torch.sum(v_next_state, dim=2)
            Advantages = episodes.rewards + self.gamma * v_next_state - v
        
        # calcute the mean of all elements
        # loss is scalar
        if return_type == 'general':
            loss = - torch.mean(log_probs * Advantages)
        else:
            # baseline
            loss = - torch.mean(log_probs * (returns - Advantages))
        return loss 


    def step(self, episodes, return_type = 'general'):
        # return_type = 'general' or 'baseline'
        critic_update_steps = 1
        for i in range(critic_update_steps):
            criticloss = self.critic_loss(episodes)
            # print(criticloss.requires_grad)
            self.critic_opt.zero_grad()
            # self.policy_opt.zero_grad()
            criticloss.backward()
            self.critic_opt.step()
            # self.policy_opt.step()
        
        policyloss = self.policy_loss(episodes, return_type)
        self.policy_opt.zero_grad()
        policyloss.backward()
        self.policy_opt.step()


    def to(self, device, **kwargs):
        self.policy.to(device, **kwargs)
        self.device = device
