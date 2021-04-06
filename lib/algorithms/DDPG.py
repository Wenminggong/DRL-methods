#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  7 20:44:20 2021

@author: wenminggong

deep deterministic policy gradient, DDPG
"""

import torch
import numpy as np
from torch.optim import Adam, SGD


class DDPG(object):
    def __init__(self, policy, critic, target_policy, target_critic, lr=1e-2, gamma=0.95, device='cpu'):
        # policy: DNN model; critic: state-action value function critic; lr:learning rate; device = 'cpu' or 'cuda'
        self.policy = policy
        self.critic = critic
        self.target_policy = target_policy
        self.target_critic = target_critic
        self.lr = lr
        self.gamma = gamma
        self.policy_opt = Adam(self.policy.parameters(), lr=self.lr)
        self.critic_opt = Adam(self.critic.parameters(), lr=self.lr)
        self.to(device)
        
        self.target_policy.load_state_dict(self.policy.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())


    def critic_loss(self, states, actions, rewards, next_states):
        q = self.critic(states, actions)
        # if q.dim() > 2:
        #     # if q dimension > 2, we should sum different dimension of q of dnn output
        #     q = torch.sum(q, dim=2)
        # print(q.shape)
        with torch.set_grad_enabled(False):
            next_state_actions = self.target_policy(next_states)
            q_next_state_action = self.target_critic(next_states, next_state_actions)
            # if q_next_state_action.dim() > 2:
            # # if q_next_state dimension > 2, we should sum different dimension of q_next_state of dnn output
            #     q_next_state_action = torch.sum(q_next_state_action, dim=2)
            q_label = rewards + self.gamma * q_next_state_action
        # print(q_next_state_action.shape)
        # print(episodes.rewards.shape)
        loss = torch.nn.MSELoss()
        output = loss(q, q_label.float())
        return output


    def policy_loss(self, states):
        actions = self.policy(states)
        # if actions.dim() > 2:
        #     actions = torch.sum(actions, dim=2)
        # with torch.set_grad_enabled(False):
        q = self.critic(states, actions)
        # if q.dim() > 2:
        # # if q dimension > 2, we should sum different dimension of q of dnn output
        #     q = torch.sum(q, dim=2)
        loss = -1 * torch.mean(q)
        return loss 


    def step(self, states, actions, rewards, next_states):
        critic_update_steps = 1
        for i in range(critic_update_steps):
            criticloss = self.critic_loss(states, actions, rewards, next_states)
            # print('critic_loss:{}'.format(criticloss))
            self.critic_opt.zero_grad()
            criticloss.backward()
            self.critic_opt.step()
        
        
        policyloss = self.policy_loss(states)
        # print('policy_loss:{}'.format(policyloss))
        self.policy_opt.zero_grad()
        policyloss.backward()
        self.policy_opt.step()
        
        
    def soft_update_policy(self, tau=0.99):
        """
        Copies the parameters from policy to target_policy using the below update
        y = tau*y + (1 - tau)*x
        """
        for target_param, param in zip(self.target_policy.parameters(), self.policy.parameters()):
            target_param.data.copy_(
                target_param.data * tau + param.data * (1-tau))
            
            
    def soft_update_critic(self, tau=0.99):
        """
        Copies the parameters from policy to target_policy using the below update
        y = tau*y + (1 - tau)*x
        """
        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(
                target_param.data * tau + param.data * (1-tau))
            
            
    def hard_update_policy(self):
        """
        Copies the parameters from policy to target_policy using the below update
        y = x
        """
        for target_param, param in zip(self.target_policy.parameters(), self.policy.parameters()):
            target_param.data.copy_(param.data)
            
            
    def hard_update_critic(self):
        """
        Copies the parameters from policy to target_policy using the below update
        y = x
        """
        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data)


    def to(self, device, **kwargs):
        self.policy.to(device, **kwargs)
        self.device = device
