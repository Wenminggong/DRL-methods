#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  1 20:31:50 2020

@author: wenminggong

define the class of episodes which contain states, actions, rewards and returns 
copy from https://github.com/HeyuanMingong/irl_cs/blob/master/myrllib/episodes/episode.py
"""

import numpy as np
import torch
import torch.nn.functional as F

class BatchEpisodes(object):
    def __init__(self, batch_size, gamma=0.95, device='cpu'):
        self.batch_size = batch_size
        self.gamma = gamma
        self.device = device

        self._observations_list = [[] for _ in range(batch_size)] # the element is also a list
        self._actions_list = [[] for _ in range(batch_size)]
        self._log_probs_list = [[] for _ in range(batch_size)]
        self._rewards_list = [[] for _ in range(batch_size)]
        # self._mask_list = []

        self._observations = None
        self._actions = None
        self._log_probs = None
        self._rewards = None
        self._returns = None
        self._mask = None

    @property  # next function can be referred as a property
    def observations(self):
        if self._observations is None:
            observation_shape = self._observations_list[0][0].shape
            # observations: dim0: time steps; dim1: episodes; dim2: observation_shape
            observations = np.zeros((len(self), self.batch_size) + observation_shape, dtype=np.float32)
            for i in range(self.batch_size):
                length = len(self._observations_list[i])
                observations[:length, i] = np.stack(self._observations_list[i], axis=0)
            self._observations = torch.from_numpy(observations).to(self.device)
        return self._observations

    @property
    def actions(self):
        if self._actions is None:
            action_shape = self._actions_list[0][0].shape
            actions = np.zeros((len(self), self.batch_size)
                + action_shape, dtype=np.float32)
            for i in range(self.batch_size):
                length = len(self._actions_list[i])
                actions[:length, i] = np.stack(self._actions_list[i], axis=0)
            self._actions = torch.from_numpy(actions).to(self.device)
        return self._actions

    @property
    def log_probs(self):
        if self._log_probs is None:
            log_prob_shape = self._log_probs_list[0][0].shape
            log_probs = np.zeros((len(self), self.batch_size)
                + log_prob_shape, dtype=np.float32)
            for i in range(self.batch_size):
                length = len(self._log_probs_list[i])
                log_probs[:length, i] = np.stack(self._log_probs_list[i], axis=0)
            self._log_probs = torch.from_numpy(log_probs).to(self.device)
        return self._log_probs

    @property
    def rewards(self):
        if self._rewards is None:
            rewards = np.zeros((len(self), self.batch_size), dtype=np.float32)
            for i in range(self.batch_size):
                length = len(self._rewards_list[i])
                rewards[:length, i] = np.stack(self._rewards_list[i], axis=0)
            self._rewards = torch.from_numpy(rewards).to(self.device)
        return self._rewards

    @property
    def returns(self):
        if self._returns is None:
            return_ = np.zeros(self.batch_size, dtype=np.float32)
            returns = np.zeros((len(self), self.batch_size), dtype=np.float32)
            rewards = self.rewards.cpu().numpy()
            # mask = self.mask.cpu().numpy()
            #????????????
            for i in range(len(self) - 1, -1, -1):
                # return_ = self.gamma * return_ + rewards[i] * mask[i]
                return_ = self.gamma * return_ + rewards[i]
                returns[i] = return_
            self._returns = torch.from_numpy(returns).to(self.device)
        return self._returns

    @property
    def mask(self):
        if self._mask is None:
            mask = np.zeros((len(self), self.batch_size), dtype=np.float32)
            for i in range(self.batch_size):
                length = len(self._actions_list[i])
                mask[:length, i] = 1.0
            self._mask = torch.from_numpy(mask).to(self.device)
        return self._mask

    # def gae(self, values, tau=1.0):
    #     # Add an additional 0 at the end of values for
    #     # the estimation at the end of the episode
    #     values = values.squeeze(2).detach()
    #     values = F.pad(values * self.mask, (0, 0, 0, 1))

    #     deltas = self.rewards + self.gamma * values[1:] - values[:-1]
    #     advantages = torch.zeros_like(deltas).float()
    #     gae = torch.zeros_like(deltas[0]).float()
    #     for i in range(len(self) - 1, -1, -1):
    #         gae = gae * self.gamma * tau + deltas[i]
    #         advantages[i] = gae

    #     return advantages

    def append(self, observations, actions, rewards, batch_ids, log_probs=None):
        if log_probs is None:
            log_probs = np.zeros(actions.shape)
        for observation, action, reward, batch_id, log_prob in zip(
                observations, actions, rewards, batch_ids, log_probs):
            if batch_id is None:
                continue
            self._observations_list[batch_id].append(observation.astype(np.float32))
            self._actions_list[batch_id].append(action.astype(np.float32))
            self._rewards_list[batch_id].append(reward.astype(np.float32))
            self._log_probs_list[batch_id].append(log_prob.astype(np.float32))

    def __len__(self):
        # return the length of the longest episode
        return max(map(len, self._rewards_list))

    def evaluate(self):
        # return the mean return of batchsize episodes
        return torch.mean(torch.sum(self.rewards, dim=0)).item()