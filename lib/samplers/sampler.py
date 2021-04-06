#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  1 19:22:06 2020

@author: wenminggong

sampling the states and actions from interacting with the environment
copy from https://github.com/HeyuanMingong/irl_cs/blob/master/myrllib/samplers/sampler.py
"""

import gym
import torch
import multiprocessing as mp
import numpy as np 
from lib.envs.subproc_vec_env import SubprocVecEnv
from lib.samplers.episode import BatchEpisodes
from lib.envs.Continuous_Navigation import Navigation2DEnv

def make_env(env_name):
    def _make_env():
        if env_name == 'Navigation2DEnv':
            return Navigation2DEnv()
        return gym.make(env_name)
    # return function's address
    return _make_env


class BatchSampler(object):
    def __init__(self, env_name, batch_size, num_workers=mp.cpu_count() - 1):
        self.env_name = env_name
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        self.queue = mp.Queue()
        self.envs = SubprocVecEnv([make_env(env_name) for _ in range(num_workers)], queue=self.queue)
        # self._env = gym.make(env_name)

    def sample(self, policy, params=None, gamma=0.95, device='cpu'):
        episodes = BatchEpisodes(batch_size=self.batch_size, gamma=gamma, device=device)
        for i in range(self.batch_size):
            # put batch id into the queue
            self.queue.put(i)
        # # 向queue中传输batch结束的信号
        for _ in range(self.num_workers):
            self.queue.put(None)
        observations, batch_ids = self.envs.reset()
        dones = [False]
        # 当队列queue不为空或者有一个done为false时，执行循环
        # while (not all(dones)) or (not self.queue.empty()):
        while (not self.queue.empty()) or (not all(dones)):
            # 前向采样不需要计算梯度
            with torch.no_grad():
                observations_tensor = torch.from_numpy(observations).to(device=device)
                pi = policy.forward(observations_tensor, params=params)
                # actions_tensor = torch.clamp(pi.sample(), min = -0.1, max = 0.1)
                actions_tensor = pi.sample()
                actions = actions_tensor.cpu().numpy()
                # print(actions)
                log_probs_tensor = pi.log_prob(actions_tensor)
                log_probs = log_probs_tensor.cpu().numpy()
            new_observations, rewards, dones, new_batch_ids, _ = self.envs.step(actions)
            episodes.append(observations, actions, rewards, batch_ids, log_probs)
            observations, batch_ids = new_observations, new_batch_ids
        # print('queue:', self.queue.get())
        return episodes

    # def reset_task(self, task):
    #     tasks = [task for _ in range(self.num_workers)]
    #     reset = self.envs.reset_task(tasks)
    #     return all(reset)

    # def sample_tasks(self, num_tasks):
    #     tasks = self._env.unwrapped.sample_tasks(num_tasks)
    #     return tasks
