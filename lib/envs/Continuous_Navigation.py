#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 22 10:12:49 2020

@author: wenminggong

2D navigation environment for continuous state and action space
adapted from https://github.com/HeyuanMingong/irl_cs/blob/master/myrllib/envs/navigation.py
"""

import numpy as np
import gym
from gym import spaces


class Navigation2DEnv(gym.Env):
    def __init__(self):
        super(Navigation2DEnv, self).__init__()
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf,
            shape=(2,), dtype=np.float32)
        self.action_space = spaces.Box(low=-0.1, high=0.1,
            shape=(2,), dtype=np.float32)

        ### the default goal position 
        self._goal = np.array([0.5, 0.5], dtype=np.float32)
        self._state = np.array([-0.5, -0.5], dtype=np.float32)

        ### three puddles with different sizes
        self._r_small = 0.1 
        self._r_medium = 0.15
        self._r_large = 0.2
        self._small = np.array([-0.3, 0.2], dtype=np.float32)
        self._medium = np.array([0.25, 0.25], dtype=np.float32)
        self._large = np.array([0.1, -0.1], dtype=np.float32)


    def reset(self):
        self._state = np.array([-0.5,-0.5], dtype=np.float32)
        return self._state

    def step(self, action):
        action = np.clip(action, -0.1, 0.1)
        # assert self.action_space.contains(action)
        temp_state = self._state + action
        navigable = self.check_puddle(temp_state)
        if navigable:
            self._state = temp_state 
            reward_puddle = 0.0
        else:
            reward_puddle = -0.1

        x = self._state[0] - self._goal[0]
        y = self._state[1] - self._goal[1]
        reward_dist = -np.sqrt(x ** 2 + y ** 2)
        # reward_ctrl = - 0.01 * np.square(action).sum()
        reward = reward_dist + reward_puddle
        done = ((np.abs(x) < 0.01) and (np.abs(y) < 0.01))

        return self._state, reward, done, {}

    def check_puddle(self, pos):
        navigable = True 
        x = pos[0]; y = pos[1]
        dist_small = np.sqrt((x-self._small[0])**2 + (y-self._small[1])**2)
        if dist_small <= self._r_small:
            navigable = False 
        dist_medium = np.sqrt((x-self._medium[0])**2 + (y-self._medium[1])**2)
        if dist_medium <= self._r_medium:
            navigable = False 
        dist_large = np.sqrt((x-self._large[0])**2 + (y-self._large[1])**2)
        if dist_large <= self._r_large:
            navigable = False 
        return navigable 
