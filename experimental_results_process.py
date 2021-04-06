#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  6 14:58:09 2020

@author: wenminggong

plot the experimental results
"""

import numpy as np
import os
import matplotlib.pyplot as plt
import sys


if __name__ == "__main__":
    print(sys.version)
    
    rewards_a2c_general = np.load(os.path.join('saves/navigation2d', 'a2c_general_Navigation2DEnv_rewards.npy'))
    rewards_a2c_baseline = np.load(os.path.join('saves/navigation2d', 'a2c_baseline_Navigation2DEnv_rewards.npy'))
    rewards_reinforce_baseline = np.load(os.path.join('saves/navigation2d', 'reinforce_baseline_Navigation2DEnv_rewards.npy'))
    
    x = np.arange(0, 500, 20)
    
    plt.figure(figsize=(6,4))
    plt.plot(x, rewards_a2c_general[x], lw=2, c = 'r', marker = 'o', label = 'A2C_general')
    plt.plot(x, rewards_a2c_baseline[x], lw=2, c = 'g', marker = 's', label = 'A2C_baseline')
    plt.plot(x, rewards_reinforce_baseline[x], lw=2, c = 'b', marker = '^', label = 'REINFORCE = '+'Baseline')
    
    plt.legend()
    plt.ylabel("Average return")
    plt.xlabel("Policy iterations")
    plt.title("A2C vs REINFORCE in Navigation 2D")
    plt.grid(axis = 'y', ls = '--')
    plt.savefig('a2c_vs_reinforce_2', format='eps')
    plt.show()
