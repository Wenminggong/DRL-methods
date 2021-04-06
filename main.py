#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  1 19:37:29 2020

@author: wenminggong

main for training policy gradient algorithms in multi-environments
"""

### common lib
import sys
import gym
import numpy as np
import argparse 
import torch
from tqdm import tqdm #长循环中显示进度条
import os
import time 
from torch.optim import Adam, SGD 
import scipy.io as sio
import copy 
from collections import OrderedDict

### personal lib
from lib.envs.Continuous_Navigation import Navigation2DEnv
from lib.samplers.episode import BatchEpisodes 
from lib.samplers.sampler import BatchSampler 
from lib.policies.gaussian_policy import GaussianDNNPolicy 
from lib.policies.critic_state_value_function import StateValueCritic
from lib.algorithms.REINFORCE import REINFORCE
from lib.algorithms.TRPO import TRPO 
from lib.algorithms.NPG import NPG 
from lib.algorithms.A2C import A2C


#-----------------arguments-----------------------
def Args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_workers', type=int, default=4, 
        help='number of cpu processors for parallelly sampling of gym environment')
    
    return parser.parse_args() 


if __name__ == '__main__':
    print(sys.version)
    start_time = time.time()
    # set env name
    env_name = 'Navigation2DEnv' 
    # env_name = 'HalfCheetah-v2'
    # set batch size
    batch_size = 20
    # set nums processers
    num_workers = 7
    
    # generate a sampler
    sampler = BatchSampler(env_name, batch_size, num_workers)
    
    state_dim = int(np.prod(sampler.envs.observation_space.shape))
    action_dim = int(np.prod(sampler.envs.action_space.shape))
    print('state dim: %d; action dim: %d' %(state_dim,action_dim))
    
    model_path = 'saves/navigation2d'
    # model_path = 'saves/mujoco'
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    
    hidden_size = 100
    num_layers = 2
    #create a policy
    policy = GaussianDNNPolicy(state_dim, action_dim, hidden_sizes=(hidden_size,) * num_layers)
    # create a learner
    lr = 0.005
    # set the discount rate gamma
    gamma = 0.99
    device = 'cuda'
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    
    # set learner type: reinforce or trpo or npg or a2c
    learner_type = 'a2c'
    
    if learner_type == 'reinforce':
        learner = REINFORCE(policy, lr=lr, device=device)
        return_type = 'baseline'
    elif learner_type == 'trpo':
        learner = TRPO(policy, device=device)
    elif learner_type == 'npg':
        learner = NPG(policy, device=device)
    elif learner_type == 'a2c':
        critic = StateValueCritic(state_dim, hidden_sizes=(hidden_size,) * num_layers)
        learner = A2C(policy, critic, lr=lr, gamma = gamma, device=device)
        return_type = 'baseline'
    # set the number of policy iterations 
    num_iter = 500
    rewards = np.zeros(num_iter)
    
    # select the type of reinforce: 'causality' or 'total_rewards' or 'baseline'
    # return_type = 'causality'
    #-------training-------------------- 
    for idx in tqdm(range(num_iter)):
        episodes = sampler.sample(policy, gamma = gamma, device=device)
        rewards[idx] = episodes.evaluate()
        learner.step(episodes, return_type) 
        # learner.step(episodes)
        print('average return:', rewards[idx])
        print('std:', torch.exp(OrderedDict(policy.named_parameters())['sigma']))

    ### save the model
    # name_model = os.path.join(model_path, 'reinforce_'+return_type+'_' + str(lr) +env_name+'_model.pkl')
    name_model = os.path.join(model_path, learner_type+'_'+return_type+'_'+env_name + '_model.pkl')
    print('Save the model to %s' %name_model)
    torch.save(policy, name_model)
    name_critic = os.path.join(model_path, learner_type+'_'+return_type+'_'+env_name + '_critic.pkl')
    print('Save the model to %s' %name_critic)
    torch.save(critic, name_critic)
    # name_reward = os.path.join(model_path, 'reinforce_'+return_type+'_'+str(lr)+env_name+'_rewards.npy')
    name_reward = os.path.join(model_path, learner_type+'_'+return_type+'_'+env_name + '_rewards.npy')
    print('Save the rewards to %s' %name_reward)
    np.save(name_reward, rewards)
    
    print('Running time: %.2f' %(time.time()-start_time))







