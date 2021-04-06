#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  2 15:29:42 2020

@author: wenminggong

adapted from https://github.com/HeyuanMingong/irl_cs/blob/master/myrllib/envs/subproc_vec_env.py
"""

import numpy as np
import multiprocessing as mp
import gym
import sys
is_py2 = (sys.version[0] == '2')
if is_py2:
    import Queue as queue
else:
    import queue as queue

class EnvWorker(mp.Process):
    def __init__(self, remote, env_fn, queue, lock):
        super(EnvWorker, self).__init__()
        self.remote = remote
        self.env = env_fn()
        
        self.queue = queue
        self.lock = lock
        self.task_id = None
        self.done = False
        self.episode_steps = 0

    def empty_step(self):
        observation = np.zeros(self.env.observation_space.shape,
                               dtype=np.float32)
        reward, done = 0.0, True
        return observation, reward, done, {}

    def try_reset(self):
        self.episode_steps = 0
        with self.lock:
            # if self.queue.empty():
            #     print('queue empty')
            #     self.task_id = None
            #     self.done = True
            # else:
            #     self.task_id = self.queue.get(True)
            #     print('batch id:', self.task_id)
            #     self.done = False
            try:
                self.task_id = self.queue.get(True)
                # print('batch id:', self.task_id)
                self.done = (self.task_id is None)
            except queue.Empty:
                self.task_id = None
                self.done = True
        observation = (np.zeros(self.env.observation_space.shape,
            dtype=np.float32) if self.done else self.env.reset())
        return observation

    def run(self):
        while True:
            command, data = self.remote.recv()
            if command == 'step':
                observation, reward, done, info = (self.empty_step()
                    if self.done else self.env.step(data))
                # 最多探索100步
                # print('step:',self.episode_steps)
                self.episode_steps += 1
                if self.episode_steps == 100:
                    done = True
                if done and (not self.done):
                    observation = self.try_reset()
                self.remote.send((observation, reward, done, self.task_id, info))
            elif command == 'reset':
                observation = self.try_reset()
                self.remote.send((observation, self.task_id))
            # elif command == 'reset_task':
            #     self.env.unwrapped.reset_task(data)
            #     self.remote.send(True)
            # elif command == 'close':
            #     self.remote.close()
            #     break
            elif command == 'get_spaces':
                self.remote.send((self.env.observation_space, self.env.action_space))
            else:
                raise NotImplementedError()

class SubprocVecEnv(gym.Env):
    def __init__(self, env_factory, queue):
        # env_factory: a list of the addresses of a function; queue: a mp queue
        self.lock = mp.Lock()
        # create multi pipes
        self.remotes, self.work_remotes = zip(*[mp.Pipe() for _ in env_factory])
        # create multi processes
        self.workers = [EnvWorker(remote, env_fn, queue, self.lock) for (remote, env_fn) in zip(self.work_remotes, env_factory)]
        # worker is a process
        for worker in self.workers:
            # 守护进程，主进程结束子进程也结束
            worker.daemon = True
            worker.start()
        for remote in self.work_remotes:
            remote.close()
        # self.waiting = False
        # self.closed = False

        self.remotes[0].send(('get_spaces', None))
        #管道读取时，如果为空，则自动阻塞等待
        observation_space, action_space = self.remotes[0].recv()
        self.observation_space = observation_space
        self.action_space = action_space

    def step(self, actions):
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))
            
        results = [remote.recv() for remote in self.remotes]
        observations, rewards, dones, task_ids, infos = zip(*results)
        return np.stack(observations), np.stack(rewards), np.stack(dones), task_ids, infos

    # def step_async(self, actions):
    #     for remote, action in zip(self.remotes, actions):
    #         remote.send(('step', action))
    #     self.waiting = True

    # def step_wait(self):
    #     results = [remote.recv() for remote in self.remotes]
    #     self.waiting = False
    #     observations, rewards, dones, task_ids, infos = zip(*results)
    #     return np.stack(observations), np.stack(rewards), np.stack(dones), task_ids, infos

    def reset(self):
        for remote in self.remotes:
            remote.send(('reset', None))
        results = [remote.recv() for remote in self.remotes]
        observations, task_ids = zip(*results)
        return np.stack(observations), task_ids

    # def reset_task(self, tasks):
    #     for remote, task in zip(self.remotes, tasks):
    #         remote.send(('reset_task', task))
    #     return np.stack([remote.recv() for remote in self.remotes])

    # def close(self):
    #     if self.closed:
    #         return
    #     if self.waiting:
    #         for remote in self.remotes:            
    #             remote.recv()
    #     for remote in self.remotes:
    #         remote.send(('close', None))
    #     for worker in self.workers:
    #         worker.join()
    #     self.closed = True