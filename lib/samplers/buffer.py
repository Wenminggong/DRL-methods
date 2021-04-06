import numpy as np
import random
import torch
from collections import deque #双向队列


class MemoryBuffer(object):
    def __init__(self, buffer_size):
        self.buffer = deque(maxlen=buffer_size)
        self.maxSize = buffer_size
        self.len = 0

    def sample(self, count):
        """
        samples a random batch from the replay memory buffer
        :param count: batch size
        :return: batch (numpy array)
        """
        batch = []
        count = min(count, self.len)
        batch = random.sample(self.buffer, count)

        s_tensor = torch.tensor([arr[0] for arr in batch])
        a_tensor = torch.tensor([arr[1] for arr in batch])
        r_tensor = torch.tensor([arr[2] for arr in batch])
        next_s_tensor = torch.tensor([arr[3] for arr in batch])

        return s_tensor, a_tensor, r_tensor, next_s_tensor

    def len(self):
        return self.len

    def add(self, s, a, r, s1):
        """
        adds a particular transaction in the memory buffer
        :param s: current state
        :param a: action taken
        :param r: reward received
        :param s1: next state
        :return:
        """
        transition = (s,a,r,s1)
        self.len += 1
        if self.len > self.maxSize:
            self.len = self.maxSize
        self.buffer.append(transition)
