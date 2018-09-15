'''
Experience Replay Mamory Data Structure
Author: Yuya Jeremy Ong (yuyajeremyong@gmail.com)
'''
from __future__ import print_function
import random
from collections import namedtuple

Experience = namedtuple('Experience', ['obs', 'action', 'reward', 'next_obs', 'done'])

# Cylic/Ring Buffer based Experience Memory Structure
class ExperienceMemory(object):
    def __init__(self, capacity=int(1e6)):
        self.capacity = capacity
        self.mem_idx = 0
        self.memory = []

    def store(self, exp):
        # Expand Capacity of Memory Structure
        if self.mem_idx < self.capacity: self.memory.append(None)

        self.memory[self.mem_idx % self.capacity] = exp
        self.mem_idx += 1

    def sample(self, batch_size):
        assert batch_size <= len(self.memory)
        return random.sample(self.memory, batch_size)

    def get_size(self):
        return len(self.memory)
