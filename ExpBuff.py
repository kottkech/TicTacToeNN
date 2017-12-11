import numpy as np
import random

class expBuff():
    def __init__(self, bSize = 1000):
        self.buffer = []
        self.bSize = bSize

    def add(self,exp):
        if len(self.buffer) + len(exp) >= self.bSize:
            self.buffer[0:(len(exp)+len(self.buffer))-self.bSize] = []
        self.buffer.extend(exp)

    def sample(self,size):
        return np.reshape(np.array(random.sample(self.buffer,size)),[size,5])