"""
@author: Mohamed Hosny Ahmed
@date: 17 / 5 / 2016
@purpose: generate random numbers
"""

import time
import numpy as np

class RandomGenerator(object):
    def __init__(self, start, end, size, spread=0.5):
        self.x = 0
        self.random = []
        self.start = start+1
        self.valueAsString = ""
        self.end = end
        self.size = size
        self.b = 0
        self.spread = spread

    def rand_generator(self):
        self.random.clear()
        self.run()

    def run(self):
        if self.b != 0:
            self.b *= self.spread
        else:
            self.b = self.initGenerator()

        for i in range(1, self.size + 1):
            old = np.mod(self.b, self.end)
            self.random.append(old)
            self.b += old

    def initGenerator(self):
        self.valueAsString = str(time.time())
        index = self.valueAsString.index(".")
        self.valueAsString = self.valueAsString[index:]
        value = np.float(self.valueAsString)
        self.b = value * self.size * self.start
        return self.b

    def get_random_vector(self):
        return np.float16(self.random)