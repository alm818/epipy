import unittest
from epipy.core import *
import numpy as np

class TwoPatchMobility(Mobility):
    def __init__(self):
        self.matrix = [
            [0.8, 0.2],
            [0.2, 0.8]
        ]

class TwoPatchPopulation:
    def __init__(self):
        self.N = 2
        self.pop = [10000, 10000]

class Test(unittest.TestCase):
    def test(self):
        self.assertTrue(True)

if __name__ == '__main__':
    unittest.main()
