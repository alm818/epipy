import unittest
from epipy.core import *
import numpy as np

class OnePatchMobility(Mobility):
    def __init__(self):
        self.matrix = [1.0]

class OnePatchPopulation(Population):
    def __init__(self):
        self.N = 1
        self.pop = [10000]

class VitalSIR(Epidemic):
    def __init__(self, pop):
        S = Compartment("Susceptible")
        I = Compartment("Infectious")
        R = Compartment("Recovered")
        self.compartments = [S, I, R]

        mu = 0.0001
        beta = 0.2
        gamma = 0.05
        mob = OnePatchMobility()

        self.edges = [
            InwardEdge(mu*2, S),
            OutwardEdge(mu, S),
            OutwardEdge(mu, I),
            OutwardEdge(mu, R),
            TransitionEdge(gamma, I, R),
            InfectionEdge(beta, S, I, np.asarray(mob.matrix))
        ]

        self.pop = pop

    def infect(self):
        self.infector = {(0,1):10.0}

class Test(unittest.TestCase):
    def test(self):
        pop = OnePatchPopulation()
        model = VitalSIR(pop)
        model.run(365)

if __name__ == '__main__':
    unittest.main()
