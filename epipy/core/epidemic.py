import numpy as np

class Compartment:
    def __init__(self, name, alive=True):
        self.name = name
    def _set_index(self, index):
        self.index = index;

class Edge:
    def __init__(self, rate):
        self.rate = rate

class TwoEdge(Edge):
    def __init__(self, rate, X, Y):
        super().__init__(rate)
        self.X = X
        self.Y = Y

class OneEdge(Edge):
    def __init__(self, rate, X):
        super().__init__(rate)
        self.X = X

class TransitionEdge(TwoEdge):
    def __init__(self, rate, X, Y):
        super().__init__(rate, X, Y)

class InfectionEdge(TwoEdge):
    def __init__(self, rate, X, Y, matrix):
        super().__init__(rate, X, Y)
        self.matrix = matrix

class InwardEdge(OneEdge):
    def __init__(self, rate, X):
        super().__init__(rate, X)

class OutwardEdge(OneEdge):
    def __init__(self, rate, X):
        super().__init__(rate, X)

class Epidemic:
    def __init__(self):
        self.compartments = []
        self.edges = []
        self.infector = {}
        self.pop = None

    def init(self):
        for index, com in enumerate(self.compartments):
            com._set_index(index)

    def infect(self):
        raise NotImplementedError

    def run(self, T):
        self.init()
        self.infect()
        self.y = np.zeros((self.pop.N, len(self.compartments)))
        self.y[:, 0] = self.pop.pop
        for (patch_index, com_index), val in self.infector.items():
            if com_index > 0:
                assert self.y[patch_index, 0] >= val, "Infector cannot infect more than the population"
                self.y[patch_index, 0] -= val
                self.y[patch_index, com_index] += val
        y = self.forward(0, self.y)

    def forward(self, t, y):
        current_pop = np.matmul(y, np.ones((len(self.compartments), 1)))
        dy = np.zeros((self.pop.N, len(self.compartments)))
        for edge in self.edges:
            if isinstance(edge, InwardEdge):
                dy[:, edge.X.index] += current_pop.squeeze()*edge.rate
            elif isinstance(edge, OutwardEdge):
                dy[:, edge.X.index] -= y[:, edge.X.index]*edge.rate
            elif isinstance(edge, TransitionEdge):
                tmp = y[:, edge.X.index]*edge.rate
                dy[:, edge.X.index] -= tmp
                dy[:, edge.Y.index] += tmp
            elif isinstance(edge, InfectionEdge):
                P = np.asarray(edge.matrix)
                probability = np.divide(np.matmul(P.T, y[:, edge.Y.index][np.newaxis]), np.matmul(P.T, current_pop))
                PB = np.matmul(np.multiply(P, np.diag([edge.rate])), probability)
                tmp = np.multiply(y[:, edge.X.index][np.newaxis], PB).squeeze()
                dy[:, edge.X.index] -= tmp
                dy[:, edge.Y.index] += tmp
            else:
                raise TypeError("Unknown Edge:", edge)
        return y + dy
