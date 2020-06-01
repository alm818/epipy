class Compartment:

    def __init__(self, name, alive=True):
        self.name = name


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
        self.from = from


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
