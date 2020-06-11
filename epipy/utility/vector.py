from numba import jit, prange

@jit(nopython=True, parallel=True)
def multiply(a, b):
    return np.multiply(a, b)

@jit(nopython=True, parallel=True)
def add(a, b):
    return np.add(a, b)

@jit(nopython=True, parallel=True)
def divide(a, b):
    return np.divide(a, b)
