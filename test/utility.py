import numpy as np

def generate_test(t, *argv):
    """randomly select parameters for t tests
    Parameters
    ----------
    t
        number of tests
    argv
        list of parameters (each with a list of possible values)
    Returns
    -------
    ndarray of shape (t, n_parameters)
    """
    tests = np.empty((t, len(argv)))
    for i in range(len(argv)):
        n_choice = len(argv[i])
        choices = np.random.randint(0, n_choice, t)
        tests[:,i] = np.array([argv[i][x] for x in choices])
    return tests

def sum_duplicated_coo(data, row, col):
    """sum duplicated coo entries
    """
    stored = {}
    nnz = len(data)
    for i in range(nnz):
        p = int(row[i]), int(col[i])
        if p not in stored:
            stored[p] = 0
        stored[p] += data[i]
    data_, row_, col_ = np.zeros((3, len(stored)))
    for i, ((r, c), v) in enumerate(stored.items()):
        data_[i] = v
        row_[i] = r
        col_[i] = c
    return data_, row_, col_
