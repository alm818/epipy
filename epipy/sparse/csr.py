import numpy as np
from scipy.sparse import csr_matrix
from numba import jit, prange

# Deprecated
# @jit(nopython=True, parallel=True)
# def coo_tocsr(M, N, data_, row_ind, col_ind):
#     """
#     Numba parallel version of https://github.com/scipy/scipy/blob/3b36a57/scipy/sparse/sparsetools/coo.h#L34
#     and https://github.com/scipy/scipy/blob/3b36a574dc657d1ca116f6e230be694f3de31afc/scipy/sparse/sparsetools/csr.h#L319
#     """
#     # coo_tocsr
#     nnz = len(data_)
#     data = np.zeros(nnz)
#     indices = np.zeros(nnz, dtype=np.int32)
#     indptr = np.zeros(M+1, dtype=np.int32)
#
#     for i in prange(nnz):
#         indptr[row_ind[i]] += 1
#
#     cumsum = 0
#     for i in range(M):
#         temp = indptr[i]
#         indptr[i] = cumsum
#         cumsum += temp
#     indptr[M] = nnz
#
#     for i in prange(nnz):
#         row = int(row_ind[i])
#         dest = indptr[row]
#
#         indices[dest] = col_ind[i]
#         data[dest] = data_[i]
#
#         indptr[row] += 1
#
#     last = 0
#     for i in range(M+1):
#         temp = indptr[i]
#         indptr[i] = last
#         last = temp
#
#     # csr_sort_indices
#     for i in prange(M):
#         row_start = indptr[i]
#         row_end = indptr[i+1]
#
#         temp2 = np.zeros((row_end - row_start, 2))
#         temp2[:,0] = indices[row_start:row_end]
#         temp2[:,1] = data[row_start:row_end]
#
#         sorted_ind = temp2[:,0].argsort()
#         temp2 = temp2[sorted_ind]
#
#         indices[row_start:row_end] = temp2[:,0]
#         data[row_start:row_end] = temp2[:,1]
#
#     return data, indices, indptr

@jit(nopython=True, parallel=True)
def coo_sparse_copy(data, row, col):
    nnz = len(data)
    data_, row_, col_ = np.zeros((3, nnz))
    data_ = np.zeros(nnz)
    row_ = np.zeros(nnz)
    col_ = np.zeros(nnz)
    for i in prange(nnz):
        data_[i] = data[i]
        row_[i] = row[i]
        col_[i] = col[i]
    return data_, row_, col_

@jit(nopython=True, parallel=True)
def csr_sparse_copy(data, indices, indptr):
    M = len(indptr)-1
    nnz = len(data)
    data_ = np.zeros(nnz)
    indices_ = np.zeros(nnz)
    indptr_ = np.zeros(M+1)
    for i in prange(nnz):
        data_[i] = data[i]
        indices_[i] = indices[i]
    for i in prange(M+1):
        indptr_[i] = indptr[i]
    return data_, indices_, indptr_

@jit(nopython=True, parallel=True)
def sparse_vec_multiplication(data, indices, indptr, b):
    M = len(indptr)-1
    res = np.zeros(len(b))
    for row in prange(M):
        for i in prange(indptr[row], indptr[row+1]):
            res[row] += data[i]*b[indices[i]]
    return res

@jit(nopython=True, parallel=True)
def sparse_right_scale(data, indices, indptr, b):
    data_, indices_, indptr_ = csr_sparse_copy(data, indices, indptr)
    M = len(indptr_)-1
    for row in prange(M):
        for i in prange(indptr_[row], indptr_[row+1]):
            data_[i] *= b[int(indices_[i])]
    return data_, indices_, indptr_

@jit(nopython=True, parallel=True)
def sparse_transform(data, indices, indptr, values, row_ind, col_ind, t):
    data_, indices_, indptr_ = csr_sparse_copy(data, indices, indptr)
    for i in prange(len(values)):
        row = int(row_ind[i])
        col = int(col_ind[i])
        left = int(indptr_[row])
        right = int(indptr_[row+1]-1)
        while left <= right:
            mid = int((left+right) / 2)
            if indices_[mid] < col:
                left = mid + 1
            elif indices_[mid] > col:
                right = mid - 1
            else:
                # indices_[mid] == col
                data_[mid] += (values[i]-data_[mid])*t
                break
    return data_, indices_, indptr_

class rigid_csr_matrix:
    """
        Rigid Compressed Sparse Row matrix:
            - a csr matrix that does not allow new nz entries after initialization
            - fast matrix-vector multiplication, matrix-matrix addition, matrix-scaling by utilizing parallelism

        This can be instantiated in several ways:

            rigid_csr_matrix((data, (row_ind, col_ind)), shape=(M, N))
                where ``data``, ``row_ind`` and ``col_ind`` satisfy the
                relationship ``a[row_ind[k], col_ind[k]] = data[k]``.

                Input: row_ind, col_ind should not be duplicated

            rigid_csr_matrix((data, indices, indptr), shape=(M, N))
                is the standard CSR representation where the column indices for
                row i are stored in ``indices[indptr[i]:indptr[i+1]]`` and their
                corresponding values are stored in ``data[indptr[i]:indptr[i+1]]``.
                If the shape parameter is not supplied, the matrix dimensions
                are inferred from the index arrays.

        Attributes
        ----------
        shape : 2-tuple
            Shape of the matrix
        nnz
            Number of stored values, including explicit zeros
        data
            CSR format data array of the matrix
        indices
            CSR format index array of the matrix
        indptr
            CSR format index pointer array of the matrix
    """


    def __init__(self, args, shape):
        self.shape = shape
        if len(args) == 3:
            self.data, self.indices, self.indptr = args
        else:
            wrapper = csr_matrix(args, shape=shape)
            self.data, self.indices, self.indptr = wrapper.data, wrapper.indices, wrapper.indptr
        self.nnz = len(self.data)

    def right_scale(self, vec):
        """sparse matrix multiply with a diagonal matrix with entries vec
        Parameters
        ----------
        vec
            ndarray of shape (n,)
        Returns
        -------
        rigid_csr_matrix
        """
        assert self.shape[1] == vec.shape[0], "Bad dimension"
        data_, indices_, indptr_ = sparse_right_scale(self.data, self.indices, self.indptr, vec)
        return rigid_csr_matrix((data_, indices_, indptr_), shape=self.shape)

    def mul_vec(self, vec):
        """sparse matrix multiply with a vector vec
        Parameters
        ----------
        vec
            ndarray of shape (n,)
        Returns
        -------
        ndarray of shape (n,)
        """
        assert self.shape[1] == vec.shape[0], "Bad dimension"
        return sparse_vec_multiplication(self.data, self.indices, self.indptr, vec)

    def transform(self, values, row_ind, col_ind, t):
        """sparse matrix transform values at row_ind[i],col_ind[i] to data[i] after one unit time for t time
        Note: the row_ind[i], col_ind[i] has to be a nnz entry of the sparse matrix
            values, row_ind, col_ind have to be ndarray
            row_ind, col_ind cannot be repeated
        Examples
        --------
          [[4, 0, 9, 0],
           [0, 7, 0, 0],
           [0, 0, 0, 0],
           [0, 0, 0, 5]]
           values = [6], row_ind = [0], col_ind = [0], t = 0.75
        Returns
        -------
          [[5.5, 0, 9, 0],
           [0, 7, 0, 0],
           [0, 0, 0, 0],
           [0, 0, 0, 5]]
        """
        data_, indices_, indptr_ = sparse_transform(self.data, self.indices, self.indptr, values, row_ind, col_ind, t)
        return rigid_csr_matrix((data_, indices_, indptr_), shape=self.shape)

    def get_csr_matrix(self):
        """return scipy csr_matrix
        """
        return csr_matrix((self.data, self.indices, self.indptr), shape=self.shape)

    def get_transpose(self):
        """return a transpose rigid_csr_matrix
        """
        mat = self.get_csr_matrix().tocoo()
        return rigid_csr_matrix((mat.data, (mat.col, mat.row)), shape=(self.shape[1], self.shape[0]))
