import unittest
from epipy.sparse import rigid_csr_matrix
from scipy.sparse import csr_matrix
from test import generate_test, sum_duplicated_coo
import numpy as np

@unittest.skip("Finished tested, disable for faster unittest")
class TestTransform(unittest.TestCase):
    def test_transform_small(self):
        N = 3
        data = np.array([1.0,2.0,3.0,4.0])
        row = np.array([0,0,1,2])
        col = np.array([0,1,2,2])
        """
          [[1, 2, 0],
           [0, 0, 3],
           [0, 0, 4]]
        """
        t = 0.75
        data2 = np.array([6])
        row2 = np.array([2])
        col2 = np.array([2])
        """
        Expected:
          [[1, 2, 0],
           [0, 0, 3],
           [0, 0, 5.5],
        """
        answer = csr_matrix((data,(row, col)), shape=(N,N))
        rmat = rigid_csr_matrix((data,(row, col)), shape=(N,N))

        answer[2,2] = 5.5
        result = rmat.transform(data2, row2, col2, t).get_csr_matrix()

        dif = np.sum(np.abs(answer - result))
        self.assertEqual(dif, 0)

    def test_transform_large(self):
        Ns = [30000, 50000, 70000, 90000]
        degs = [30, 40]
        obtains = [100, 1000, 5000]
        ts = [0.25, 0.5, 0.75]
        T = 10
        tests = generate_test(T, Ns, degs, obtains, ts)

        for j in range(T):
            N, deg, obtain, t = tests[j]
            N, deg, obtain = int(N), int(deg), int(obtain)
            row, col = np.random.randint(0, N, (2, N*deg))
            data, data2 = np.random.rand(2, N*deg)

            choose = np.random.randint(0, N*deg, (obtain))
            row2 = row[choose]
            col2 = col[choose]
            data2 = data2[choose]
            data2, row2, col2 = sum_duplicated_coo(data2, row2, col2)
            obtain = len(data2)

            mat1 = csr_matrix((data,(row, col)), shape=(N,N))
            mat2 = mat1.copy()
            for i in range(obtain):
                mat2[int(row2[i]), int(col2[i])] = data2[i]

            rmat = rigid_csr_matrix((data,(row, col)), shape=(N,N))

            answer = mat1 + (mat2-mat1)*t
            result = rmat.transform(data2, row2, col2, t).get_csr_matrix()

            dif = np.abs(answer - result)
            row, col = dif.nonzero()

            self.assertEqual(len(row), 0)

            # For debug
            # if len(row) > 0:
            #     for i in range(len(row)):
            #         print("At", row[i], ",", col[i], "Ori:", mat1[row[i], col[i]], "Ans:", answer[row[i], col[i]], "Res:", result[row[i], col[i]])


@unittest.skip("Finished tested, disable for faster unittest")
class TestMulVec(unittest.TestCase):
    def test_mul_vec_small(self):
        N = 3
        data = np.array([1.0,2.0,3.0,4.0])
        row = np.array([0,0,1,2])
        col = np.array([0,1,2,2])
        b = np.array([1,1,1])
        """
          [[1, 2, 0],  [1
           [0, 0, 3],   1
           [0, 0, 4]]   1]
        """
        """
        Expected:
          [3, 3, 4]
        """
        mat = csr_matrix((data,(row, col)), shape=(N,N))
        rmat = rigid_csr_matrix((data,(row, col)), shape=(N,N))

        answer = np.array([3, 3, 4])
        result = rmat.mul_vec(b)

        dif = np.sum(np.abs(answer - result))
        self.assertEqual(dif, 0)

    def test_mul_vec_big(self):
        Ns = [30000, 50000, 70000, 90000]
        degs = [30, 40]
        magnitudes = [1000, 100000, 1000000, 10000000]
        T = 10
        tests = generate_test(T, Ns, degs, magnitudes)

        for j in range(T):
            N, deg, magnitude = tests[j]
            N, deg = int(N), int(deg)
            row, col = np.random.randint(0, N, (2, N*deg))
            data = np.random.rand(N*deg)
            b = np.random.rand(N) * magnitude

            mat = csr_matrix((data,(row, col)), shape=(N,N))
            rmat = rigid_csr_matrix((data,(row, col)), shape=(N,N))

            answer = mat * b
            result = rmat.mul_vec(b)

            dif = np.abs(answer - result)
            nz = dif.nonzero()[0]

            self.assertEqual(len(nz), 0)

if __name__ == '__main__':
    unittest.main()
