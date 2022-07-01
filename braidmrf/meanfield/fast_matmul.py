import numpy as np
from numba import jit, njit, prange
from numba import float32
from time import time as timer

@njit(parallel=True)
def matmul(A, B, C):
    nCols = len(C)
    for j in prange(nCols):
        tmp = B[:,j]
        C[j] = np.dot(A, tmp)

def test_matmul(A,B):
    nRows = A.shape[0]
    nK = B.shape[1]
    C = np.zeros(nK, dtype=float)
    ts = timer()
    matmul(A, B, C)
    te = timer()
    print("matmul GPU time:", te - ts)

    ts = timer()
    ans = np.matmul(A, B)
    te = timer()
    print("NumPy time:", te - ts)
    np.testing.assert_allclose(ans, C, rtol=1e-5)
    return ans