import numpy as np
from numba import jit 

@jit
def qwk(a1, a2, max_rat):
    assert(len(a1) == len(a2))
    a1 = np.asarray(a1, dtype=int)
    a2 = np.asarray(a2, dtype=int)

    w = np.zeros((max_rat + 1, max_rat + 1))
    for i in range(max_rat + 1):
        for j in range(max_rat + 1):
            w[i, j] = (i - j) * (i - j)/ (max_rat * max_rat)

    hist1 = np.zeros((max_rat + 1, ))
    hist2 = np.zeros((max_rat + 1, ))

    o = 0
    for i, j in zip(a1, a2):
        hist1[i] += 1
        hist2[j] += 1
        o +=  w[i, j]

    e = 0
    for i in range(max_rat + 1):
        for j in range(max_rat + 1):
            e += hist1[i] * hist2[j] * w[i, j]

    e = e / a1.shape[0]

    return 1 - o / e
