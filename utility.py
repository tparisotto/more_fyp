import numpy as np


def int_to_1hot(n, dim):
    vec = np.zeros(dim)
    vec[n] = 1
    return vec
