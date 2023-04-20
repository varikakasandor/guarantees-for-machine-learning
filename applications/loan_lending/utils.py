import numpy as np


def find_min_ya_p_ya(Y, A):
    cnt = np.zeros((2, 2))  # TODO: make it work for arbitrary protected attribute not just binary
    for y, a in zip(Y, A):
        cnt[y, a] += 1
    return np.amin(cnt) / len(Y)
