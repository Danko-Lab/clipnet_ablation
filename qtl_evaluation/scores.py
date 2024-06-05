import numpy as np


def l2_score(x, y, pseudocount=1e-5):
    return np.sqrt(np.sum(np.square(x - y), axis=1))


def sum_log_score(x, y, pseudocount=1e-5):
    return np.sum(np.log10(x + pseudocount) - np.log10(y + pseudocount), axis=1)


def sum_score(x, y):
    return np.sum(x - y, axis=1)
