import numpy as np
from config import PROCESSING_TIMES, WEIGHTS, SETUP_TIMES

def compute_completion_times(order):
    C = np.zeros(len(order))
    C[0] = PROCESSING_TIMES[order[0]]

    for k in range(1, len(order)):
        prev, curr = order[k-1], order[k]
        C[k] = C[k-1] + SETUP_TIMES[prev, curr] + PROCESSING_TIMES[curr]

    return C


def total_weighted_completion_time(order):
    C = compute_completion_times(order)
    return np.sum(WEIGHTS[order] * C)


def fitness(order):
    T = total_weighted_completion_time(order)
    return 1.0 / T