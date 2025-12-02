import os
import numpy as np

OUTPUT_DIR = 'Figures'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Problem settings
N_JOBS = 7
PROCESSING_TIMES = np.array([12, 7, 15, 5, 9, 11, 8])
WEIGHTS = np.array([4, 9, 3, 7, 5, 6, 8])

SETUP_TIMES = np.array([
    [0,4,6,5,7,3,4],
    [4,0,5,6,4,5,6],
    [6,5,0,4,5,7,6],
    [5,6,4,0,3,4,5],
    [7,4,5,3,0,6,4],
    [3,5,7,4,6,0,5],
    [4,6,6,5,4,5,0]
])

# GA Parameters
POP_SIZE = 50
MAX_GEN = 100
ELITISM = 2
TOURNAMENT_K = 3

# Mutation and crossover options
CROSSOVER_METHODS = ["order", "pmx", "cycle"]
MUTATION_METHODS = ["swap", "inversion", "scramble"]
