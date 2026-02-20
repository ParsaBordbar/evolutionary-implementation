"""
mutation.py  –  Uniform bit-flip mutation

The paper uses uniform mutation with rate Pm = 1/D where D is
the number of features. This means on average ONE gene flips per
individual — enough to maintain diversity without destroying good solutions.

Why 1/D?  It's a classic heuristic: if you flip too many bits
the child looks random; too few and you barely explore. 1/D gives
exactly one expected flip regardless of chromosome length.
"""

import numpy as np


def uniform_mutation(individual: np.ndarray,
                     rng: np.random.Generator,
                     mutation_rate: float) -> np.ndarray:
    """
    Flip each bit independently with probability mutation_rate.

    Parameters
    ----------
    individual    : binary array of shape (D,)
    rng           : numpy random generator
    mutation_rate : probability of flipping each bit (paper uses 1/D)

    Returns
    -------
    mutated : new binary array (original is NOT modified)
    """
    # Generate a random float for each gene
    flip_mask = rng.random(len(individual)) < mutation_rate

    # XOR with 1 flips 0→1 and 1→0
    mutated = individual.copy()
    mutated[flip_mask] ^= 1

    # Repair: if mutation wiped all features, force one back on.
    # This is a common technique called "feasibility repair" —
    # rather than rejecting the solution, we nudge it to be valid.
    if mutated.sum() == 0:
        mutated[rng.integers(0, len(mutated))] = 1

    return mutated