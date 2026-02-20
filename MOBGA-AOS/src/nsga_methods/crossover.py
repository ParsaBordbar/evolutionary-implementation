"""
crossover.py  –  Five binary crossover operators for MOBGA-AOS

All operators take two parent arrays (np.uint8) and return two children.
Parents are NEVER modified in place — we always work on copies.

The five operators have different search characteristics:
  1. Single-point   – swaps one segment (low disruption)
  2. Two-point      – swaps middle segment (medium disruption)
  3. Uniform        – independently swaps each gene (high disruption)
  4. Shuffle        – shuffles genes before single-point (removes positional bias)
  5. Reduced surrogate – only crosses at positions where parents differ (efficient)
"""

import numpy as np


# ── 1. Single-point crossover ────────────────────────────────────────────────
def single_point(p1: np.ndarray, p2: np.ndarray,
                 rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """
    Pick one random cut point k. Child 1 = p1[:k] + p2[k:], Child 2 = p2[:k] + p1[k:]

    Example (L=8, k=3):
      p1 = [1,0,1 | 0,0,1,1,0]
      p2 = [0,1,0 | 1,1,0,0,1]
      c1 = [1,0,1 | 1,1,0,0,1]
      c2 = [0,1,0 | 0,0,1,1,0]
    """
    L = len(p1)
    k = rng.integers(1, L)          # cut point in [1, L-1]

    c1 = np.empty(L, dtype=np.uint8)
    c2 = np.empty(L, dtype=np.uint8)

    c1[:k] = p1[:k];  c1[k:] = p2[k:]
    c2[:k] = p2[:k];  c2[k:] = p1[k:]

    return c1, c2


# ── 2. Two-point crossover ───────────────────────────────────────────────────
def two_point(p1: np.ndarray, p2: np.ndarray,
              rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """
    Pick two cut points k1 < k2. Swap the MIDDLE segment between them.

    Example (L=8, k1=2, k2=5):
      p1 = [1,0 | 1,0,0 | 1,1,0]
      p2 = [0,1 | 0,1,1 | 0,0,1]
      c1 = [1,0 | 0,1,1 | 1,1,0]   ← middle from p2
      c2 = [0,1 | 1,0,0 | 0,0,1]   ← middle from p1
    """
    L = len(p1)
    # Pick two distinct points and sort them
    pts = rng.choice(L - 1, size=2, replace=False) + 1
    k1, k2 = int(pts.min()), int(pts.max())

    c1 = p1.copy()
    c2 = p2.copy()
    c1[k1:k2] = p2[k1:k2]
    c2[k1:k2] = p1[k1:k2]

    return c1, c2


# ── 3. Uniform crossover ─────────────────────────────────────────────────────
def uniform(p1: np.ndarray, p2: np.ndarray,
            rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """
    For each gene position, flip a coin. If heads, swap that gene between parents.

    This gives the highest disruption — any combination of parent genes is reachable.
    Good for escaping local optima but can disrupt good building blocks.

    Example (L=6, mask=[1,0,1,0,0,1]):
      p1 = [1,0,1,0,0,1]
      p2 = [0,1,0,1,1,0]
      c1 = [0,0,0,0,0,0]   ← swapped at positions 0,2,5
      c2 = [1,1,1,1,1,1]
    """
    L = len(p1)
    swap_mask = rng.integers(0, 2, size=L, dtype=np.uint8)   # random 0/1 for each gene

    c1 = np.where(swap_mask, p2, p1).astype(np.uint8)
    c2 = np.where(swap_mask, p1, p2).astype(np.uint8)

    return c1, c2


# ── 4. Shuffle crossover ─────────────────────────────────────────────────────
def shuffle(p1: np.ndarray, p2: np.ndarray,
            rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """
    Shuffle gene positions, apply single-point crossover, then unshuffle.

    Why? Single-point crossover has a positional bias — genes at the ends
    are less likely to be separated than genes in the middle. Shuffling
    first removes that bias, making all gene pairs equally likely to cross.

    Steps:
      1. Create a random permutation of gene indices
      2. Shuffle both parents using that permutation
      3. Apply single-point crossover on the shuffled versions
      4. Unshuffle children back to original order
    """
    L = len(p1)
    perm    = rng.permutation(L)          # random shuffle order
    inv_perm = np.argsort(perm)           # inverse permutation for unshuffling

    # Shuffle
    p1s = p1[perm]
    p2s = p2[perm]

    # Single-point on shuffled
    k = rng.integers(1, L)
    c1s = np.empty(L, dtype=np.uint8)
    c2s = np.empty(L, dtype=np.uint8)
    c1s[:k] = p1s[:k];  c1s[k:] = p2s[k:]
    c2s[:k] = p2s[:k];  c2s[k:] = p1s[k:]

    # Unshuffle
    c1 = c1s[inv_perm]
    c2 = c2s[inv_perm]

    return c1, c2


# ── 5. Reduced surrogate crossover ──────────────────────────────────────────
def reduced_surrogate(p1: np.ndarray, p2: np.ndarray,
                      rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """
    Only cross at positions where the two parents DIFFER.

    Why? If parents have the same gene at a position, crossing there produces
    identical children — it's wasted effort. By restricting crossover to
    positions of difference, every crossover is guaranteed to produce
    at least one child that differs from both parents.

    Steps:
      1. Find all positions where p1[i] != p2[i]
      2. If none differ → return copies (nothing to cross)
      3. Pick one of those positions as the cut point
      4. Apply single-point crossover
    """
    diff_positions = np.where(p1 != p2)[0]

    # If parents are identical, crossover has no effect
    if len(diff_positions) == 0:
        return p1.copy(), p2.copy()

    # Pick a random cut point from the differing positions
    k = int(rng.choice(diff_positions))

    c1 = p1.copy()
    c2 = p2.copy()
    c1[k:] = p2[k:]
    c2[k:] = p1[k:]

    return c1, c2


# ── Operator pool (indexed 0–4, matching the paper's Q=5) ───────────────────
CROSSOVER_OPS = [
    single_point,
    two_point,
    uniform,
    shuffle,
    reduced_surrogate,
]
OPERATOR_NAMES = [
    "SinglePoint",
    "TwoPoint",
    "Uniform",
    "Shuffle",
    "ReducedSurrogate",
]


def apply_crossover(p1: np.ndarray, p2: np.ndarray,
                    op_idx: int,
                    rng: np.random.Generator,
                    crossover_rate: float = 0.9) -> tuple[np.ndarray, np.ndarray]:
    """
    Apply the chosen crossover operator with probability crossover_rate.
    If crossover doesn't happen, children are copies of parents.
    """
    if rng.random() < crossover_rate:
        return CROSSOVER_OPS[op_idx](p1, p2, rng)
    else:
        return p1.copy(), p2.copy()