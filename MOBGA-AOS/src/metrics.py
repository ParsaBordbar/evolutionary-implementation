"""
metrics.py  –  IGD and Hypervolume metrics

These are the two metrics used in the paper to compare algorithms.

IGD (Inverted Generational Distance):
  Measures how close the found Pareto front is to the true Pareto front.
  Lower is better.
  IGD = average distance from each TRUE front point to nearest FOUND point.

HV (Hypervolume):
  Measures the volume of objective space dominated by the found Pareto front
  (relative to a reference point, usually the worst possible values).
  Higher is better.
  For 2 objectives this is the area under the Pareto front curve.

Note: Since we don't know the true Pareto front, the paper uses
the union of all algorithms' Pareto fronts as the reference "true" front.
"""

import numpy as np
from nsga_methods.nsga2 import fast_non_dominated_sort


def compute_igd(pareto_found: np.ndarray,
                pareto_true:  np.ndarray) -> float:
    """
    Inverted Generational Distance.

    For each point in the TRUE front, find the nearest point in the FOUND front.
    IGD = mean of those minimum distances.

    Parameters
    ----------
    pareto_found : shape (n, M)  — our algorithm's Pareto front
    pareto_true  : shape (m, M)  — reference "true" Pareto front

    Returns
    -------
    igd : float  (lower is better)
    """
    if len(pareto_found) == 0:
        return np.inf

    # For each true point, compute distances to all found points
    # pareto_true: (m, M), pareto_found: (n, M)
    # diff: (m, n, M)
    diff = pareto_true[:, None, :] - pareto_found[None, :, :]  # (m, n, M)
    dists = np.sqrt((diff ** 2).sum(axis=2))                   # (m, n)
    min_dists = dists.min(axis=1)                              # (m,)

    return float(min_dists.mean())


def compute_hv(pareto_front: np.ndarray,
               reference_point: np.ndarray) -> float:
    """
    Hypervolume for 2-objective minimization problems.

    The hypervolume is the area of the region that is:
      - dominated by at least one solution in pareto_front, AND
      - dominates the reference_point

    For 2D, we compute this exactly by sweeping along one axis:
      1. Sort solutions by f1 (ascending)
      2. Sweep from left to right, accumulating rectangular areas

    Parameters
    ----------
    pareto_front    : shape (n, 2)  — Pareto front solutions [f1, f2]
    reference_point : shape (2,)    — worst point (must dominate all front points)

    Returns
    -------
    hv : float  (higher is better)
    """
    if len(pareto_front) == 0:
        return 0.0

    # Filter solutions that are actually dominated by reference point
    valid = np.all(pareto_front < reference_point, axis=1)
    pf = pareto_front[valid]

    if len(pf) == 0:
        return 0.0

    # Sort by first objective (ascending)
    order = np.argsort(pf[:, 0])
    pf = pf[order]

    hv = 0.0
    prev_f1 = pf[0, 0]
    prev_f2 = reference_point[1]

    # Sweep: each solution contributes a rectangle
    # Width = difference in f1 from previous solution to current
    # Height = difference in f2 from current solution to reference
    for i in range(len(pf)):
        f1, f2 = pf[i, 0], pf[i, 1]
        width  = f1 - (prev_f1 if i == 0 else pf[i-1, 0])
        # Actually, sweep from left edge
        pass

    # Cleaner sweep implementation:
    hv = 0.0
    f2_current = reference_point[1]

    for i in range(len(pf)):
        f1 = pf[i, 0]
        f2 = pf[i, 1]

        # Width = distance to next point in f1 (or reference if last)
        next_f1 = pf[i+1, 0] if i + 1 < len(pf) else reference_point[0]
        width   = next_f1 - f1

        # Height = f2 of reference minus f2 of this solution
        height  = reference_point[1] - f2

        if width > 0 and height > 0:
            hv += width * height

    return float(hv)


def get_reference_point(objectives_list: list[np.ndarray],
                        margin: float = 1.1) -> np.ndarray:
    """
    Compute a reference point for HV as (max_f1, max_f2) × margin.

    The reference point must be dominated by (i.e., worse than) ALL solutions,
    otherwise the HV calculation is meaningless.
    """
    all_obj = np.vstack(objectives_list)
    ref = all_obj.max(axis=0) * margin
    return ref


def build_true_pareto_front(objectives_list: list[np.ndarray]) -> np.ndarray:
    """
    Build the reference "true" Pareto front by taking the union of all
    algorithms' Pareto fronts and finding the non-dominated solutions.

    This is what the paper does since the true Pareto front is unknown.
    """

    all_obj = np.vstack(objectives_list)
    fronts  = fast_non_dominated_sort(all_obj)
    return all_obj[fronts[0]]