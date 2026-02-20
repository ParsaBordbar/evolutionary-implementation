"""
nsga2.py  –  NSGA-II selection mechanics

Two key components:
  1. Fast non-dominated sort   → assigns a rank (front) to each individual
  2. Crowding distance         → measures how isolated a solution is on its front

Together they implement NSGA-II's environmental selection:
  - Prefer lower rank (closer to true Pareto front)
  - Among equal rank, prefer higher crowding distance (more diverse)

Both objectives are MINIMIZED (error rate and feature count).
"""

import numpy as np


# ── Pareto dominance ─────────────────────────────────────────────────────────

def dominates(f_a: np.ndarray, f_b: np.ndarray) -> bool:
    """
    Return True if solution A dominates solution B.

    A dominates B when:
      - A is no worse than B on ALL objectives, AND
      - A is strictly better than B on AT LEAST ONE objective

    Since both objectives are minimized, "better" means "smaller".
    """
    return bool(np.all(f_a <= f_b) and np.any(f_a < f_b))


# ── Fast non-dominated sort ──────────────────────────────────────────────────

def fast_non_dominated_sort(objectives: np.ndarray) -> list[list[int]]:
    """
    Classic NSGA-II fast non-dominated sort (Deb et al. 2002).

    Parameters
    ----------
    objectives : shape (N, M)  — N solutions, M objectives (all minimized)

    Returns
    -------
    fronts : list of lists, where fronts[0] = Pareto front (best),
             fronts[1] = second front, etc.
             Each inner list contains indices into the objectives array.

    How it works:
      For each solution p, track:
        - n_p : how many solutions dominate p  (domination count)
        - S_p : list of solutions that p dominates

      Front 1 = all solutions with n_p == 0 (nobody beats them).
      Then for each solution q in S_p (things p dominates),
      decrement n_q. When n_q hits 0, q joins the next front.
    """
    N = len(objectives)

    # n[i] = number of solutions that dominate solution i
    n = np.zeros(N, dtype=np.int32)
    # S[i] = list of solution indices that solution i dominates
    S = [[] for _ in range(N)]

    fronts = [[]]

    for i in range(N):
        for j in range(i + 1, N):
            if dominates(objectives[i], objectives[j]):
                S[i].append(j)
                n[j] += 1
            elif dominates(objectives[j], objectives[i]):
                S[j].append(i)
                n[i] += 1

        if n[i] == 0:
            fronts[0].append(i)

    current_front = 0
    while fronts[current_front]:
        next_front = []
        for i in fronts[current_front]:
            for j in S[i]:
                n[j] -= 1
                if n[j] == 0:
                    next_front.append(j)
        current_front += 1
        fronts.append(next_front)

    # Remove the last empty front
    return [f for f in fronts if f]


# ── Crowding distance ────────────────────────────────────────────────────────

def crowding_distance(objectives: np.ndarray, front: list[int]) -> np.ndarray:
    """
    Compute crowding distance for solutions in one front.

    Crowding distance measures how much "space" surrounds each solution.
    Boundary solutions (best/worst on any objective) get infinite distance,
    ensuring they're always preserved. Interior solutions get distance
    proportional to the gap between their neighbours.

    Parameters
    ----------
    objectives : shape (N, M)  — ALL solutions' objectives
    front      : list of indices belonging to this front

    Returns
    -------
    distances : shape (len(front),)  — crowding distance for each solution in front
    """
    n = len(front)
    if n <= 2:
        return np.full(n, np.inf)

    front_arr = np.array(front)
    obj = objectives[front_arr]   # shape (n, M)
    M   = obj.shape[1]

    distances = np.zeros(n)

    for m in range(M):
        # Sort solutions by this objective
        order = np.argsort(obj[:, m])
        sorted_obj = obj[order, m]

        # Boundary solutions always kept
        distances[order[0]]  = np.inf
        distances[order[-1]] = np.inf

        # Normalise by objective range (avoid division by zero)
        obj_range = sorted_obj[-1] - sorted_obj[0]
        if obj_range == 0:
            continue

        # Interior solutions: add normalised gap to neighbours
        for i in range(1, n - 1):
            distances[order[i]] += (sorted_obj[i + 1] - sorted_obj[i - 1]) / obj_range

    return distances


# ── Environmental selection ──────────────────────────────────────────────────

def environmental_selection(objectives: np.ndarray, N: int) -> np.ndarray:
    """
    Select N solutions from a combined parent+offspring pool (size 2N).

    Selection rule (NSGA-II):
      1. Fill slots with complete fronts (front 0, then 1, etc.)
      2. When the next front doesn't fit entirely:
         sort it by crowding distance (descending) and take the best fitting

    Parameters
    ----------
    objectives : shape (2N, M)  — objectives for all 2N solutions
    N          : target population size

    Returns
    -------
    selected_idx : shape (N,)  — indices of selected solutions
    """
    fronts = fast_non_dominated_sort(objectives)
    selected = []

    for front in fronts:
        if len(selected) + len(front) <= N:
            # Entire front fits
            selected.extend(front)
        else:
            # Partial front — pick by crowding distance
            needed = N - len(selected)
            dist   = crowding_distance(objectives, front)
            # Sort front members by crowding distance (highest = most isolated = preferred)
            order  = np.argsort(-dist)   # descending
            selected.extend([front[i] for i in order[:needed]])
            break

    return np.array(selected, dtype=np.int32)