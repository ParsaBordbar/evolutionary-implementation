"""
mobga_aos.py  –  Main MOBGA-AOS algorithm (Algorithm 1 from the paper)

This ties everything together:
  data → population → evolution loop → Pareto front output

Parameters (from paper):
  maxFEs = 300,000   (fitness evaluations budget)
  N      = 100       (population size)
  Q      = 5         (crossover operators)
  LP     = 5         (generations between OSP updates)
  Pc     = 0.9       (crossover rate)
  Pm     = 1/D       (mutation rate)
  k-NN   k=3, 3-fold CV
"""

import numpy as np
from functools import lru_cache
from knn import knn_cross_val_error
from nsga_methods.crossover import apply_crossover
from nsga_methods.mutation import uniform_mutation
from nsga_methods.nsga2 import fast_non_dominated_sort, environmental_selection
from aos import AdaptiveOperatorSelection


class MOBGA_AOS:
    def __init__(self,
                 X_train: np.ndarray,
                 y_train: np.ndarray,
                 n_features: int,
                 max_fes: int    = 300_000,
                 pop_size: int   = 100,
                 crossover_rate: float = 0.9,
                 Q: int          = 5,
                 LP: int         = 5,
                 seed: int       = 42,
                 verbose: bool   = True):

        self.X_train = X_train
        self.y_train = y_train
        self.D       = n_features
        self.max_fes = max_fes
        self.N       = pop_size
        self.Pc      = crossover_rate
        self.Pm      = 1.0 / n_features
        self.verbose = verbose

        self.rng = np.random.default_rng(seed)
        self.aos = AdaptiveOperatorSelection(Q=Q, LP=LP)

        # Fitness cache: tuple(chromosome) → (f1, f2)
        # Avoids recomputing k-NN for identical chromosomes
        self._cache: dict[tuple, tuple[float, float]] = {}

        self.nFE = 0   # fitness evaluation counter

    # ── Fitness evaluation (with caching) ───────────────────────────────────

    def _evaluate(self, individual: np.ndarray) -> tuple[float, float]:
        """
        Compute both objectives for one individual.

        f1 = 3-fold CV classification error (%) using k-NN(k=3)
        f2 = number of selected features

        The cache uses the chromosome as a key (converted to tuple).
        This is crucial for performance: many individuals in the
        population may share the same chromosome over generations.
        """
        key = individual.tobytes()

        if key not in self._cache:
            f2 = float(individual.sum())

            if f2 == 0:
                # Penalize empty masks — assign worst possible error.
                # This stops zero-feature chromosomes from being "free"
                # low-f2 solutions that dominate everything on that axis.
                f1 = 100.0
            else:
                f1 = knn_cross_val_error(self.X_train, self.y_train,
                                         individual, k=3, n_folds=3)
                self.nFE += 1

            self._cache[key] = (f1, f2)

        return self._cache[key]

    def _evaluate_population(self, population: np.ndarray) -> np.ndarray:
        """Evaluate all N individuals, return objectives array shape (N, 2)."""
        objectives = np.zeros((len(population), 2))
        for i, ind in enumerate(population):
            objectives[i] = self._evaluate(ind)
        return objectives

    # ── Population initialization ────────────────────────────────────────────

    def _init_population(self) -> np.ndarray:
        """
        Random binary population of shape (N, D).
        Each gene is independently 0 or 1 with equal probability.

        We guarantee every individual has at least 1 feature selected —
        an all-zero chromosome creates a trivial solution that floods
        the Pareto front with meaningless zero-feature entries.
        """
        pop = self.rng.integers(0, 2, size=(self.N, self.D), dtype=np.uint8)
        for i in range(self.N):
            if pop[i].sum() == 0:
                pop[i, self.rng.integers(0, self.D)] = 1
        return pop

    # ── Main evolution loop ─────────────────────────────────────────────────

    def run(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Execute MOBGA-AOS (Algorithm 1 from the paper).

        Returns
        -------
        pareto_solutions  : binary array shape (n_pareto, D)
        pareto_objectives : float array  shape (n_pareto, 2) — [f1, f2]
        """
        # ── Initialization ──
        population  = self._init_population()
        objectives  = self._evaluate_population(population)
        generation  = 0

        if self.verbose:
            print(f"[Init] nFE={self.nFE}  pop={self.N}  D={self.D}")

        # ── Evolution ──
        while self.nFE < self.max_fes:
            offspring_list     = []
            offspring_obj_list = []

            # Produce N offspring (N/2 pairs)
            for _ in range(self.N // 2):
                # 1. Select crossover operator via roulette wheel
                op_idx = self.aos.select_operator(self.rng)

                # 2. Randomly select two parents from current population
                p_idx = self.rng.choice(self.N, size=2, replace=False)
                p1, p2 = population[p_idx[0]], population[p_idx[1]]

                # 3. Crossover then mutation
                c1, c2 = apply_crossover(p1, p2, op_idx, self.rng, self.Pc)
                c1 = uniform_mutation(c1, self.rng, self.Pm)
                c2 = uniform_mutation(c2, self.rng, self.Pm)

                # 4. Evaluate children (uses cache if already seen)
                if self.nFE >= self.max_fes:
                    break

                c1_obj = np.array(self._evaluate(c1))
                c2_obj = np.array(self._evaluate(c2))

                # 5. Credit assignment: reward/penalize the used operator
                parents_obj  = objectives[p_idx]          # shape (2, 2)
                children_obj = np.array([c1_obj, c2_obj]) # shape (2, 2)
                self.aos.credit_assignment(parents_obj, children_obj, op_idx)

                offspring_list.extend([c1, c2])
                offspring_obj_list.extend([c1_obj, c2_obj])

            # ── End of generation ──
            self.aos.end_generation()
            generation += 1

            if not offspring_list:
                break

            # 6. Combine parent and offspring populations
            offspring     = np.array(offspring_list,     dtype=np.uint8)
            offspring_obj = np.array(offspring_obj_list, dtype=np.float64)

            combined_pop = np.vstack([population, offspring])
            combined_obj = np.vstack([objectives, offspring_obj])

            # 7. Environmental selection: pick best N from combined pool
            selected_idx = environmental_selection(combined_obj, self.N)
            population   = combined_pop[selected_idx]
            objectives   = combined_obj[selected_idx]

            if self.verbose and generation % 20 == 0:
                fronts   = fast_non_dominated_sort(objectives)
                pf_size  = len(fronts[0])
                best_err = objectives[fronts[0], 0].min()
                best_feat= objectives[fronts[0], 1].min()
                ops      = self.aos.probs
                print(f"[Gen {generation:4d}] nFE={self.nFE:6d}  "
                      f"PF_size={pf_size:3d}  "
                      f"best_err={best_err:.3f}%  "
                      f"min_feat={best_feat:.0f}  "
                      f"ops=[{', '.join(f'{p:.2f}' for p in ops)}]")

        # ── Extract final Pareto front ──
        fronts = fast_non_dominated_sort(objectives)
        pf_idx = fronts[0]
        pareto_solutions  = population[pf_idx]
        pareto_objectives = objectives[pf_idx]

        if self.verbose:
            print(f"\n[Done] nFE={self.nFE}  Pareto solutions: {len(pf_idx)}")
            print(f"  Best error    : {pareto_objectives[:, 0].min():.3f}%")
            print(f"  Fewest features: {pareto_objectives[:, 1].min():.0f}")

        return pareto_solutions, pareto_objectives