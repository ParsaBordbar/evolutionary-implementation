"""
mobga_aos.py  –  Main MOBGA-AOS algorithm (Algorithm 1 from the paper)

Paper parameters:
  maxFEs = 300,000   fitness evaluation budget
  N      = 100       population size
  Q      = 5         crossover operators in pool
  LP     = 5         generations between OSP updates
  Pc     = 0.9       crossover rate
  Pm     = 1/D       mutation rate (one expected flip per individual)
  k-NN   k=3, 3-fold CV for fitness
"""

import numpy as np
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
                 max_fes: int         = 300_000,
                 pop_size: int        = 100,
                 crossover_rate: float = 0.9,
                 Q: int               = 5,
                 LP: int              = 5,
                 seed: int            = 42,
                 verbose: bool        = True):

        self.X_train  = X_train
        self.y_train  = y_train
        self.D        = n_features
        self.max_fes  = max_fes
        self.N        = pop_size
        self.Pc       = crossover_rate
        self.Pm       = 1.0 / n_features   # Eq. paper: Pm = 1/D
        self.verbose  = verbose

        self.rng = np.random.default_rng(seed)
        self.aos = AdaptiveOperatorSelection(Q=Q, LP=LP)

        # Fitness cache: chromosome bytes → (f1, f2)
        # Identical chromosomes are FREE — nFE only increments on real k-NN calls
        self._cache: dict[bytes, tuple[float, float]] = {}
        self.nFE = 0

    # ─────────────────────────────────────────────────────────────────────────
    # FITNESS EVALUATION  (Equations 2 and 3 from the paper)
    # ─────────────────────────────────────────────────────────────────────────

    def _evaluate(self, individual: np.ndarray) -> tuple[float, float]:
        """
        f1 = 3-fold CV classification error (%)   [Equation 2]
        f2 = number of selected features           [Equation 3]

        Cache key: individual.tobytes() — fast O(D) hash, avoids k-NN
        re-computation for chromosomes we've already seen.

        IMPORTANT: nFE only increments when we actually call k-NN.
        Cache hits are free and do NOT consume the fitness budget.
        """
        key = individual.tobytes()

        if key not in self._cache:
            f2 = float(individual.sum())

            if f2 == 0:
                # Feasibility penalty: no features → worst possible error
                # Don't call k-NN (nothing to classify with) and
                # don't count this as a fitness evaluation
                f1 = 100.0
            else:
                f1 = knn_cross_val_error(
                    self.X_train, self.y_train,
                    individual, k=3, n_folds=3
                )
                self.nFE += 1   # ← only real k-NN calls count

            self._cache[key] = (f1, f2)

        return self._cache[key]   # cache hit: free, nFE unchanged

    def _evaluate_population(self, population: np.ndarray) -> np.ndarray:
        """Batch evaluate all N individuals → objectives array (N, 2)."""
        objectives = np.zeros((len(population), 2))
        for i, ind in enumerate(population):
            objectives[i] = self._evaluate(ind)
        return objectives

    # ─────────────────────────────────────────────────────────────────────────
    # POPULATION INITIALIZATION  (Section 3.2)
    # ─────────────────────────────────────────────────────────────────────────

    def _init_population(self) -> np.ndarray:
        """
        Random binary population shape (N, D).

        Repair: guarantee every individual has at least 1 feature selected.
        All-zero chromosomes are meaningless and would create trivial
        Pareto front solutions that crowd out real ones.
        """
        pop = self.rng.integers(0, 2, size=(self.N, self.D), dtype=np.uint8)
        for i in range(self.N):
            if pop[i].sum() == 0:
                pop[i, self.rng.integers(0, self.D)] = 1
        return pop

    # ─────────────────────────────────────────────────────────────────────────
    # MAIN EVOLUTION LOOP  (Algorithm 1 from the paper)
    # ─────────────────────────────────────────────────────────────────────────

    def run(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Execute MOBGA-AOS.

        Returns
        -------
        pareto_solutions  : shape (n_pareto, D)   binary feature masks
        pareto_objectives : shape (n_pareto, 2)   [f1, f2] for each solution
        """
        # ── Step 1: Initialize population P and evaluate ──────────────────
        population = self._init_population()
        objectives = self._evaluate_population(population)
        generation = 0

        if self.verbose:
            print(f"[Init] nFE={self.nFE}  pop={self.N}  D={self.D}")

        # ── Main loop: repeat until fitness budget exhausted ──────────────
        while self.nFE < self.max_fes:

            offspring_list     = []
            offspring_obj_list = []

            # ── Step 2: Produce N offspring (N/2 crossover pairs) ─────────
            for _ in range(self.N // 2):

                # Check budget before doing work — this is the CORRECT place.
                # The break exits the for-loop; the while condition then
                # catches it and terminates the outer loop cleanly.
                if self.nFE >= self.max_fes:
                    break

                # (a) Select crossover operator via AOS roulette wheel
                #     Each operator has a probability; higher performers
                #     get selected more often (Section 3.6)
                op_idx = self.aos.select_operator(self.rng)

                # (b) Pick two random parents from current population
                p_idx = self.rng.choice(self.N, size=2, replace=False)
                p1 = population[p_idx[0]]
                p2 = population[p_idx[1]]

                # (c) Apply selected crossover operator (Pc = 0.9)
                #     then uniform mutation (Pm = 1/D)
                c1, c2 = apply_crossover(p1, p2, op_idx, self.rng, self.Pc)
                c1 = uniform_mutation(c1, self.rng, self.Pm)
                c2 = uniform_mutation(c2, self.rng, self.Pm)

                # (d) Evaluate children (cache-aware)
                c1_obj = np.array(self._evaluate(c1))
                c2_obj = np.array(self._evaluate(c2))

                # (e) Credit assignment: did this operator help or hurt?
                #     Compares children vs parents using Pareto dominance
                #     Updates nReward / nPenalty for this operator
                #     (Algorithm 2 in the paper, Section 3.5)
                parents_obj  = objectives[p_idx]              # (2, 2)
                children_obj = np.array([c1_obj, c2_obj])     # (2, 2)
                self.aos.credit_assignment(parents_obj, children_obj, op_idx)

                offspring_list.extend([c1, c2])
                offspring_obj_list.extend([c1_obj, c2_obj])

            # ── Step 3: End-of-generation bookkeeping ─────────────────────
            # Stores this generation's rewards/penalties into RD/PN matrices.
            # Every LP generations, recalculates operator probabilities.
            self.aos.end_generation()
            generation += 1

            if not offspring_list:
                break   # budget hit on very first pair of this generation

            # ── Step 4: Environmental selection (NSGA-II style) ───────────
            offspring     = np.array(offspring_list,     dtype=np.uint8)
            offspring_obj = np.array(offspring_obj_list, dtype=np.float64)

            # Combine parents P and offspring Pnew → R (size 2N)
            combined_pop = np.vstack([population, offspring])
            combined_obj = np.vstack([objectives, offspring_obj])

            # Select best N from R using:
            #   1. Non-dominated rank (front number) — prefer front 0
            #   2. Crowding distance — prefer isolated solutions
            selected_idx = environmental_selection(combined_obj, self.N)
            population   = combined_pop[selected_idx]
            objectives   = combined_obj[selected_idx]

            # ── Logging ───────────────────────────────────────────────────
            if self.verbose and generation % 20 == 0:
                fronts    = fast_non_dominated_sort(objectives)
                pf_size   = len(fronts[0])
                best_err  = objectives[fronts[0], 0].min()
                best_feat = objectives[fronts[0], 1].min()
                ops       = self.aos.probs
                print(f"[Gen {generation:5d}] nFE={self.nFE:7d}  "
                      f"PF={pf_size:3d}  "
                      f"err={best_err:.3f}%  "
                      f"feat={best_feat:.0f}  "
                      f"ops=[{', '.join(f'{p:.2f}' for p in ops)}]")

        # ── Extract final Pareto front ────────────────────────────────────
        fronts = fast_non_dominated_sort(objectives)
        pf_idx = fronts[0]
        pareto_solutions  = population[pf_idx]
        pareto_objectives = objectives[pf_idx]

        if self.verbose:
            print(f"\n[Done] nFE={self.nFE}  "
                  f"Pareto solutions: {len(pf_idx)}  "
                  f"Generations: {generation}")
            print(f"  Best train error : {pareto_objectives[:, 0].min():.3f}%")
            print(f"  Fewest features  : {pareto_objectives[:, 1].min():.0f}")

        return pareto_solutions, pareto_objectives