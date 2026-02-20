"""
aos.py  –  Adaptive Operator Selection (AOS) mechanism

This is the key innovation of MOBGA-AOS over plain NSGA-II.

Instead of always using the same crossover operator, we maintain
a probability distribution over the Q=5 operators and update it
every LP=5 generations based on which operators produced
non-dominated children (reward) vs dominated children (penalty).

The intuition: if single-point crossover keeps producing children
that dominate their parents, it should be selected more often.
If uniform crossover keeps producing worse children, its probability
should drop.
"""

import numpy as np


class AdaptiveOperatorSelection:
    """
    Tracks operator performance and updates selection probabilities.

    Key attributes
    --------------
    Q       : number of operators (5 in the paper)
    LP      : how many generations between probability updates (5 in paper)
    probs   : current selection probability for each operator  shape (Q,)
    RD      : reward matrix  shape (LP, Q)  — rewards per generation per operator
    PN      : penalty matrix shape (LP, Q)  — penalties per generation per operator
    k       : current generation index within the LP window (0 to LP-1)
    """

    DELTA = 1e-4   # small constant to avoid division by zero (δ in paper)

    def __init__(self, Q: int = 5, LP: int = 5):
        self.Q  = Q
        self.LP = LP

        # Start with equal probability for all operators (1/Q each)
        self.probs = np.full(Q, 1.0 / Q)

        # Running reward/penalty counts for current generation
        self.n_reward  = np.zeros(Q, dtype=np.float64)
        self.n_penalty = np.zeros(Q, dtype=np.float64)

        # History matrices: LP rows (one per generation), Q columns (one per operator)
        self.RD = np.zeros((LP, Q), dtype=np.float64)
        self.PN = np.zeros((LP, Q), dtype=np.float64)

        self.k = 0   # current row in RD/PN

    def select_operator(self, rng: np.random.Generator) -> int:
        """
        Select an operator index using roulette wheel selection.

        Roulette wheel: imagine a wheel divided into Q sectors,
        each sector's size proportional to that operator's probability.
        Spin the wheel (pick a random number in [0,1]) and see which sector it lands in.
        """
        # np.random.choice with p= does exactly roulette wheel selection
        return int(rng.choice(self.Q, p=self.probs))

    def credit_assignment(self, parents_obj: np.ndarray,
                          children_obj: np.ndarray,
                          op_idx: int) -> None:
        """
        Update reward/penalty counts for op_idx based on child quality.

        Parameters
        ----------
        parents_obj  : shape (2, M)  — objectives of the two parents
        children_obj : shape (2, M)  — objectives of the two children
        op_idx       : which crossover operator was used

        Logic (from Algorithm 2 in the paper):
          Case 1: One parent dominates the other (p1 dominates p2)
            → compare each child against the BETTER parent (p1)
            → if child is NOT dominated by p1 → reward
            → if child IS dominated by p1     → penalty

          Case 2: Parents are non-dominated w.r.t. each other
            → compare each child against BOTH parents
            → if child is NOT dominated by either parent → reward
            → if child IS dominated by at least one parent → penalty
        """
        p1_obj, p2_obj = parents_obj[0], parents_obj[1]

        # Determine dominance relationship between parents
        p1_dom_p2 = self._dominates(p1_obj, p2_obj)
        p2_dom_p1 = self._dominates(p2_obj, p1_obj)

        if p1_dom_p2 or p2_dom_p1:
            # Case 1: one parent dominates the other
            better = p1_obj if p1_dom_p2 else p2_obj

            for child_obj in children_obj:
                if self._dominates(better, child_obj):
                    # better parent dominates child → bad crossover
                    self.n_penalty[op_idx] += 1
                else:
                    # child not dominated by better parent → good crossover
                    self.n_reward[op_idx] += 1
        else:
            # Case 2: parents are mutually non-dominated
            for child_obj in children_obj:
                p1_beats_child = self._dominates(p1_obj, child_obj)
                p2_beats_child = self._dominates(p2_obj, child_obj)

                if not p1_beats_child and not p2_beats_child:
                    # child beats or ties both parents → good
                    self.n_reward[op_idx] += 1
                else:
                    self.n_penalty[op_idx] += 1

    def end_generation(self) -> None:
        """
        Called at the end of each generation.
        Stores current reward/penalty counts into the history matrices,
        then resets the running counts.
        If LP generations have passed, update the operator probabilities.
        """
        self.RD[self.k] = self.n_reward
        self.PN[self.k] = self.n_penalty

        # Reset for next generation
        self.n_reward  = np.zeros(self.Q, dtype=np.float64)
        self.n_penalty = np.zeros(self.Q, dtype=np.float64)

        self.k += 1

        if self.k == self.LP:
            self._update_probabilities()
            self.k = 0

    def _update_probabilities(self) -> None:
        """
        Recalculate selection probabilities from the LP-generation history.

        From equations (6)–(10) in the paper:

          S1_q = sum of rewards   for operator q over last LP gens
          S2_q = sum of penalties for operator q over last LP gens

          S3_q = δ  if S1_q == 0  (prevent zero division)
                 S1_q otherwise

          S4_q = S1_q / (S3_q + S2_q)   ← "success rate"

          p_q  = S4_q / sum(S4)          ← normalize to sum=1
        """
        S1 = self.RD.sum(axis=0)    # shape (Q,) — total rewards per operator
        S2 = self.PN.sum(axis=0)    # shape (Q,) — total penalties per operator

        S3 = np.where(S1 == 0, self.DELTA, S1)
        S4 = S1 / (S3 + S2)

        # Normalize to get probabilities
        total = S4.sum()
        if total == 0:
            self.probs = np.full(self.Q, 1.0 / self.Q)
        else:
            self.probs = S4 / total

        # Reset history matrices for next LP window
        self.RD = np.zeros((self.LP, self.Q), dtype=np.float64)
        self.PN = np.zeros((self.LP, self.Q), dtype=np.float64)

    @staticmethod
    def _dominates(a: np.ndarray, b: np.ndarray) -> bool:
        """A dominates B: A <= B on all objectives, A < B on at least one."""
        return bool(np.all(a <= b) and np.any(a < b))