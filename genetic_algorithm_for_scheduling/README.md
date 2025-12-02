# Genetic Algorithm for Scheduling Using Permutation Representation problem 

##  Problem representation & objective

Instance: single-machine scheduling with sequence-dependent setup times.
	•	Jobs: \{0,1,\dots,n-1\} (I use 0-based indices in code)
	•	Processing times: p_i
	•	Weights: w_i
	•	Setup times: s_{i,j} — time to switch from job i to job j

Schedule representation: a permutation X = [x_1, x_2, \dots, x_n] where x_k is the job index scheduled at position k.

Completion times:
	•	C_{x_1} = p_{x_1}
	•	C_{x_k} = C_{x_{k-1}} + s_{x_{k-1},x_k} + p_{x_k}

Objective (minimize):
T(X) = \sum_{k=1}^{n} w_{x_k} \, C_{x_k}

GA fitness (we maximize):
\text{fitness}(X) = \frac{1}{T(X)}

Important: The chromosome must be an integer permutation (values 0..n-1). Indexing arrays (processing times, weights, setup times) requires integers — floats or placeholders will break indexing.


## GA high-level workflow (step-by-step)
	1.	Initialization: Create POP_SIZE random permutations with np.random.permutation(n) (integer dtype).
	2.	Evaluation: Compute fitness for each individual via 1 / T(X).
	3.	Selection: Build a mating pool using tournament selection (k=3). Repeat until pool size equals population.
	4.	Crossover: Pair parents and apply chosen permutation-preserving crossover to produce children.
	5.	Mutation: For each child apply the chosen mutation operator.
	6.	Elitism: Copy the top ELITISM individuals (by fitness) unchanged to the next generation.
	7.	Replacement: Form the new population with elites + children (ensure all are valid integer permutations).
	8.	Repeat until MAX_GEN.
	9.	Output: best permutation found, its T(X), generation count, and fitness evolution (max and average per generation).


## Responsibility summary
	•	config.py: central place for all parameters (easy to change).
	•	problem.py: maps a permutation → completion times → objective T(X) → fitness 1/T(X).
	•	operators_*: each file implements a family of operators, must guarantee valid integer permutations on output.
	•	ga.py: orchestrates selection → crossover → mutation → elitism; collects fitness statistics.
	•	main.py: runs all crossover/mutation combinations and produces plots saved to Figures/.


## Structure Of The Code
```
> ctx3 print
┌── 📂 Project structure:
└── genetic_algorithm_for_scheduling (544 bytes)
    ├── .gitignore (31 bytes)
    ├── Figures (352 bytes)
    │   ├── cycle_inversion.png (147175 bytes)
    │   ├── cycle_scramble.png (140544 bytes)
    │   ├── cycle_swap.png (138532 bytes)
    │   ├── order_inversion.png (149422 bytes)
    │   ├── order_scramble.png (143759 bytes)
    │   ├── order_swap.png (148261 bytes)
    │   ├── pmx_inversion.png (136934 bytes)
    │   ├── pmx_scramble.png (134139 bytes)
    │   └── pmx_swap.png (145274 bytes)
    ├── README.md (592 bytes)
    ├── config.py (601 bytes)
    ├── ga.py (1614 bytes)
    ├── main.py (829 bytes)
    ├── operators_crossover.py (1216 bytes)
    ├── operators_mutation.py (510 bytes)
    ├── operators_selection.py (400 bytes)
    ├── problem.py (556 bytes)
    ├── pyproject.toml (178 bytes)
    └── uv.lock (152 bytes)
```