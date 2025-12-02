<img width="400" height="400" alt="8queens" src="https://github.com/user-attachments/assets/bb062757-1e77-4bde-a417-d018de3b22ca" />


# Genetic Algorithm Solution to the N-Queens Problem

A sophisticated implementation of a genetic algorithm designed to solve the classic N-Queens problem. This project evolves a population of candidate solutions through selection, crossover, mutation, and survival mechanisms to find valid board configurations where no two queens can attack each other.

## Problem Overview

The N-Queens problem asks: **"How can N queens be placed on an N×N chessboard such that no two queens threaten each other?"**

Two queens attack each other if they share the same row, column, or diagonal. For the standard 8-Queens variant, finding just one valid solution by brute force would require checking millions of combinations. A genetic algorithm efficiently explores the solution space by iteratively improving candidate solutions.

### Solution Representation

Each solution (chromosome) is represented as a permutation array of length N:
- **Position in array** = column number
- **Value at position** = row number where the queen is placed

Example for N=8:
```
[4, 2, 7, 3, 6, 8, 5, 1]
    ↓  ↓  ↓  ↓  ↓  ↓  ↓  ↓
Column: 1  2  3  4  5  6  7  8
```

This automatically satisfies the column constraint (each value appears exactly once). The algorithm only needs to avoid row and diagonal conflicts.

## Genetic Algorithm Pipeline

The GA follows a classic evolutionary approach:

1. **Initialization** – Generate a random population of valid permutations
2. **Fitness Evaluation** – Score each solution by counting conflicts (row and diagonal)
3. **Parent Selection** – Choose the best-fit individuals for reproduction
4. **Crossover** – Combine parent solutions using multiple crossover strategies
5. **Mutation** – Introduce variation through gene swaps or bitwise changes
6. **Survival Selection** – Keep the fittest individuals for the next generation
7. **Convergence** – Repeat until optimal solution found or max generations reached

## Features

### Multiple Crossover Operators

- **Cut-and-Fill (CutFill)** – Combines parent segments and fills gaps intelligently
- **Partially Mapped Crossover (PMX)** – Preserves parent structure through mapping
- **Multi-Cut Crossover** – Creates multiple cut points for enhanced exploration

### Flexible Mutation Strategies

- **Swap Mutation** – Exchanges two randomly selected genes for permutation preservation
- **Bitwise Mutation** – Randomly assigns new values with automatic repair mechanism

### Advanced Selection Methods

- **Tournament Selection** – Selects best individuals from random samples
- **Elitism** – Preserves top solutions across generations

## Installation

Requires Python 3.13+ and dependencies listed in `pyproject.toml`.

### Using UV (Recommended)

```bash
uv sync
```

### Using Pip

```bash
pip install -e .
```

## Configuration

Edit `configs.py` to customize algorithm parameters:

```python
class Config:
    population_size: int = 100                    # Solutions per generation
    parent_selection_count: int = 5               # Parents for reproduction
    ga_pipeline_rounds: int = 1000                # Max generations to evolve
    n_queens: int = 8                             # Board dimension
    mutation_probability: float = 0.5             # [0.2, 0.5, 1.0]
    crossover_probability: float = 1.0            # [0.5, 1.0]
    mutation_type: str = "swap"                   # "swap" or "bitwise"
    random_seed: int = 42                         # Reproducibility
```

## Usage

### Run Basic Algorithm

```bash
uv run main.py
```

Or with Python:

```bash
python3 main.py
```

### Run Comprehensive Experiments

Uncomment the experiment function in `main.py` to test all parameter combinations:

```python
def main():
    set_global_seed()
    run_experiments()  # Tests all combinations
    analyze_and_plot_ga_results("ga_experiment_results.csv")
```

This tests combinations of:
- Mutation types: swap, bitwise
- Mutation probabilities: 0.2, 0.5, 1.0
- Crossover probabilities: 0.5, 1.0
- Crossover modes: cutfill, pmx, multi-cut
- Multi-cut variants: 1, 2, 3 cuts
- Elitism: enabled, disabled

Results are saved to `ga_experiment_results.csv` and visualized as plots in the `plots/` directory.

## Code Structure

**ga.py** – Core GA implementation
- `generate_population()` – Creates initial random solutions
- `fitness_evaluation_vectorized()` – Scores solutions efficiently using NumPy
- `parent_selection()` – Tournament-style parent selection
- `crossover()` – Three crossover implementations
- `mutation()` – Swap and bitwise mutation operators
- `survival_selection()` – Selects fittest for next generation
- `simple_GA_pipeline()` – Main evolutionary loop

**utils.py** – Helper functions
- `generate_chromosome()` – Creates random valid permutation
- `repair_child()` – Fixes invalid solutions from bitwise mutation
- `select_a_random_chromosome()` – Random index selection
- `log_generation()` – Records generation statistics

**plot.py** – Analysis and visualization
- `analyze_and_plot_ga_results()` – Generates comparative performance plots

**configs.py** – Centralized configuration management

## Output

### Console Output

```
--- Run: mut=0.5, cross=1.0, mode=pmx, cuts=1, elitism=False ---
----- GA Summary -----
Generation: 42
Mean Fitness: 0.956
Best Fitness: 1.0
✅ Found solution in 42 generations
```

### Generated Plots

- **speed_by_mutation_mode.png** – Convergence speed across parameters
- **elitism_effect.png** – Impact of elitism on performance
- **distribution_by_mode.png** – Variance in convergence time by crossover type
- **multi_cut_performance.png** – Multi-cut variant comparison

### CSV Results

`ga_experiment_results.csv` contains:
- mutation_prob, mutationType, crossover_prob, mode, cuts, elitism
- final_best_fitness, generations (time to solution)

## Performance Insights

Based on comprehensive experiments:

- **Success Rate:** ~97% across all configurations
- **Typical Convergence:** 0–100 generations
- **Best Performer:** PMX with swap mutation at 0.5–1.0 mutation rate
- **Fastest Mode:** Multi-cut with 1 cut often converges instantly
- **Most Reliable:** Swap mutation over bitwise

## Algorithm Details

### Fitness Calculation

```
penalty = conflicts_same_row + conflicts_on_diagonal
fitness = 1 / (1 + penalty)
```

A perfect solution has fitness = 1.0 (zero conflicts). The algorithm terminates when this is achieved.

### Conflict Detection

For each pair of queens:
- Same row: `q[i] == q[j]`
- Same diagonal: `|q[i] - q[j]| == |i - j|`

### Vectorized Operations

All fitness evaluations use NumPy vectorization for performance:

```python
i, j = np.triu_indices(n, k=1)  # Upper triangle pairs
same_row = q[i] == q[j]
same_diag = np.abs(q[i] - q[j]) == np.abs(i - j)
penalty = np.sum(same_row | same_diag)
```

## Dependencies

- **numpy** – Numerical computing and vectorization
- **pandas** – Data analysis and CSV export
- **matplotlib** – Visualization and plotting
- **python-dotenv** – Environment configuration


## License

This project is provided as-is for educational purposes.

## Author

Evolutionary Algorithms Lab – Genetic Algorithm Research and Development
