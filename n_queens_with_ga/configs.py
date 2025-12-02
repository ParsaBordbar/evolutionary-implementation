class Config:
    population_size: int = 100
    parent_selection_count: int = 5
    ga_pipeline_rounds: int = 1000          # rounds of running of the algorithm
    max_evaluations: int = 10000
    n_queens: int = 8
    mutation_probability: float = 0.5       # values from the HW: 0.2, 0.5 & 1
    crossover_probability: float = 1.0      # values from the HW: 0.5 & 1 
    mutation_type: str = "swap"          # values from the HW: swap & bitwise
    random_seed: int = 42                          # random seed for reproducibility


cfg = Config()