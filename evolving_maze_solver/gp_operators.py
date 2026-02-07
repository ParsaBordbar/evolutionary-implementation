import random
from agent import Agent
from tree_nodes import generate_tree


def evaluate(ind, maze, start, goal, max_steps, return_agent=False):
    """
    Evaluate fitness of an individual (GP tree).
    
    FITNESS FORMULA (from textbook principles):
    F = s + 2*d + 10*w + 5*l
    
    Where:
    - s = steps taken (1 to max_steps)
    - d = Manhattan distance to goal (0 = reached goal)
    - w = wall hits (collisions with walls/boundaries)
    - l = revisits (how many times agent revisited same cells)
    
    LOWER FITNESS IS BETTER (minimization problem)
    
    OPTIMAL FITNESS = 0 when:
    - Reached goal (d = 0)
    - No wall hits (w = 0)
    - No revisits (l = 0)
    - Minimal steps (s = Manhattan distance from start to goal)
    
    This formula encourages:
    1. Reaching the goal (2*d term makes distance very important)
    2. Avoiding walls (10*w term heavily penalizes collisions)
    3. No backtracking (5*l term penalizes revisits)
    4. Efficiency (s term rewards shorter paths)
    """
    # Create new agent at start position
    agent = Agent(start)

    for _ in range(max_steps):
        ind.tree.execute(agent, maze)
        
        if (agent.x, agent.y) == goal:
            break

    #fitness
    s = agent.steps
    d = abs(agent.x - goal[0]) + abs(agent.y - goal[1])  # Manhattan distance
    w = agent.wall_hits
    l = agent.revisit_count
    
    # Compute total fitness
    ind.fitness = s + 2*d + 10*w + 5*l

    if return_agent:
        return agent


def select(population):
    """
    Fitness-proportional selection (Roulette Wheel Selection).
    
    - Each individual has probability proportional to its fitness
    - Since we're MINIMIZING, we use 1/(fitness+1) for probability
    - Higher fitness = lower probability of selection

    For population size 200, this would use top 32% as group 1.
    """
    total_inverse_fitness = sum(1.0 / (ind.fitness + 1) for ind in population)
    
    r = random.uniform(0, total_inverse_fitness)
    accumulator = 0
    
    for ind in population:
        accumulator += 1.0 / (ind.fitness + 1)
        if accumulator >= r:
            return ind.copy()
    
    return population[-1].copy()


def tournament_selection(population, tournament_size=3):
    """
    Tournament selection (ALTERNATIVE - better for large populations).
    
    - Randomly pick k individuals
    - Return the best one
    - Simple and effective
    - tournament_size controls selection pressure
    
    This is often BETTER than fitness-proportional selection because:
    - No problems with very small fitness differences
    - Computationally efficient
    - Easy to parallelize
    """
    tournament = random.sample(population, tournament_size)
    winner = min(tournament, key=lambda ind: ind.fitness)
    return winner.copy()


def collect_nodes(node, parent=None, index=None, nodes=None):
    if nodes is None:
        nodes = []
    
    nodes.append((node, parent, index))
    
    for i, child in enumerate(node.get_children()):
        collect_nodes(child, node, i, nodes)
    
    return nodes


def crossover(p1, p2):
    child = p1.copy()
    
    child_nodes = collect_nodes(child.tree)
    parent2_nodes = collect_nodes(p2.tree)

    if not child_nodes or not parent2_nodes:
        return child

    node1, parent1, idx1 = random.choice(child_nodes)
    node2, _, _ = random.choice(parent2_nodes)

    if parent1 is None:
        # Replacing root
        child.tree = node2.copy()
    else:
        # Replacing subtree
        parent1.set_child(idx1, node2.copy())

    return child


def mutate(ind, max_depth):

    # 20% mutation rate
    if random.random() < 0.2:
        
        nodes = collect_nodes(ind.tree)
        if not nodes:
            return
        
        node, parent, idx = random.choice(nodes)
        
        new_subtree = generate_tree(max_depth, method=random.choice(['grow', 'full']))
        
        if parent is None:
            # Replacing root
            ind.tree = new_subtree
        else:
            # Replacing subtree
            parent.set_child(idx, new_subtree)


def over_selection(population, group1_percent=32):
    """
    Over-selection for large populations (TEXTBOOK METHOD).
    
    ALGORITHM (from slides):
    1. Rank population by fitness
    2. Divide into two groups:
       - Group 1: best x% of population
       - Group 2: other (100-x)%
    3. 80% of selections from group 1
    4. 20% of selections from group 2
    
    PARAMETERS (from textbook):
    "For pop size = 1000, 2000, 4000, 8000: x = 32%, 16%, 8%, 4%"
    
    For our pop size of 200, we use 32%.
    
    MOTIVATION (from textbook):
    "To increase efficiency" - focuses search on promising areas
    while maintaining some diversity.
    """
    # Sort population by fitness (best first)
    sorted_pop = sorted(population, key=lambda ind: ind.fitness)
    
    # Calculate group sizes
    group1_size = int(len(population) * group1_percent / 100)
    
    group1 = sorted_pop[:group1_size]
    group2 = sorted_pop[group1_size:]
    
    # 80% chance to select from group 1
    if random.random() < 0.8 and group1:
        return select(group1)
    elif group2:
        return select(group2)
    else:
        return select(group1)  # Fallback