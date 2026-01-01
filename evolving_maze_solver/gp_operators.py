import random
from agent import Agent

def evaluate(ind, maze, start, goal, max_steps, return_agent=False):
    """
    Evaluate fitness of an individual.
    
    Since the agent has NO sensing abilities, it must learn through
    pure sequence evolution - trial and error.
    
    Only solutions that actually reach the goal get fitness = 0.
    Everything else gets harsh penalties based on distance from goal.
    """
    agent = Agent(start)

    for step_count in range(max_steps):
        ind.tree.execute(agent, maze)
        
        if (agent.x, agent.y) == goal:
            break

    # Calculate fitness
    if (agent.x, agent.y) == goal:
        # Perfect! Reached the goal
        ind.fitness = agent.steps  # Reward shorter paths slightly
    else:
        # Did NOT reach goal - massive penalty
        distance_to_goal = abs(agent.x - goal[0]) + abs(agent.y - goal[1])
        
        # Fitness formula: very harsh for not reaching goal
        # This forces evolution to find actual solutions
        ind.fitness = (max_steps * 2) + (distance_to_goal * 100) + (agent.wall_hits * 50)

    if return_agent:
        return agent


def select(population):
    """
    Fitness-proportional selection using roulette wheel.
    
    Lower fitness = higher probability of selection
    """
    # Find max fitness
    max_fitness = max(i.fitness for i in population)
    min_fitness = min(i.fitness for i in population)
    
    # Avoid division by zero
    if max_fitness == min_fitness:
        return population[random.randint(0, len(population)-1)].copy()
    
    # Invert: higher fitness value -> lower probability
    inverted = [max_fitness - i.fitness for i in population]
    total = sum(inverted)
    
    if total <= 0:
        return population[random.randint(0, len(population)-1)].copy()
    
    r = random.uniform(0, total)
    acc = 0
    for i, ind in enumerate(population):
        acc += inverted[i]
        if acc >= r:
            return ind.copy()
    
    return population[-1].copy()


def collect_nodes(node, parent=None, index=None, nodes=None):
    """Collect all nodes in tree for crossover"""
    if nodes is None:
        nodes = []
    nodes.append((node, parent, index))
    for i, child in enumerate(node.get_children()):
        collect_nodes(child, node, i, nodes)
    return nodes


def crossover(p1, p2):
    """
    Subtree crossover - swap random subtrees between parents.
    """
    child = p1.copy()
    n1 = collect_nodes(child.tree)
    n2 = collect_nodes(p2.tree)

    if not n1 or not n2:
        return child

    node1, parent, idx = random.choice(n1)
    node2, _, _ = random.choice(n2)

    if parent is None:
        child.tree = node2.copy()
    else:
        parent.set_child(idx, node2.copy())

    return child


def mutate(ind, max_depth):
    """
    Mutation: 15% chance to replace a random subtree with new random tree.
    """
    if random.random() < 0.15:
        from tree_nodes import generate_tree
        # Find a random node and replace it
        nodes = collect_nodes(ind.tree)
        if nodes:
            node, parent, idx = random.choice(nodes)
            new_tree = generate_tree(max_depth, method=random.choice(['grow', 'full']))
            if parent is None:
                ind.tree = new_tree
            else:
                parent.set_child(idx, new_tree)