import random
from agent import Agent

def evaluate(ind, maze, start, goal, max_steps, return_agent=False):
    """
    Evaluate fitness using the CORRECT formula:
    
    F = s + 2*d + 10*w + 5*l
    
    where:
      s = steps taken
      d = distance to goal (Manhattan distance)
      w = wall hits
      l = loops (steps - unique_cells_visited)
    
    This formula ALWAYS applies, even when goal is reached!
    When goal reached: d=0, so F = s + 10*w + 5*l
    Only F=0 when: optimal steps + 0 wall hits + 0 loops
    """
    agent = Agent(start)

    for _ in range(max_steps):
        ind.tree.execute(agent, maze)
        if (agent.x, agent.y) == goal:
            break

    # Always use the fitness formula
    s = agent.steps
    d = abs(agent.x - goal[0]) + abs(agent.y - goal[1])
    w = agent.wall_hits
    l = agent.steps - len(agent.visited)
    
    ind.fitness = s + 2*d + 10*w + 5*l

    if return_agent:
        return agent


def select(population):
    """
    Fitness-proportional selection using roulette wheel.
    
    Lower fitness = higher probability of selection
    """
    total = sum(1/(i.fitness + 1) for i in population)
    r = random.uniform(0, total)
    acc = 0
    for ind in population:
        acc += 1/(ind.fitness + 1)
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
    Mutation: 20% chance to replace a random subtree.
    """
    if random.random() < 0.2:
        from tree_nodes import generate_tree
        nodes = collect_nodes(ind.tree)
        if nodes:
            node, parent, idx = random.choice(nodes)
            new_tree = generate_tree(max_depth, method=random.choice(['grow', 'full']))
            if parent is None:
                ind.tree = new_tree
            else:
                parent.set_child(idx, new_tree)