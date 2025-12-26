import random
from agent import Agent

def evaluate(ind, maze, start, goal, max_steps, return_agent=False):
    agent = Agent(start)

    for _ in range(max_steps):
        ind.tree.execute(agent, maze)
        if (agent.x, agent.y) == goal:
            break

    if (agent.x, agent.y) == goal:
        ind.fitness = 0
    else:
        s = agent.steps
        d = abs(agent.x - goal[0]) + abs(agent.y - goal[1])
        w = agent.wall_hits
        l = agent.steps - len(agent.visited)
        ind.fitness = s + 2*d + 10*w + 5*l

    if return_agent:
        return agent


def select(population):
    total = sum(1/(i.fitness+1) for i in population)
    r = random.uniform(0, total)
    acc = 0
    for ind in population:
        acc += 1/(ind.fitness+1)
        if acc >= r:
            return ind.copy()


def collect_nodes(node, parent=None, index=None, nodes=None):
    if nodes is None:
        nodes = []
    nodes.append((node, parent, index))
    for i, child in enumerate(node.get_children()):
        collect_nodes(child, node, i, nodes)
    return nodes


def crossover(p1, p2):
    child = p1.copy()
    n1 = collect_nodes(child.tree)
    n2 = collect_nodes(p2.tree)

    node1, parent, idx = random.choice(n1)
    node2, _, _ = random.choice(n2)

    if parent is None:
        child.tree = node2.copy()
    else:
        parent.set_child(idx, node2.copy())

    return child


def mutate(ind, max_depth):
    if random.random() < 0.2:
        from tree_nodes import generate_tree
        ind.tree = generate_tree(max_depth)