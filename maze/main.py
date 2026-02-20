import random
from enum import Enum
from copy import deepcopy

# ================= CONFIG =================

MAZE = [
[0,0,1,0,0,0,1,0,0,0],
[1,0,1,0,1,0,1,0,1,0],
[1,0,0,0,1,0,0,0,1,0],
[1,1,1,0,1,1,1,0,1,0],
[0,0,0,0,0,0,1,0,0,0],
[0,1,1,1,1,0,1,1,1,0],
[0,0,0,0,1,0,0,0,0,0],
[1,1,1,0,1,1,1,1,0,1],
[0,0,0,0,0,0,0,1,0,0],
[0,1,1,1,1,1,0,0,0,0]
]

START = (0,0)
GOAL = (9,9)

POP_SIZE = 200
MAX_GEN = 150
MAX_DEPTH = 6
MAX_STEPS = 60
CROSSOVER_RATE = 0.9
MUTATION_RATE = 0.1

# ================= DIRECTIONS =================

class Direction(Enum):
    UP    = (0,-1)
    DOWN  = (0,1)
    LEFT  = (-1,0)
    RIGHT = (1,0)

# ================= GP NODES =================

class Node:
    def collect(self, out):
        raise NotImplementedError

class Move(Node):
    def __init__(self, d):
        self.d = d
    def collect(self, out):
        out.append(self.d)

class Prog2(Node):
    def __init__(self, a, b):
        self.a = a
        self.b = b
    def collect(self, out):
        self.a.collect(out)
        self.b.collect(out)

def extract_program(tree):
    program = []
    tree.collect(program)
    return program

class Rand(Node):
    def __init__(self, a, b):
        self.a = a
        self.b = b
    def eval(self):
        return random.choice([self.a.eval(), self.b.eval()])

# ================= TREE GENERATION =================

def random_tree(depth):
    if depth == 0:
        return Move(random.choice(list(Direction)))
    if random.random() < 0.5:
        return Prog2(random_tree(depth-1), random_tree(depth-1))
    return Rand(random_tree(depth-1), random_tree(depth-1))

# ================= FITNESS =================

def fitness(tree):
    program = extract_program(tree)
    if not program:
        return 1e9

    x, y = START
    visited = set()
    walls = loops = 0

    for step in range(MAX_STEPS):
        move = program[step % len(program)]
        dx, dy = move.value
        nx, ny = x + dx, y + dy

        if not (0 <= nx < 10 and 0 <= ny < 10):
            walls += 1
            continue

        if MAZE[ny][nx] == 1:
            walls += 1

        if (nx, ny) in visited:
            loops += 1

        visited.add((nx, ny))
        x, y = nx, ny

        if (x, y) == GOAL:
            return step  # small value, not instantly zero

    dist = abs(x - GOAL[0]) + abs(y - GOAL[1])
    return MAX_STEPS + 2 * dist + 10 * walls + 5 * loops

# ================= GP OPERATORS =================

def select(pop, fits):
    i,j = random.sample(range(len(pop)), 2)
    return pop[i] if fits[i] < fits[j] else pop[j]

def subtree_mutation(node, depth=0):
    if random.random() < 0.1:
        return random_tree(2)
    if isinstance(node, (Prog2, Rand)):
        node.a = subtree_mutation(node.a, depth+1)
        node.b = subtree_mutation(node.b, depth+1)
    return node

def crossover(a, b):
    if random.random() > CROSSOVER_RATE:
        return a.clone()
    return random.choice([a.clone(), b.clone()])

# ================= EVOLUTION =================

def evolve():
    pop = [random_tree(4) for _ in range(POP_SIZE)]

    for g in range(MAX_GEN):
        fits = [fitness(ind) for ind in pop]
        best = min(fits)
        print(f"Gen {g}: best = {best}")

        if best == 0:
            print("Goal reached")
            break

        new_pop = []
        for _ in range(POP_SIZE):
            p1 = select(pop, fits)
            p2 = select(pop, fits)
            child = crossover(p1, p2)
            if random.random() < MUTATION_RATE:
                child = subtree_mutation(child)
            new_pop.append(child)

        pop = new_pop

if __name__ == "__main__":
    evolve()
