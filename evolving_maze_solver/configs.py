from enum import Enum

# Maze: agent has NO sensing - must learn pure sequence
# This makes random solutions nearly impossible
MAZE = [
    [0, 1, 0, 1, 0, 0, 1, 0, 0, 0],
    [0, 1, 0, 1, 1, 0, 1, 1, 1, 0],
    [0, 0, 0, 0, 1, 0, 0, 0, 1, 0],
    [1, 1, 1, 0, 1, 1, 1, 0, 1, 0],
    [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 1, 1, 1, 0, 1, 1, 1, 1],
    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    [1, 1, 1, 0, 1, 1, 1, 1, 0, 1],
    [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
    [0, 1, 1, 1, 1, 1, 0, 0, 0, 0]
]

# Maze Confs
START = (0, 0)
GOAL = (9, 9)

# GP-OP Confs
POP_SIZE = 200
MAX_GEN = 150
MAX_DEPTH = 6   # Deeper trees needed for longer sequences
MAX_STEPS = 50  # Max moves per episode

# Move Enum
class Direction(Enum):
    UP = (0, -1)
    DOWN = (0, 1)
    LEFT = (-1, 0)
    RIGHT = (1, 0)