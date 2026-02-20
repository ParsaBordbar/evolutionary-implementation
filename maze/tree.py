import random
from enum import Enum
from dataclasses import dataclass

from node_types import MoveNode, RandNode, SeqNode

class Direction(Enum):
    UP = (0, -1)
    DOWN = (0, 1)
    LEFT = (-1, 0)
    RIGHT = (1, 0)

def random_tree(depth):
    if depth == 0:
        return MoveNode(random.choice(list(Direction)))

    if random.random() < 0.5:
        return SeqNode(
            random_tree(depth - 1),
            random_tree(depth - 1)
        )
    else:
        return RandNode(
            random_tree(depth - 1),
            random_tree(depth - 1)
        )
