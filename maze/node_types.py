from dataclasses import dataclass
import random
from configs import Direction


class Node:
    def execute(self, state):
        raise NotImplementedError

@dataclass
class MoveNode(Node):
    direction: Direction

    def execute(self, state):
        state["moves"].append(self.direction)

@dataclass
class SeqNode(Node):
    left: Node
    right: Node

    def execute(self, state):
        self.left.execute(state)
        self.right.execute(state)

@dataclass
class RandNode(Node):
    left: Node
    right: Node

    def execute(self, state):
        random.choice([self.left, self.right]).execute(state)
