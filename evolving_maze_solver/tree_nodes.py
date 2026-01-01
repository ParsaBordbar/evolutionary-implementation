import random
from configs import Direction


class Node:
    """Base class for all tree nodes"""
    def execute(self, agent, maze):
        raise NotImplementedError

    def copy(self):
        raise NotImplementedError

    def get_children(self):
        return []

    def set_child(self, index, child):
        raise IndexError


class MoveNode(Node):
    """Leaf node that moves the agent in a specific direction"""
    def __init__(self, direction):
        self.direction = direction

    def execute(self, agent, maze):
        agent.move(self.direction, maze)

    def copy(self):
        return MoveNode(self.direction)

    def __repr__(self):
        return f"MOVE({self.direction.name})"


class Sequence(Node):
    """Internal node that executes two subtrees in sequence"""
    def __init__(self, left, right):
        self.left = left
        self.right = right

    def execute(self, agent, maze):
        self.left.execute(agent, maze)
        self.right.execute(agent, maze)

    def copy(self):
        return Sequence(self.left.copy(), self.right.copy())

    def get_children(self):
        return [self.left, self.right]

    def set_child(self, index, child):
        if index == 0:
            self.left = child
        elif index == 1:
            self.right = child
        else:
            raise IndexError("Invalid child index")

    def __repr__(self):
        return f"SEQ({self.left},{self.right})"


def random_move_node():
    """Create a random move node"""
    return MoveNode(random.choice(list(Direction)))


def generate_tree(depth, method='grow', goal=(9, 9)):
    """
    Generate a random tree for genetic programming.
    
    Since we're only using MOVE and SEQUENCE nodes,
    this creates a linear program of moves.
    
    Args:
        depth: Maximum tree depth
        method: 'grow' or 'full'
        goal: Goal position (not used here, but kept for compatibility)
    
    Returns:
        Root node of the tree
    """
    if depth == 0:
        # Leaf node - always a move
        return random_move_node()
    
    if method == 'grow':
        # 50% chance to terminate with a move
        if random.random() < 0.5:
            return random_move_node()
        else:
            # Create a sequence
            return Sequence(
                generate_tree(depth - 1, method, goal),
                generate_tree(depth - 1, method, goal)
            )
    
    elif method == 'full':
        # Full method: always create sequences until depth 1
        if depth == 1:
            return random_move_node()
        else:
            return Sequence(
                generate_tree(depth - 1, method, goal),
                generate_tree(depth - 1, method, goal)
            )