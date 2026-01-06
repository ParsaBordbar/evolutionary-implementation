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
    """Leaf node - executes a MOVE action"""
    def __init__(self, direction):
        self.direction = direction

    def execute(self, agent, maze):
        agent.move(self.direction, maze)

    def copy(self):
        return MoveNode(self.direction)

    def get_children(self):
        return []

    def __repr__(self):
        return f"MOVE({self.direction.name})"


class IfWallUp(Node):
    """Internal node - checks if wall is above (UP direction)"""
    def __init__(self, true_branch, false_branch):
        self.true_branch = true_branch
        self.false_branch = false_branch

    def execute(self, agent, maze):
        if self._has_wall_in_direction(agent, maze, Direction.UP):
            self.true_branch.execute(agent, maze)
        else:
            self.false_branch.execute(agent, maze)

    @staticmethod
    def _has_wall_in_direction(agent, maze, direction):
        dx, dy = direction.value
        nx, ny = agent.x + dx, agent.y + dy
        
        # Out of bounds counts as wall
        if nx < 0 or ny < 0 or ny >= len(maze) or nx >= len(maze[0]):
            return True
        
        # Check if wall
        return maze[ny][nx] == 1

    def copy(self):
        return IfWallUp(self.true_branch.copy(), self.false_branch.copy())

    def get_children(self):
        return [self.true_branch, self.false_branch]

    def set_child(self, index, child):
        if index == 0:
            self.true_branch = child
        elif index == 1:
            self.false_branch = child
        else:
            raise IndexError("Invalid child index")

    def __repr__(self):
        return "IF_WALL_UP"


class IfWallDown(Node):
    """Internal node - checks if wall is below (DOWN direction)"""
    def __init__(self, true_branch, false_branch):
        self.true_branch = true_branch
        self.false_branch = false_branch

    def execute(self, agent, maze):
        if self._has_wall_in_direction(agent, maze, Direction.DOWN):
            self.true_branch.execute(agent, maze)
        else:
            self.false_branch.execute(agent, maze)

    @staticmethod
    def _has_wall_in_direction(agent, maze, direction):
        dx, dy = direction.value
        nx, ny = agent.x + dx, agent.y + dy
        
        if nx < 0 or ny < 0 or ny >= len(maze) or nx >= len(maze[0]):
            return True
        
        return maze[ny][nx] == 1

    def copy(self):
        return IfWallDown(self.true_branch.copy(), self.false_branch.copy())

    def get_children(self):
        return [self.true_branch, self.false_branch]

    def set_child(self, index, child):
        if index == 0:
            self.true_branch = child
        elif index == 1:
            self.false_branch = child
        else:
            raise IndexError("Invalid child index")

    def __repr__(self):
        return "IF_WALL_DOWN"


class IfWallLeft(Node):
    """Internal node - checks if wall is to the left (LEFT direction)"""
    def __init__(self, true_branch, false_branch):
        self.true_branch = true_branch
        self.false_branch = false_branch

    def execute(self, agent, maze):
        if self._has_wall_in_direction(agent, maze, Direction.LEFT):
            self.true_branch.execute(agent, maze)
        else:
            self.false_branch.execute(agent, maze)

    @staticmethod
    def _has_wall_in_direction(agent, maze, direction):
        dx, dy = direction.value
        nx, ny = agent.x + dx, agent.y + dy
        
        if nx < 0 or ny < 0 or ny >= len(maze) or nx >= len(maze[0]):
            return True
        
        return maze[ny][nx] == 1

    def copy(self):
        return IfWallLeft(self.true_branch.copy(), self.false_branch.copy())

    def get_children(self):
        return [self.true_branch, self.false_branch]

    def set_child(self, index, child):
        if index == 0:
            self.true_branch = child
        elif index == 1:
            self.false_branch = child
        else:
            raise IndexError("Invalid child index")

    def __repr__(self):
        return "IF_WALL_LEFT"


class IfWallRight(Node):
    """Internal node - checks if wall is to the right (RIGHT direction)"""
    def __init__(self, true_branch, false_branch):
        self.true_branch = true_branch
        self.false_branch = false_branch

    def execute(self, agent, maze):
        if self._has_wall_in_direction(agent, maze, Direction.RIGHT):
            self.true_branch.execute(agent, maze)
        else:
            self.false_branch.execute(agent, maze)

    @staticmethod
    def _has_wall_in_direction(agent, maze, direction):
        dx, dy = direction.value
        nx, ny = agent.x + dx, agent.y + dy
        
        if nx < 0 or ny < 0 or ny >= len(maze) or nx >= len(maze[0]):
            return True
        
        return maze[ny][nx] == 1

    def copy(self):
        return IfWallRight(self.true_branch.copy(), self.false_branch.copy())

    def get_children(self):
        return [self.true_branch, self.false_branch]

    def set_child(self, index, child):
        if index == 0:
            self.true_branch = child
        elif index == 1:
            self.false_branch = child
        else:
            raise IndexError("Invalid child index")

    def __repr__(self):
        return "IF_WALL_RIGHT"


def random_move_node():
    """Create a random MOVE leaf node"""
    return MoveNode(random.choice(list(Direction)))


def generate_tree(depth, method='grow', goal=(9, 9)):
    """
    Generate a random tree for genetic programming.
    
    Leaf nodes: MOVE actions only
    Internal nodes: Condition checks (IF_WALL_UP/DOWN/LEFT/RIGHT)
    
    Args:
        depth: Maximum tree depth (0 = leaf only)
        method: 'grow' or 'full'
        goal: Goal position (not used, kept for compatibility)
    
    Returns:
        Root node of the tree
    """
    if depth == 0:
        # Leaf node - must be a MOVE
        return random_move_node()
    
    if method == 'grow':
        # Randomly choose terminal or non-terminal
        if random.random() < 0.5:
            return random_move_node()
        else:
            # Choose a condition node
            condition_class = random.choice([IfWallUp, IfWallDown, IfWallLeft, IfWallRight])
            return condition_class(
                generate_tree(depth - 1, method, goal),
                generate_tree(depth - 1, method, goal)
            )
    
    elif method == 'full':
        # Full method: always create non-terminals until depth 1
        if depth == 1:
            return random_move_node()
        else:
            condition_class = random.choice([IfWallUp, IfWallDown, IfWallLeft, IfWallRight])
            return condition_class(
                generate_tree(depth - 1, method, goal),
                generate_tree(depth - 1, method, goal)
            )