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


class IfWallNearby(Node):
    """Internal node that branches based on wall detection"""
    def __init__(self, true_branch, false_branch):
        self.true_branch = true_branch
        self.false_branch = false_branch

    def execute(self, agent, maze):
        if self.wall_nearby(agent, maze):
            self.true_branch.execute(agent, maze)
        else:
            self.false_branch.execute(agent, maze)

    @staticmethod
    def wall_nearby(agent, maze):
        """Check if there's a wall in any of the 4 adjacent cells"""
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        for dx, dy in directions:
            nx, ny = agent.x + dx, agent.y + dy
            # If out of bounds or wall, return True
            if not (0 <= nx < len(maze[0]) and 0 <= ny < len(maze)):
                return True
            if maze[ny][nx] == 1:
                return True
        return False

    def copy(self):
        return IfWallNearby(self.true_branch.copy(), self.false_branch.copy())

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
        return f"IF_WALL({self.true_branch},{self.false_branch})"


class IfGoalClose(Node):
    """Internal node that branches based on distance to goal"""
    def __init__(self, true_branch, false_branch, goal):
        self.true_branch = true_branch
        self.false_branch = false_branch
        self.goal = goal

    def execute(self, agent, maze):
        if self.goal_close(agent):
            self.true_branch.execute(agent, maze)
        else:
            self.false_branch.execute(agent, maze)

    def goal_close(self, agent):
        """Check if goal is within Manhattan distance of 5"""
        distance = abs(agent.x - self.goal[0]) + abs(agent.y - self.goal[1])
        return distance <= 5

    def copy(self):
        return IfGoalClose(self.true_branch.copy(), self.false_branch.copy(), self.goal)

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
        return f"IF_CLOSE({self.true_branch},{self.false_branch})"


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
    return MoveNode(random.choice(list(Direction)))


def generate_tree(depth, method='grow', goal=(9, 9)):

    if depth == 0:
        return random_move_node()
    
    if method == 'grow': # randomly choose between terminal and non-terminal nodes
        if random.random() < 0.5:
            return random_move_node()
        else:
            node_type = random.choice([IfWallNearby, IfGoalClose, Sequence])
            if node_type == IfWallNearby:
                return IfWallNearby(
                    generate_tree(depth - 1, method, goal),
                    generate_tree(depth - 1, method, goal)
                )
            elif node_type == IfGoalClose:
                return IfGoalClose(
                    generate_tree(depth - 1, method, goal),
                    generate_tree(depth - 1, method, goal),
                    goal
                )
            else:  # Sequence
                return Sequence(
                    generate_tree(depth - 1, method, goal),
                    generate_tree(depth - 1, method, goal)
                )
    
    elif method == 'full': # always create full binary trees at each level
        if depth == 1:
            # At leaf level, only create terminal nodes
            return random_move_node()
        else:
            node_type = random.choice([IfWallNearby, IfGoalClose, Sequence])
            if node_type == IfWallNearby:
                return IfWallNearby(
                    generate_tree(depth - 1, method, goal),
                    generate_tree(depth - 1, method, goal)
                )
            elif node_type == IfGoalClose:
                return IfGoalClose(
                    generate_tree(depth - 1, method, goal),
                    generate_tree(depth - 1, method, goal),
                    goal
                )
            else:  # Sequence
                return Sequence(
                    generate_tree(depth - 1, method, goal),
                    generate_tree(depth - 1, method, goal)
                )