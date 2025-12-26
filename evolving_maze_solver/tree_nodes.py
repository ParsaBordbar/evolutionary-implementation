import random
from configs import Direction

class Node:
    def execute(self, agent, maze):
        raise NotImplementedError

    def copy(self):
        raise NotImplementedError

    def get_children(self):
        return []

    def set_child(self, index, child):
        raise IndexError


class MoveNode(Node):
    def __init__(self, direction):
        self.direction = direction

    def execute(self, agent, maze):
        agent.move(self.direction, maze)

    def copy(self):
        return MoveNode(self.direction)

    def __repr__(self):
        return f"MOVE({self.direction.name})"


class IfWallNearby(Node):
    def __init__(self, true_branch, false_branch):
        self.true_branch = true_branch
        self.false_branch = false_branch

    def execute(self, agent, maze):
        if self.wall_nearby(agent, maze):
            self.true_branch.execute(agent, maze)
        else:
            self.false_branch.execute(agent, maze)

    def wall_nearby(self, agent, maze):
        for dx, dy in [(0,1),(0,-1),(1,0),(-1,0)]:
            nx, ny = agent.x + dx, agent.y + dy
            if 0 <= nx < len(maze[0]) and 0 <= ny < len(maze):
                if maze[ny][nx] == 0:
                    return False
        return True

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
            raise IndexError

    def __repr__(self):
        return f"IF({self.true_branch},{self.false_branch})"


def random_move_node():
    return MoveNode(random.choice(list(Direction)))


def generate_tree(depth):
    if depth == 0 or random.random() < 0.5:
        return random_move_node()
    return IfWallNearby(generate_tree(depth-1), generate_tree(depth-1))