# TODO Implement the Nodes logic and tree!
class Node:
    def execute(self, agent, maze):
        raise NotImplementedError
    
class MoveNode(Node):
    def __init__(self, direction):
        self.direction = direction

    def execute(self, agent, maze):
        agent.move(self.direction)