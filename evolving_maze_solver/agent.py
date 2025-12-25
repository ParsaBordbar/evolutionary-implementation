class Agent:
    def __init__(self, start):
        self.x, self.y = start
        self.steps = 0
        self.wall_hits = 0
        self.visited = set()
        self.visited.add((self.x, self.y))
        
    def move(self, direction, maze):
        dx, dy = direction
        new_x = self.x + dx
        new_y = self.y + dy

        if new_x < 0 or new_y < 0 or new_y >= len(maze) or new_x >= len(maze[0]):
            self.wall_hits += 1
            self.steps += 1
            return

        # Wall check
        if maze[new_y][new_x] == 1:
            self.wall_hits += 1
            self.steps += 1
            return

        self.x = new_x
        self.y = new_y
        self.steps += 1
        self.visited.add((self.x, self.y))
