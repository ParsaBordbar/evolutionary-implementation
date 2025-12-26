class Agent:
    def __init__(self, start):
        self.x, self.y = start
        self.steps = 0
        self.wall_hits = 0
        self.path = []
        self.visited = set()
        self.visited.add((self.x, self.y))
            
    def move(self, direction, maze):
        dx, dy = direction.value
        nx, ny = self.x + dx, self.y + dy

        self.steps += 1

        if nx < 0 or ny < 0 or ny >= len(maze) or nx >= len(maze[0]):
            self.wall_hits += 1
            return

        if maze[ny][nx] == 1:
            self.wall_hits += 1
            return

        self.x, self.y = nx, ny
        self.visited.add((self.x, self.y))
        self.path.append((self.x, self.y))
