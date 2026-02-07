class Agent:
    def __init__(self, start):
        self.x, self.y = start
        self.steps = 0
        self.wall_hits = 0
        self.path = [(self.x, self.y)]
        self.visited = set()
        self.visited.add((self.x, self.y))
        self.revisit_count = 0  # Track how many times we revisit cells
            
    def move(self, direction, maze):
        dx, dy = direction.value
        nx, ny = self.x + dx, self.y + dy

        self.steps += 1

        # Check if move is VALID
        # Invalid if: out of bounds OR hits wall
        if nx < 0 or ny < 0 or ny >= len(maze) or nx >= len(maze[0]):
            # Out of bounds - count as wall hit, don't move
            self.wall_hits += 1
            return
        
        if maze[ny][nx] == 1:
            # Hit a wall - count collision, don't move
            self.wall_hits += 1
            return

        # VALID MOVE
        self.x, self.y = nx, ny
        
        # loop penalty
        if (self.x, self.y) in self.visited:
            self.revisit_count += 1
        
        self.visited.add((self.x, self.y))
        self.path.append((self.x, self.y))