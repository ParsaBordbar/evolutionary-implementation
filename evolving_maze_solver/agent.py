class Agent:
    """
    Agent that navigates the maze.
    
    Key points:
    - Can move in ALL 4 directions (even through walls)
    - Wall cells are "penalty cells", not blocking walls
    - Collisions are penalized in fitness, not prevented
    """
    def __init__(self, start):
        self.x, self.y = start
        self.steps = 0
        self.wall_hits = 0
        self.path = [(self.x, self.y)]
        self.visited = set()
        self.visited.add((self.x, self.y))
            
    def move(self, direction, maze):
        """
        Move agent in given direction.
        
        Rules:
        - Agent ALWAYS increments steps counter
        - Agent CAN move through walls (they don't block)
        - Wall collisions (maze[ny][nx] == 1) increment wall_hits counter
        - Position ALWAYS updates (even if moving through wall)
        """
        dx, dy = direction.value
        nx, ny = self.x + dx, self.y + dy

        self.steps += 1

        # Check bounds - count as collision
        if nx < 0 or ny < 0 or ny >= len(maze) or nx >= len(maze[0]):
            self.wall_hits += 1
            return

        # Check if wall - count as collision but ALLOW movement
        if maze[ny][nx] == 1:
            self.wall_hits += 1

        # ALWAYS update position (even through walls)
        self.x, self.y = nx, ny
        self.visited.add((self.x, self.y))
        self.path.append((self.x, self.y))