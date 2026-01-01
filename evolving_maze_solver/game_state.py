class GameState:    
    def __init__(self, grid, start_pos, goal_pos):
        self.grid = grid
        self.x, self.y = start_pos
        self.goal_x, self.goal_y = goal_pos
        self.wall_collisions = 0
        self.path_taken = []
    
    def get_all_possible_moves(self):
        return ['up', 'down', 'left', 'right']
    
    def _calculate_new_position(self, direction):
        """Calculate where we would be after move"""
        x, y = self.x, self.y
        
        if direction == 'up':
            y -= 1
        elif direction == 'down':
            y += 1
        elif direction == 'left':
            x -= 1
        elif direction == 'right':
            x += 1
        
        return x, y
    
    # def is_wall(self, x, y):
    #     """Check if position has a wall"""
    #     if x < 0 or x >= len(self.grid[0]) or y < 0 or y >= len(self.grid):
    #         return True  # Out of bounds = wall
    #     return self.grid[y][x] == 1  # 1 = wall
    
    def is_node(self, x, y):
        """Check if position is a valid node (not wall, in bounds)"""
        if x < 0 or x >= len(self.grid[0]) or y < 0 or y >= len(self.grid):
            return False
        return self.grid[y][x] in [0, 2, 3]  # 0=empty, 2=node, 3=goal
    
    def move(self, direction):
        """
        move in ANY direction.
        Even if it hits a wall!
        """
        new_x, new_y = self._calculate_new_position(direction)
        
        # Check if wall
        wall_hit = self.is_wall(new_x, new_y)
        
        # Track wall collision
        if wall_hit:
            self.wall_collisions += 1
            # Position does NOT change
        else:
            # Only move if not a wall
            self.x, self.y = new_x, new_y
        
        # Record the move
        self.path_taken.append({
            'direction': direction,
            'wall_hit': wall_hit,
            'final_x': self.x,
            'final_y': self.y
        })
        
        return {
            'moved': not wall_hit,
            'wall_hit': wall_hit,
            'x': self.x,
            'y': self.y,
            'is_goal': self.is_goal()
        }
    
    def is_goal(self):
        return (self.x == self.goal_x and self.y == self.goal_y)
    
    def is_at_node(self):
        if self.x < 0 or self.x >= len(self.grid[0]) or self.y < 0 or self.y >= len(self.grid):
            return False
        return self.grid[self.y][self.x] in [2, 3]
    
    def copy(self):
        new_state = GameState(self.grid, (self.x, self.y), (self.goal_x, self.goal_y))
        new_state.wall_collisions = self.wall_collisions
        new_state.path_taken = self.path_taken.copy()
        return new_state
