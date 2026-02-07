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


# ============================================================================
# TERMINAL NODES (Leaves - Actions)
# ============================================================================

class MoveNode(Node):
    """
    Terminal node - executes a MOVE action.
    
    This is a TERMINAL in GP terminology (leaf node).
    It performs an action and returns.
    """
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


# ============================================================================
# FUNCTION NODES (Internal - Decision Points)
# ============================================================================

class IfWallNearby(Node):
    """
    Function node - checks if ANY wall is adjacent (in any of 4 directions).
    
    This is a FUNCTION in GP terminology (internal node).
    It evaluates a condition and executes one of two subtrees.
    
    This is BETTER than having separate IfWallUp, IfWallDown, etc.
    because it's more general and creates simpler, more effective trees.
    """
    def __init__(self, true_branch, false_branch):
        self.true_branch = true_branch
        self.false_branch = false_branch

    def execute(self, agent, maze):
        # Check all 4 directions for walls
        has_wall = False
        for direction in Direction:
            dx, dy = direction.value
            nx, ny = agent.x + dx, agent.y + dy
            
            # Out of bounds OR wall cell = obstacle
            if (nx < 0 or ny < 0 or ny >= len(maze) or nx >= len(maze[0]) or
                maze[ny][nx] == 1):
                has_wall = True
                break
        
        if has_wall:
            self.true_branch.execute(agent, maze)
        else:
            self.false_branch.execute(agent, maze)

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
            raise IndexError("IfWallNearby has only 2 children (index 0 or 1)")

    def __repr__(self):
        return "IF_WALL_NEARBY"


class IfGoalClose(Node):
    """
    Function node - checks if goal is within Manhattan distance of 5.
    
    This helps the agent know when it's getting close to the target
    and should adjust its strategy.
    """
    def __init__(self, true_branch, false_branch):
        self.true_branch = true_branch
        self.false_branch = false_branch
        self.threshold = 5  # Consider "close" if within 5 steps

    def execute(self, agent, maze):
        from configs import GOAL  # Import here to avoid circular dependency
        
        # Calculate Manhattan distance to goal
        distance = abs(agent.x - GOAL[0]) + abs(agent.y - GOAL[1])
        
        if distance <= self.threshold:
            self.true_branch.execute(agent, maze)
        else:
            self.false_branch.execute(agent, maze)

    def copy(self):
        return IfGoalClose(self.true_branch.copy(), self.false_branch.copy())

    def get_children(self):
        return [self.true_branch, self.false_branch]

    def set_child(self, index, child):
        if index == 0:
            self.true_branch = child
        elif index == 1:
            self.false_branch = child
        else:
            raise IndexError("IfGoalClose has only 2 children (index 0 or 1)")

    def __repr__(self):
        return "IF_GOAL_CLOSE"


class Sequence(Node):
    """
    Function node - executes two actions in sequence.
    
    This allows trees to perform multiple moves per execution,
    creating more complex behaviors.
    
    Example: "Move RIGHT, then Move DOWN"
    """
    def __init__(self, first, second):
        self.first = first
        self.second = second

    def execute(self, agent, maze):
        self.first.execute(agent, maze)
        self.second.execute(agent, maze)

    def copy(self):
        return Sequence(self.first.copy(), self.second.copy())

    def get_children(self):
        return [self.first, self.second]

    def set_child(self, index, child):
        if index == 0:
            self.first = child
        elif index == 1:
            self.second = child
        else:
            raise IndexError("Sequence has only 2 children (index 0 or 1)")

    def __repr__(self):
        return "SEQUENCE"


# ============================================================================
# TREE GENERATION FUNCTIONS
# ============================================================================

def random_terminal():
    """
    Create a random TERMINAL node (leaf).
    
    In GP, terminals are the base elements that perform actions.
    Here, our only terminal type is MoveNode with random direction.
    """
    return MoveNode(random.choice(list(Direction)))


def random_function(depth, method, goal):
    """
    Create a random FUNCTION node (internal node with children).
    
    Functions in GP are operators that have child nodes.
    We have 3 function types:
    - IfWallNearby: Conditional based on obstacles
    - IfGoalClose: Conditional based on proximity to goal
    - Sequence: Sequential execution of two subtrees
    """
    function_class = random.choice([IfWallNearby, IfGoalClose, Sequence])
    
    # Recursively generate children
    child1 = generate_tree(depth - 1, method, goal)
    child2 = generate_tree(depth - 1, method, goal)
    
    return function_class(child1, child2)


def generate_tree(depth, method='grow', goal=(9, 9)):
    """
    Generate a random GP tree using GROW or FULL method.
    
    This implements the TEXTBOOK STANDARD algorithms:
    
    FULL METHOD:
    - All branches have EXACTLY depth = Dmax
    - Creates balanced, bushy trees
    - At depth < Dmax: ONLY functions (internal nodes)
    - At depth = Dmax: ONLY terminals (leaves)
    
    GROW METHOD:
    - Branches have depth ≤ Dmax (variable)
    - Creates irregular, varied trees
    - At depth < Dmax: RANDOM choice of functions OR terminals
    - At depth = Dmax: ONLY terminals (leaves)
    
    Args:
        depth: Remaining depth (0 = must be terminal)
        method: 'grow' or 'full'
        goal: Goal position (for compatibility)
    
    Returns:
        Root node of the generated tree
    """
    
    # BASE CASE: depth 0 = MUST be terminal
    if depth == 0:
        return random_terminal()
    
    # FULL METHOD: Always use functions until depth 1
    if method == 'full':
        if depth == 1:
            # Next level must be terminals, so this is the last function level
            return random_terminal()
        else:
            # Use function node, children will be at depth-1
            return random_function(depth, method, goal)
    
    # GROW METHOD: Random choice between terminal and function
    elif method == 'grow':
        # 50% chance to terminate early OR forced terminal at depth 1
        if random.random() < 0.5 or depth == 1:
            return random_terminal()
        else:
            return random_function(depth, method, goal)
    
    else:
        raise ValueError(f"Unknown method: {method}. Use 'grow' or 'full'")


# ============================================================================
# INITIALIZATION: Ramped Half-and-Half
# ============================================================================

def ramped_half_and_half(population_size, max_depth):
    """
    Ramped Half-and-Half initialization (TEXTBOOK STANDARD).
    
    This creates diversity in the initial population by:
    1. Using a range of depths (1 to max_depth)
    2. For each depth, creating half with GROW and half with FULL
    
    Example with pop_size=200, max_depth=6:
    - Depth 1: 33 trees (17 grow, 16 full)
    - Depth 2: 33 trees (17 grow, 16 full)
    - Depth 3: 33 trees (17 grow, 16 full)
    - Depth 4: 33 trees (17 grow, 16 full)
    - Depth 5: 34 trees (17 grow, 17 full)
    - Depth 6: 34 trees (17 grow, 17 full)
    
    This is superior to using one method or one depth because:
    - Gets both bushy (full) and irregular (grow) trees
    - Gets both shallow and deep trees
    - Maximizes initial diversity
    """
    from configs import GOAL
    population = []
    
    # Distribute population across depth levels
    trees_per_depth = population_size // max_depth
    remainder = population_size % max_depth
    
    for depth in range(1, max_depth + 1):
        # How many trees at this depth?
        count = trees_per_depth
        if depth <= remainder:
            count += 1
        
        # Half grow, half full
        grow_count = count // 2
        full_count = count - grow_count
        
        # Generate trees
        for _ in range(grow_count):
            population.append(generate_tree(depth, 'grow', GOAL))
        
        for _ in range(full_count):
            population.append(generate_tree(depth, 'full', GOAL))
    
    return population