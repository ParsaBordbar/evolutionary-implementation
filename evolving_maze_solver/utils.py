import matplotlib.pyplot as plt
import numpy as np


def visualize_maze(maze, start, goal, path=None):
    """
    Visualize the maze with optional path overlay.
    
    Args:
        maze: 2D list representing the maze (0=open, 1=wall)
        start: Starting position (x, y)
        goal: Goal position (x, y)
        path: List of positions visited by the agent
    """
    grid = np.array(maze)

    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Display maze (walls in black, open spaces in white)
    ax.imshow(grid, cmap="gray_r", origin='upper')

    # Draw path if provided
    if path and len(path) > 1:
        path_array = np.array(path)
        ax.plot(path_array[:, 0], path_array[:, 1], 'b-', linewidth=2.5, 
                alpha=0.7, label=f'Agent Path ({len(path)} steps)', zorder=3)
        # Show direction with arrows
        for i in range(0, len(path_array)-1, max(1, len(path_array)//10)):
            ax.annotate('', xy=path_array[i+1], xytext=path_array[i],
                       arrowprops=dict(arrowstyle='->', color='blue', lw=1.5, alpha=0.5),
                       zorder=3)

    # Mark start and goal
    ax.scatter(start[0], start[1], c='green', s=300, marker='s', 
              label='Start', zorder=5, edgecolors='darkgreen', linewidth=2)
    ax.scatter(goal[0], goal[1], c='red', s=300, marker='*', 
              label='Goal', zorder=5, edgecolors='darkred', linewidth=2)

    ax.set_title("Maze Solution Visualization", fontsize=16, fontweight='bold')
    ax.set_xlabel("X Coordinate", fontsize=12)
    ax.set_ylabel("Y Coordinate", fontsize=12)
    ax.legend(loc='upper right', fontsize=11)
    ax.grid(False)
    ax.set_xticks(range(10))
    ax.set_yticks(range(10))
    
    plt.tight_layout()
    plt.show()


def print_tree_structure(node, depth=0, prefix=""):
    """
    Print the tree structure in a readable format.
    
    FIXED VERSION: Handles all node types correctly.
    
    Args:
        node: Root node of the tree
        depth: Current depth (for indentation)
        prefix: Prefix string for visualization
    """
    indent = "  " * depth
    
    if node is None:
        return
    
    # Get node type name
    node_type = type(node).__name__
    
    # Print current node based on type
    if node_type == "MoveNode":
        print(f"{indent}├─ MOVE({node.direction.name})")
    
    elif node_type == "IfWallNearby":
        print(f"{indent}├─ IF_WALL_NEARBY?")
        print(f"{indent}│  ├─ IF YES:")
        print_tree_structure(node.true_branch, depth + 2, "│  ")
        print(f"{indent}│  └─ IF NO:")
        print_tree_structure(node.false_branch, depth + 2, "│  ")
    
    elif node_type == "IfGoalClose":
        print(f"{indent}├─ IF_GOAL_CLOSE (distance ≤ 5)?")
        print(f"{indent}│  ├─ IF YES:")
        print_tree_structure(node.true_branch, depth + 2, "│  ")
        print(f"{indent}│  └─ IF NO:")
        print_tree_structure(node.false_branch, depth + 2, "│  ")
    
    elif node_type == "Sequence":
        print(f"{indent}├─ SEQUENCE")
        print(f"{indent}│  ├─ FIRST:")
        print_tree_structure(node.first, depth + 2, "│  ")  # FIXED: use .first
        print(f"{indent}│  └─ THEN:")
        print_tree_structure(node.second, depth + 2, "│  ")  # FIXED: use .second
    
    # Support for old node types (if they still exist)
    elif node_type in ["IfWallUp", "IfWallDown", "IfWallLeft", "IfWallRight"]:
        print(f"{indent}├─ {node_type.upper()}")
        print(f"{indent}│  ├─ IF YES:")
        print_tree_structure(node.true_branch, depth + 2, "│  ")
        print(f"{indent}│  └─ IF NO:")
        print_tree_structure(node.false_branch, depth + 2, "│  ")
    
    else:
        # Generic fallback for any other node type
        print(f"{indent}├─ {node_type}")
        # Try to get children using get_children() method
        try:
            children = node.get_children()
            for i, child in enumerate(children):
                print(f"{indent}│  ├─ CHILD {i}:")
                print_tree_structure(child, depth + 2, "│  ")
        except (AttributeError, TypeError):
            pass


def print_maze_with_path(maze, start, goal, path):
    """
    Print maze to console with path marked.
    
    Args:
        maze: 2D list representing the maze
        start: Starting position
        goal: Goal position
        path: List of positions visited
    """
    display = [['.' for _ in range(len(maze[0]))] for _ in range(len(maze))]
    
    # Mark walls
    for y in range(len(maze)):
        for x in range(len(maze[0])):
            if maze[y][x] == 1:
                display[y][x] = '█'
            elif maze[y][x] == 0:
                display[y][x] = ' '
    
    # Mark path
    for pos in path[1:-1]:  # Skip start and end
        x, y = pos
        if 0 <= x < len(maze[0]) and 0 <= y < len(maze):
            if display[y][x] == ' ':
                display[y][x] = '·'
    
    # Mark start and goal
    start_x, start_y = start
    goal_x, goal_y = goal
    display[start_y][start_x] = 'S'
    display[goal_y][goal_x] = 'G'
    
    print("\nMaze Solution Visualization (Console):")
    print("S = Start, G = Goal, · = Path, █ = Wall, (space) = Open")
    print("┌" + "─" * (len(maze[0]) * 2 - 1) + "┐")
    for row in display:
        print("│" + " ".join(row) + "│")
    print("└" + "─" * (len(maze[0]) * 2 - 1) + "┘")
    print()


def get_path_sequence(path):
    """
    Convert path coordinates to movement sequence.
    
    Args:
        path: List of (x, y) coordinates
        
    Returns:
        List of movement strings (UP, DOWN, LEFT, RIGHT)
    """
    if len(path) < 2:
        return []
    
    movements = []
    for i in range(len(path) - 1):
        x1, y1 = path[i]
        x2, y2 = path[i + 1]
        
        dx = x2 - x1
        dy = y2 - y1
        
        if dx > 0:
            movements.append("RIGHT")
        elif dx < 0:
            movements.append("LEFT")
        elif dy > 0:
            movements.append("DOWN")
        elif dy < 0:
            movements.append("UP")
    
    return movements


def print_movement_sequence(path):
    """
    Print the sequence of moves taken to solve the maze.
    
    Args:
        path: List of (x, y) coordinates
    """
    movements = get_path_sequence(path)
    
    print("\nMOVEMENT SEQUENCE")
    print("="*50)
    print(f"Total Moves: {len(movements)}")
    print("\nSequence:")
    
    # Group movements
    current_move = None
    count = 0
    sequence_str = ""
    
    for move in movements:
        if move == current_move:
            count += 1
        else:
            if current_move is not None:
                sequence_str += f"{count}×{current_move} → "
            current_move = move
            count = 1
    
    if current_move is not None:
        sequence_str += f"{count}×{current_move}"
    
    print(sequence_str)
    print("="*50)