import matplotlib.pyplot as plt
import numpy as np

def visualize_maze(maze, start, goal):
    grid = np.array(maze)

    plt.figure(figsize=(6, 6))
    plt.imshow(grid, cmap="gray_r")


    plt.scatter(start[0], start[1], c="green", s=100, label="Start")
    plt.scatter(goal[0], goal[1], c="red", s=100, label="Goal")

    plt.title("Evolved Agent Path")
    plt.legend()
    plt.grid(False)
    plt.show()