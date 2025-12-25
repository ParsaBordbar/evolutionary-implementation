from pprint import pprint
from agent import Agent
from configs import MAZE, START, Direction


def main():
    agent = Agent(start=START)
    print(Direction.DOWN)
    agent.move(maze=MAZE, direction=Direction.RIGHT.value)
    agent.move(maze=MAZE, direction=Direction.DOWN.value)
    pprint(f"agent's postion: {agent.x}, {agent.y}, agent's stpes: {agent.steps}, agent's wall hits {agent.wall_hits}")

if __name__ == "__main__":
    main()
