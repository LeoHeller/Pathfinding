import heapq
import time
from enum import Enum
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pygame
from mazelib import Maze as _Maze
from mazelib.generate.Prims import Prims


class Modes(Enum):
    walkable = 1
    obstacle = 2
    start = 3
    end = 4


class Node:
    def __init__(self, x, y, mode: Modes = Modes.walkable):
        self.x = x
        self.y = y

        self.mode = mode

        self.penalty = 1

        self.visited_for_maze_generation = False

    def set_mode(self, mode):
        self.mode = mode


class Grid:
    def __init__(self, maze_width, maze_height, node_size, node_type: Node, generate_maze=True, slow=False):

        self.height = int(maze_height * node_size)
        self.width = int(maze_width * node_size)
        self.node_size = node_size
        self.node_type = node_type
        self.generate_maze = generate_maze

        self.start_node_pos = (0, 0)
        self.end_node_pos = (0, 0)

        self.slow = slow
        # pygame stuff

        # os.environ['SDL_VIDEO_WINDOW_POS'] = "0,0"

        pygame.init()
        pygame.display.set_caption('aMAZEing')
        self.window = pygame.display.set_mode((self.width, self.height))

        self.nodes: Optional[node_type] = np.empty(
            [maze_width, maze_height],
            dtype=node_type)
        print(self.nodes.shape)

        self.window.set_alpha(None)
        pygame.display.flip()
        self.draw_grid()
        pygame.display.flip()

    def reset(self):
        self.nodes = [[self.node_type(x, y) for y in range(self.nodes.shape[0])] for x in range(self.nodes.shape[1])]
        if self.generate_maze:
            Maze(self)

    def draw_grid(self):
        # draw the squares
        self.nodes = [[self.node_type(x, y) for y in range(self.nodes.shape[0])] for x in range(self.nodes.shape[1])]
        # reset the start position
        self.start_node_pos = (0, 0)

    def draw_sq(self, x, y, color=(255, 255, 255), spacing=0):
        rect = pygame.Rect(x * self.node_size + spacing, y * self.node_size + spacing, self.node_size - spacing,
                           self.node_size - spacing)
        pygame.draw.rect(self.window, color, rect)

    def place_obstacle(self, x, y):
        self.draw_sq(x, y, color=(35, 0, 55))
        self.nodes[x, y].set_mode(Modes.obstacle)

    def place_walkable(self, x, y):
        self.draw_sq(x, y, color=(255, 255, 255))
        self.nodes[x, y].set_mode(Modes.walkable)

    def place_start(self, x, y):
        if self.nodes[self.start_node_pos].mode == Modes.start:
            self.nodes[self.start_node_pos].set_mode(Modes.walkable)
            self.draw_sq(self.start_node_pos[0], self.start_node_pos[1], color=(255, 255, 255))

        self.start_node_pos = (x, y)
        self.draw_sq(x, y, color=(65, 245, 70))
        self.nodes[x, y].set_mode(Modes.start)

    def place_end(self, x, y):
        if self.nodes[self.end_node_pos].mode == Modes.end:
            self.nodes[self.end_node_pos].set_mode(Modes.walkable)
            self.draw_sq(self.end_node_pos[0], self.end_node_pos[1], color=(255, 255, 255))

        self.end_node_pos = (x, y)
        self.draw_sq(x, y, color=(14, 14, 153))
        self.nodes[x, y].set_mode(Modes.end)


class Maze:
    def __init__(self, grid: Grid, start=(0, 0)):
        self.start = start
        self.grid = grid

        visible_space = self.grid.nodes.shape
        print(visible_space)
        self.maze = self.generate(visible_space[0] // 2, visible_space[1] // 2)

        for x in range(visible_space[0]):
            for y in range(visible_space[1]):
                if not (0 <= x <= visible_space[0]) or not (0 <= y <= visible_space[1]):
                    self.grid.place_obstacle(x, y)
                    continue
                if self.maze.grid[x, y] == 1:
                    self.grid.place_obstacle(x, y)

        grid.place_start(self.maze.start[0], self.maze.start[1])
        grid.place_end(self.maze.end[0], self.maze.end[1])

    @staticmethod
    def generate(width, height):
        m = _Maze()
        m.generator = Prims(width, height)
        m.generate()
        m.generate_entrances(True, True)
        return m

    @staticmethod
    def showPNG(grid):
        """Generate a simple image of the maze."""
        plt.figure(figsize=(10, 5))
        plt.imshow(grid, cmap=plt.cm.binary, interpolation='nearest')
        plt.xticks([]), plt.yticks([])
        plt.show()


class AStarNode(Node):
    def __init__(self, x, y, mode: Modes = Modes.walkable):
        super().__init__(x, y, mode)
        self.g_cost = 0
        self.h_cost = 0
        self.f_cost = float("inf")

        self.in_closed = False

        self.parent: Optional[AStarNode] = None

    def __lt__(self, other):  # self < other
        if self.f_cost == other.f_cost:
            return self.h_cost > other.h_cost
        return self.f_cost < other.f_cost

    def __gt__(self, other):  # self > other
        if self.f_cost == other.f_cost:
            return self.h_cost < other.h_cost
        return self.f_cost > other.f_cost

    def __eq__(self, other):  # self == other
        return self.x == other.x and self.y == other.y


class AStar:
    def __init__(self, grid, h=None, neighbors=None):
        self.h = self.manhattan_distance if h is None else h
        self.neighbors = self.cardinal_neighbors if neighbors is None else neighbors

        self.grid: Grid = grid
        self.open: List[AStarNode] = list()
        self.closed: List[AStarNode] = list()
        self.start: Optional[AStarNode] = self.grid.nodes[self.grid.start_node_pos]
        self.end: Optional[AStarNode] = self.grid.nodes[self.grid.end_node_pos]
        heapq.heappush(self.open, self.start)

    def get_path(self):
        path = None
        c = 0
        while path is None:
            path = self.next_step()
            if self.grid.slow:
                time.sleep(.001)
                pygame.display.flip()
            else:
                if c % 500 == 0:
                    pygame.display.flip()
                c += 1
        if type(path) == bool:
            return path, -1
        return path, len(path)

    @staticmethod
    def manhattan_distance(node1, node2):
        distance = abs(node1.x - node2.x) + abs(node1.y - node2.y)
        return distance

    @staticmethod
    def diagonal_distance(node1, node2):
        # c2 = a2 + b2
        return (node2.x - node1.x) ** 2 + (node2.y - node1.x) ** 2

    def neighbors_with_diagonal(self, node):
        x, y = node.x, node.y

        if x - 1 >= 0:
            yield self.grid.nodes[x - 1, y]
            yield self.grid.nodes[x - 1, y + 1]
        if y - 1 >= 0:
            yield self.grid.nodes[x, y - 1]
            yield self.grid.nodes[x + 1, y - 1]
        if x - 1 >= 0 and y - 1 >= 0:
            yield self.grid.nodes[x - 1, y - 1]

        yield self.grid.nodes[x + 1, y]
        yield self.grid.nodes[x, y + 1]
        yield self.grid.nodes[x + 1, y + 1]

    def cardinal_neighbors(self, node):
        try:
            x, y = node.x, node.y
            if x - 1 >= 0:
                yield self.grid.nodes[x - 1, y]
            if y - 1 >= 0:
                yield self.grid.nodes[x, y - 1]

            yield self.grid.nodes[x + 1, y]
            yield self.grid.nodes[x, y + 1]
        except IndexError:
            pass

    @staticmethod
    def all_min(a: list) -> list:
        a_min = min(a)
        return [i for i in a if i == a_min]

    def next_step(self):
        try:
            current = heapq.heappop(self.open)
        except IndexError:
            print("No path found")
            return False

        current.in_closed = True

        # mark as closed
        self.grid.draw_sq(current.x, current.y, color=(176, 16, 21))

        if current == self.end:
            # backtrack to start
            path = list()
            while current is not None:
                path.append(current)
                current = current.parent
            return path[::-1]  # Return reversed path

        for neighbor in self.cardinal_neighbors(current):
            if neighbor.in_closed or neighbor.mode == Modes.obstacle:
                continue

            new_path_cost = current.g_cost + 1
            if (new_path_cost < neighbor.g_cost) or neighbor not in self.open:
                # calculate g, h, f costs
                neighbor.g_cost = current.g_cost + 1
                neighbor.h_cost = AStar.manhattan_distance(neighbor, self.end)
                neighbor.f_cost = neighbor.g_cost + neighbor.h_cost
                neighbor.parent = current

                if neighbor not in self.open:
                    # mark as open
                    self.grid.draw_sq(neighbor.x, neighbor.y, color=(158, 19, 156))
                    heapq.heappush(self.open, neighbor)
