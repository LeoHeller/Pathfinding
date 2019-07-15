import pygame
from pygame.locals import DOUBLEBUF
import numpy as np
from enum import Enum
from typing import List, Optional
import time
import heapq
import random
from timeit import default_timer as timer
from mazelib import Maze as _Maze
from mazelib.generate.Prims import Prims
import matplotlib.pyplot as plt


class Modes(Enum):
    walkable = 1
    obstacle = 2
    start = 3
    end = 4


class Node:
    def __init__(self, x, y):
        self.x = x
        self.y = y

        self.mode = Modes.walkable

        self.penalty = 1

        self.visited_for_maze_generation = False

    def set_mode(self, mode):
        self.mode = mode


class Grid:
    def __init__(self, width, height, node_size, node_type: Node, generate_maze=True):
        self.width = width
        self.height = height
        self.node_size = node_size
        self.node_type = node_type

        self.generate_maze = generate_maze

        self.start_node_pos = (0, 0)
        self.end_node_pos = (0, 0)

        self.nodes: Optional[node_type] = np.empty([int(width / self.node_size), int(height / self.node_size)],
                                                   dtype=node_type)

        # pygame stuff
        pygame.init()
        self.window = pygame.display.set_mode((height, width), DOUBLEBUF)
        self.window.set_alpha(None)
        pygame.display.flip()
        self.draw_grid()
        pygame.display.flip()

    def draw_grid(self):
        # draw the squares
        for x in range(int(self.width / self.node_size)):
            for y in range(int(self.width / self.node_size)):
                self.draw_sq(x, y)
                self.nodes[x, y] = self.node_type(x, y)

        # reset the start position
        self.start_node_pos = (0, 0)
        if self.generate_maze:
            Maze(self)

    def draw_sq(self, x, y, color=(255, 255, 255), spacing=2):
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

        self.maze = self.generate(10, 10)

        w, h = pygame.display.get_surface().get_size()

        visible_space = w // self.grid.node_size, h // self.grid.node_size

        for x in range(int(self.grid.width / self.grid.node_size)):
            for y in range(int(self.grid.height / self.grid.node_size)):
                if not (0 <= x <= visible_space[0]) or not (0 <= y <= visible_space[1]):
                    self.grid.place_obstacle(x, y)
                    continue
                if self.maze.grid[x, y] == 1:
                    self.grid.place_obstacle(x, y)

        grid.place_start(*self.maze.start)
        grid.place_end(*self.maze.end)

    @staticmethod
    def generate(width, height):
        m = _Maze()
        m.generator = Prims(width, height)
        m.generate()
        m.generate_entrances(True, True)
        return m


class AStarNode(Node):
    def __init__(self, x, y):
        super().__init__(x, y)
        self.g_cost = 0
        self.h_cost = 0
        self.f_cost = float("inf")

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
        self.neighbors = self.neighbors if h is None else neighbors

        self.grid: Grid = grid
        self.open: List[AStarNode] = list()
        self.closed: List[AStarNode] = list()
        self.start: Optional[AStarNode] = self.grid.nodes[self.grid.start_node_pos]
        self.end: Optional[AStarNode] = self.grid.nodes[self.grid.end_node_pos]
        self.open.append(self.start)

    def get_path(self):
        path = None
        while path is None:
            path = self.next_step()
            time.sleep(.005)
            pygame.display.flip()
        return path

    @staticmethod
    def manhattan_distance(node1, node2):
        distance = abs(node1.x - node2.x) + abs(node1.y - node2.y)
        return distance

    @staticmethod
    def diagonal_distance(node1, node2):
        # c2 = a2 + b2
        return (node2.x - node1.x) ** 2 + (node2.y - node1.x) ** 2

    @staticmethod
    def neighbors_with_diagonal(node):
        x, y = node.x, node.y
        yield x - 1, y - 1
        yield x, y - 1
        yield x + 1, y - 1
        yield x - 1, y
        yield x + 1, y
        yield x - 1, y + 1
        yield x, y + 1
        yield x + 1, y + 1

    @staticmethod
    def cardinal_neighbors(node):
        x, y = node.x, node.y
        yield x, y - 1
        yield x - 1, y
        yield x + 1, y
        yield x, y + 1

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

        self.closed.append(current)

        # mark as closed
        self.grid.draw_sq(current.x, current.y, color=(176, 16, 21))

        if current == self.end:
            # backtrack to start
            path = list()
            while current is not None:
                path.append(current)
                current = current.parent
            return path[::-1]  # Return reversed path

        children: List[AStarNode] = list()

        for neighbor in self.neighbors(current):
            try:
                visible_space = self.grid.width // self.grid.node_size, self.grid.height // self.grid.node_size
                if self.grid.nodes[neighbor] is not None and (
                        (0 <= neighbor[0] <= visible_space[0]) and (0 <= neighbor[1] <= visible_space[1])):
                    children.append(self.grid.nodes[neighbor])
            except IndexError:
                pass

        for child in children:

            if child in self.closed or child.mode == Modes.obstacle:
                continue

            new_path_cost = current.g_cost + 1
            if (new_path_cost < child.g_cost) or child not in self.open:
                # calculate g, h, f costs
                child.g_cost = current.g_cost + 1
                child.h_cost = AStar.manhattan_distance(child, self.end)
                child.f_cost = child.g_cost + child.h_cost
                child.parent = current

                if child not in self.open:
                    # mark as open
                    self.grid.draw_sq(child.x, child.y, color=(158, 19, 156))
                    heapq.heappush(self.open, child)
