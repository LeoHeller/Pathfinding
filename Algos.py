import pygame
from pygame.locals import DOUBLEBUF
import numpy as np
from enum import Enum
from typing import List, Optional


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

    def set_mode(self, mode):
        self.mode = mode


class Grid:
    def __init__(self, width, height, node_size, node_type):
        self.width = width
        self.height = height
        self.node_size = node_size
        self.node_type = node_type

        self.start_node_pos = (0, 0)
        self.end_node_pos = (0, 0)

        self.nodes = np.empty([width, height], dtype=node_type)

        # pygame stuff
        pygame.init()
        self.window = pygame.display.set_mode((height, width), DOUBLEBUF)
        self.window.set_alpha(None)
        pygame.display.flip()
        self.draw_grid()
        pygame.display.flip()

    def draw_grid(self):
        # draw the squares
        for x in range(int(self.width / self.node_size) + 1):
            for y in range(int(self.width / self.node_size) + 1):
                self.draw_sq(x, y)
                self.nodes[x][y] = (self.node_type(x, y))

        # reset the start position
        self.start_node_pos = (0, 0)

    def draw_sq(self, x, y, color=(255, 255, 255), spacing=2):
        rect = pygame.Rect(x * self.node_size + spacing, y * self.node_size + spacing, self.node_size - spacing,
                           self.node_size - spacing)
        pygame.draw.rect(self.window, color, rect)


class AStarNode(Node):
    def __init__(self, x, y):
        super().__init__(x, y)
        self.g_cost = 0
        self.h_cost = 0
        self.f_cost = 0

        self.parent: Optional[AStarNode] = None

    def __lt__(self, other):
        return self.f_cost < other.f_cost

    def __gt__(self, other):
        return self.f_cost > other.f_cost


class AStar:
    def __init__(self, grid):
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
        return path

    @staticmethod
    def manhattan_distance(node1, node2):
        distance = abs(node1.x - node2.x) + abs(node1.y - node2.y)
        return distance

    def recalculate_node(self, node):
        g_cost = AStar.manhattan_distance(node, self.start)
        h_cost = AStar.manhattan_distance(node, self.end)
        f_cost = (g_cost + h_cost) * node.penalty

        node.g_cost = g_cost
        node.h_cost = h_cost
        node.f_cost = f_cost

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
    def neighbors(node):
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
        current = self.open[0]
        min_f_costs = AStar.all_min(self.open)
        for node in min_f_costs:
            if node.h_cost < current.h_cost:
                current = node

        self.open.remove(current)
        self.closed.append(current)

        if current == self.end:
            # backtrack to start
            path = list()
            while current is not None:
                path.append(current)
                current = current.parent
            return path[::-1]  # Return reversed path

        children: List[AStarNode] = list()

        for neighbor in AStar.neighbors(current):
            try:
                if self.grid.nodes[neighbor] is not None:
                    children.append(self.grid.nodes[neighbor])
            except IndexError:
                pass

        for child in children:

            if child in self.closed or child.mode == Modes.obstacle:
                continue

            if child.mode == Modes.obstacle:
                print("uhhhh")
            new_path_cost = current.g_cost + 1
            if (new_path_cost < child.g_cost) or child not in self.open:
                # calculate g, h, f costs
                child.g_cost = current.g_cost + 1
                child.h_cost = AStar.manhattan_distance(child, self.end)
                child.f_cost = child.g_cost + child.h_cost
                child.parent = current

                if child not in self.open:
                    self.open.append(child)
