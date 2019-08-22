import threading

import numpy as np
from mazelib import Maze as _Maze
from mazelib.generate.Prims import Prims
import matplotlib.pyplot as plt
from Algos import Modes, Node
from typing import List, Optional
import heapq, time, queue, _queue
from multiprocessing import Pool


class Grid:
    def __init__(self, maze_width, maze_height, node_type: type(Node), slow=False):
        self.node_type = node_type

        self.start_node_pos = (0, 0)
        self.end_node_pos = (0, 0)

        self.set_color = None

        self.slow = slow

        self.nodes: node_type = np.array(
            [[self.node_type(x, y) for y in range(maze_height)] for x in range(maze_width)],
            dtype=node_type
        )

    def reset(self):
        # draw the squares
        for x in range(self.nodes.shape[1]):
            for y in range(self.nodes.shape[0]):
                self.draw_sq(y, x)
                self.nodes[y, x].type = Modes.walkable

    def draw_sq(self, x, y, color=(255, 255, 255), spacing=0):
        self.set_color(x, y, color)

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

    def apply_maze(self, maze):
        pass

    def set_front_end(self, front):
        self.front = front


class Maze:
    def __init__(self, grid_shape, start=(0, 0)):
        self.start = start
        visible_space = grid_shape
        self.maze = self.generate(visible_space[0] // 2, visible_space[1] // 2)

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
        self.in_open = False

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
    def __init__(self, nodes, start_node_pos, end_node_pos, color_setter, h=None, neighbors=None):
        self.h = self.manhattan_distance if h is None else h
        self.neighbors = AStar.cardinal_neighbors if neighbors is None else neighbors
        self.draw_sq = color_setter

        self.nodes: List[AStarNode] = nodes
        self.open: List[AStarNode] = list()
        self.closed: List[AStarNode] = list()

        self.start_node_pos = start_node_pos
        self.end_node_pos = end_node_pos
        self.start: Optional[AStarNode] = self.nodes[tuple(self.start_node_pos)]
        self.end: Optional[AStarNode] = self.nodes[tuple(self.end_node_pos)]

        self.slow = True

        self.should_be_looking = True
        self.path = False

        self.open = queue.PriorityQueue()
        self.start.in_open = True
        self.open.put(self.start)

        # heapq.heappush(self.open, self.start)

    @staticmethod
    def manhattan_distance(node1, node2):
        distance = abs(node1.x - node2.x) + abs(node1.y - node2.y)
        return distance

    @staticmethod
    def diagonal_distance(node1, node2):
        # c2 = a2 + b2
        return (node2.x - node1.x) ** 2 + (node2.y - node1.y) ** 2

    @staticmethod
    def neighbors_with_diagonal(nodes, node):
        try:
            x, y = node.x, node.y
            if x - 1 >= 0:
                yield nodes[x - 1, y]
                yield nodes[x - 1, y + 1]
            if y - 1 >= 0:
                yield nodes[x, y - 1]
                yield nodes[x + 1, y - 1]
            if x - 1 >= 0 and y - 1 >= 0:
                yield nodes[x - 1, y - 1]

            yield nodes[x + 1, y]
            yield nodes[x, y + 1]
            yield nodes[x + 1, y + 1]
        except IndexError:
            pass

    @staticmethod
    def cardinal_neighbors(nodes, node):
        try:
            x, y = node.x, node.y
            if x - 1 >= 0:
                yield nodes[x - 1, y]
            if y - 1 >= 0:
                yield nodes[x, y - 1]

            yield nodes[x + 1, y]
            yield nodes[x, y + 1]
        except IndexError:
            pass

    def next_step(self):
        try:
            # current = heapq.heappop(self.open)
            current = self.open.get(block=True, timeout=4)
            # raise IndexError
        except (IndexError, _queue.Empty):
            print("No path found")
            return False
        if current.mode == Modes.obstacle:
            return

        current.in_closed = True

        # mark as closed (purple)
        self.draw_sq(current.x, current.y, color=(176, 16, 21))

        if current == self.end or (current.x, current.y) == self.end_node_pos:
            # backtrack to start
            path = list()
            while current is not None:
                path.append(current)
                current = current.parent
            return path[::-1]  # Return reversed path

        for neighbor in self.neighbors(self.nodes, current):
            if neighbor.in_closed or neighbor.mode == Modes.obstacle:
                continue

            new_path_cost = current.g_cost + 1
            if (new_path_cost < neighbor.g_cost) or not neighbor.in_open:
                # calculate g, h, f costs
                neighbor.g_cost = current.g_cost + 1
                neighbor.h_cost = self.h(neighbor, self.end)
                neighbor.f_cost = neighbor.g_cost + neighbor.h_cost
                neighbor.parent = current

                if not neighbor.in_open:
                    # mark as open
                    self.draw_sq(neighbor.x, neighbor.y, color=(158, 19, 156))
                    # heapq.heappush(self.open, neighbor)
                    self.open.put(neighbor)
                    neighbor.in_open = True

    def solve(self, num_workers):
        workers = []
        for n in range(num_workers):
            workers.append(Worker(self))
            workers[n].start()


class Worker(threading.Thread):
    def __init__(self, master):
        super().__init__()
        self.daemon = True

        self.master = master

    def run(self):
        while self.master.should_be_looking:
            try:
                if not self.master.open.empty():
                    # current = heapq.heappop(self.master.open)
                    current = self.master.open.get(block=True, timeout=5)
                else:
                    continue

            except (IndexError, _queue.Empty):
                print("No path found")
                self.master.should_be_looking = False
                continue

            if current.mode == Modes.obstacle or isinstance(current, type(None)):
                continue

            current.in_closed = True

            # mark as closed (purple)
            self.master.draw_sq(current.x, current.y, color=(176, 16, 21))

            if current == self.master.end or (current.x, current.y) == self.master.end_node_pos:
                # backtrack to start
                path = list()
                while current is not None:
                    path.append(current)
                    current = current.parent
                self.master.path = path[::-1]  # Return reversed path
                self.master.should_be_looking = False
                continue

            if isinstance(current, type(None)):
                continue

            neighbors = list(self.master.neighbors(self.master.nodes, current))
            for neighbor in neighbors:
                if neighbor.in_closed or neighbor.mode == Modes.obstacle:
                    continue

                new_path_cost = current.g_cost + 1
                if (new_path_cost < neighbor.g_cost) or not neighbor.in_open:
                    # calculate g, h, f costs
                    neighbor.g_cost = current.g_cost + 1
                    neighbor.h_cost = self.master.h(neighbor, self.master.end)
                    neighbor.f_cost = neighbor.g_cost + neighbor.h_cost
                    neighbor.parent = current

                    if not neighbor.in_open:
                        # mark as open
                        self.master.draw_sq(neighbor.x, neighbor.y, color=(158, 19, 156))
                        if self.master.should_be_looking:
                            self.master.open.put(neighbor)
                            neighbor.in_open = True
                            # heapq.heappush(self.master.open, neighbor)
