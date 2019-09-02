# -*- coding: utf-8 -*-
from random import choice
import networkx as nx
from MazeGeneration.setch import Setch  # set with random choice
import numpy as np
import imageio

DIM = [60, 40]


def gen_maze():
    """
    Generates a maze by iteratively adding to a tree.
    """
    G = nx.grid_graph(DIM)
    tree = nx.Graph()
    tree.add_node(choice(list(G)))
    neighbors = Setch(*G.neighbors(*tree.nodes()))
    while tree.order() < G.order():
        new_node = neighbors.choose()
        neighbors.remove(new_node)
        nodes_in_tree, new_neighbors = [], []
        for node in G.neighbors(new_node):
            (nodes_in_tree if node in tree else new_neighbors).append(node)
        tree.add_edge(new_node, choice(nodes_in_tree))
        neighbors += new_neighbors
    return tree


def gen_maze_longer_paths():
    """
    Similar to gen_maze, but we only look at neighbors of the last node added
    to T unless the last node added to T has no neighbors in G, then we try a
    new node in T. This should only be a tiny bit slower than gen_maze, but
    should produce much nicer mazes.
    """
    G = nx.grid_graph(DIM)
    tree = nx.Graph()
    old_node = choice(list(G))
    tree.add_node(old_node)
    all_neighbors = Setch(*G.neighbors(old_node))
    while tree.order() < G.order():
        neighbors = [node for node in G.neighbors(old_node) \
                     if node not in tree]
        try:
            new_node = choice(neighbors)
            neighbors.remove(new_node)
        except IndexError:  # Dead-end
            new_node = all_neighbors.choose()
            nodes_in_tree, neighbors = [], []
            for node in G.neighbors(new_node):
                (nodes_in_tree if node in tree else neighbors).append(node)
            old_node = choice(nodes_in_tree)
        all_neighbors.remove(new_node)
        tree.add_edge(old_node, new_node)
        all_neighbors += neighbors
        old_node = new_node
    return tree


def gen_maze_min_spanning_tree(algorithm='kruskal'):
    """
    Generates a maze by randomly assigning weights to a grid graph and then
    finding minimum spanning tree.

    Algorithm options are 'kruskal', 'prim', or 'boruvka'.
    """
    G = nx.grid_graph(DIM)
    for (u, v) in G.edges():
        G.edges[u, v]['weight'] = np.random.random()  # Change distributions?
    return nx.algorithms.minimum_spanning_tree(G, weight='weight', \
                                               algorithm=algorithm)


def maze_to_array(maze, size=1, invert=False, \
                  start="top_left", end="bottom_right"):
    """
    Cells in the maze will be of dimension size * size.

    'Invert' switches the wall and cell values.

    'start'/'end' parameters can be one of:
        'top_left', 'top_right',
        'bottom_left', 'bottom_right'
    """
    # Check that position options are valid.
    pos_options = ["top_left", "top_right", "bottom_left", "bottom_right"]
    if not all([pos in pos_options for pos in [start, end]]):
        raise ValueError("'start, end' arguments invalid.")

    maze_copy = maze.copy()  # copy, so we don't ruin the maze as we remove edges
    nodes = list(maze_copy)
    cell = np.full([size, size], 0 if invert else 1, dtype=int)
    maze_array = np.full([size * (2 * DIM[1] + 1), size * (2 * DIM[0] + 1)], \
                         1 if invert else 0, dtype=int)
    for node in nodes:
        node_x, node_y = size * (2 * node[0] + 1), size * (2 * node[1] + 1)
        maze_array[node_x: node_x + size, node_y: node_y + size] = cell
        for neighbor in list(maze_copy.neighbors(node)):
            path_x, path_y = size * (node[0] + neighbor[0] + 1), \
                             size * (node[1] + neighbor[1] + 1)
            maze_array[path_x: path_x + size, path_y: path_y + size] = cell
            maze_copy.remove_edge(node, neighbor)  # Don't add redundant cells

    # Create start and finish cells
    positions = {"top": slice(size, 2 * size),
                 "bottom": slice(size * (2 * DIM[1] - 1), size * (2 * DIM[1])),
                 "left": slice(0, size),
                 "right": slice(size * (2 * DIM[0]), size * (2 * DIM[0] + 1))
                 }
    # Parse
    y_start, x_start, y_end, x_end = start.split("_") + end.split("_")
    y_start, x_start = positions[y_start], positions[x_start]
    y_end, x_end = positions[y_end], positions[x_end]
    # Broadcast cells
    maze_array[y_start, x_start] = cell
    maze_array[y_end, x_end] = cell

    return maze_array


def array_to_image(maze_array, name=f'maze{DIM[0]}x{DIM[1]}'):
    maze_array *= 255
    imageio.imwrite(name + ".png", maze_array.astype(np.uint8), format='png')


if __name__ == "__main__":
    # maze = gen_maze()
    maze = gen_maze_longer_paths()
    # maze = gen_maze_min_spanning_tree()
    maze_array = maze_to_array(maze, 1)
    #array_to_image(maze_array)
