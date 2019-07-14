import pygame, time, sys
from pygame.locals import *
import numpy as np
from itertools import product, starmap, islice


class GameOfLife():
    def __init__(self, blocksize, max_gen, width=480, height=680):
        pygame.init()
        self.blocksize = blocksize

        self.window = pygame.display.set_mode((height, width), DOUBLEBUF)
        self.window.set_alpha(None)

        pygame.display.flip()
        self.w, self.h = pygame.display.get_surface().get_size()
        self.field = np.array(
            [[0 for j in range(int(self.w / self.blocksize))] for i in range(int(self.h / self.blocksize))])
        #self.next_field = np.array(
        #    [[0 for j in range(int(self.w / self.blocksize))] for i in range(int(self.h / self.blocksize))])

        self.board = set([])

        self.draw_grid()
        pygame.display.flip()

    def draw_sq(self, x, y, color=(255, 255, 255), spacing=2):
        rect = pygame.Rect(x * self.blocksize, y * self.blocksize, self.blocksize - spacing, self.blocksize - spacing)
        pygame.draw.rect(self.window, color, rect)

    def draw_grid(self):
        for y in range(int(self.w / self.blocksize)+1):
            for x in range(int(self.h / self.blocksize)+1):
                self.draw_sq(y, x)

    @staticmethod
    def neighbors(cell):
        x, y = cell
        yield x - 1, y - 1
        yield x, y - 1
        yield x + 1, y - 1
        yield x - 1, y
        yield x + 1, y
        yield x - 1, y + 1
        yield x, y + 1
        yield x + 1, y + 1

    def apply_iteration(self, board):
        new_board = set([])
        candidates = self.board.union(set(n for cell in self.board for n in self.neighbors(cell)))
        for cell in candidates:
            count = sum((n in self.board) for n in self.neighbors(cell))
            if count == 3 or (count == 2 and cell in self.board):
                new_board.add(cell)
        return new_board

    def update(self):
        for x, y in self.board:
            game.draw_sq(x, y, color=(255, 255, 255))
        self.board = self.apply_iteration(self.board)

        for x, y in self.board:
            game.draw_sq(x, y, color=(255, 0, 0))

    def limit(self, value, minV, maxV):
        if value < minV:
            value = minV
        elif value > maxV:
            value = maxV
        return value


running = True
paused = True
tickrate = 1
game = GameOfLife(25, 500)
clock = pygame.time.Clock()

while running:

    events = pygame.event.get()
    for event in events:
        if event.type == pygame.KEYDOWN:
            if event.key == 32:
                paused = not paused
            elif event.key == 273:
                tickrate += 10
            elif event.key == 274:
                tickrate -= 10
            elif event.key in [27, 113]:
                running = False

    tickrate = game.limit(tickrate, 1, 120)
    if paused:
        pos = pygame.mouse.get_pos()
        pressed1, pressed2, pressed3 = pygame.mouse.get_pressed()
        if pressed1:
            x, y = int(pos[0] / game.blocksize), int(pos[1] / game.blocksize)
            game.draw_sq(x, y, color=(255, 0, 0))
            game.board.add((x, y))
        elif pressed3:
            x, y = int(pos[0] / game.blocksize), int(pos[1] / game.blocksize)
            game.draw_sq(x, y, color=(255, 255, 255))
            try:
                game.board.remove((x, y))
            except Exception:
                pass

    if not paused:
        game.update()
        pygame.display.flip()
        clock.tick(tickrate)

    else:
        pygame.display.flip()
        clock.tick(60)

print("done")
pygame.quit()
