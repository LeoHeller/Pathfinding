from Algos import *


class Viz:
    def __init__(self):
        self.grid = Grid(300, 300, 50, AStarNode)
        self.a_star = AStar(self.grid)

        self.clock = pygame.time.Clock()
        self.running = True
        self.special_mode = True

        while self.running:
            self.update()

    def handle_keyboard(self):
        # keyboard input
        events = pygame.event.get()
        for event in events:
            if event.type == pygame.KEYDOWN:
                if event.key in [27, 113]:
                    self.running = False
                elif event.key == ord("r"):
                    self.grid.draw_grid()
                elif event.key == ord("s"):
                    self.special_mode = not self.special_mode

    def place_obstacle(self, x, y):
        self.grid.draw_sq(x, y, color=(35, 0, 55))
        self.grid.nodes[x, y].set_mode(Modes.obstacle)

    def place_walkable(self, x, y):
        self.grid.draw_sq(x, y, color=(255, 255, 255))
        self.grid.nodes[x, y].set_mode(Modes.walkable)

    def place_start(self, x, y):
        if self.grid.nodes[self.grid.start_node_pos].mode == Modes.start:
            self.grid.nodes[self.grid.start_node_pos].set_mode(Modes.walkable)
            self.grid.draw_sq(self.grid.start_node_pos[0], self.grid.start_node_pos[1], color=(255, 255, 255))

        self.grid.start_node_pos = (x, y)
        self.grid.draw_sq(x, y, color=(65, 245, 70))
        self.grid.nodes[x, y].set_mode(Modes.start)

    def place_end(self, x, y):
        if self.grid.nodes[self.grid.end_node_pos].mode == Modes.end:
            self.grid.nodes[self.grid.end_node_pos].set_mode(Modes.walkable)
            self.grid.draw_sq(self.grid.end_node_pos[0], self.grid.end_node_pos[1], color=(255, 255, 255))

        self.grid.end_node_pos = (x, y)
        self.grid.draw_sq(x, y, color=(14, 14, 153))
        self.grid.nodes[x, y].set_mode(Modes.end)

    def handle_mouse(self):
        # mouse input
        pos = pygame.mouse.get_pos()
        pressed1, pressed2, pressed3 = pygame.mouse.get_pressed()
        x, y = int(pos[0] / self.grid.node_size), int(pos[1] / self.grid.node_size)

        if pressed1:
            self.place_obstacle(x, y)
        elif pressed3:
            self.place_walkable(x, y)
        elif pressed2:
            if self.special_mode is True:
                self.place_start(x, y)
            else:
                self.place_end(x, y)

    def update(self):
        self.handle_keyboard()
        self.handle_mouse()

        pygame.display.flip()
        self.clock.tick(60)


Viz()

pygame.quit()