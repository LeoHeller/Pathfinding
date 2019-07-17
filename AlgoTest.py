from Algos import *


class Viz:
    def __init__(self):
        self.grid = Grid(39, 39, 25, node_type=AStarNode, generate_maze=True)

        pygame.display.flip()
        self.clock = pygame.time.Clock()
        self.running = True
        self.special_mode = True
        self.debug = False

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
                elif event.key == ord("d"):
                    self.debug = not self.debug
                elif event.key == ord(" "):
                    self.a_star = AStar(self.grid, AStar.manhattan_distance, AStar.cardinal_neighbors)
                    path, distance = self.a_star.get_path()
                    if path is False:
                        print("i cant solve that!!")
                        return
                    for node in path:
                        self.grid.draw_sq(node.x, node.y, color=(19, 151, 158))
                    print(f"took me {distance} steps")

    def handle_mouse(self):
        # mouse input
        pos = pygame.mouse.get_pos()
        pressed1, pressed2, pressed3 = pygame.mouse.get_pressed()
        x, y = int(pos[0] / self.grid.node_size), int(pos[1] / self.grid.node_size)

        if pressed1:
            if not self.debug:
                self.grid.place_obstacle(x, y)
            else:
                print(self.grid.nodes[x, y].mode)
                print(f"x: {x}, y: {y}")
                print(f"f-cost: {self.grid.nodes[x, y].f_cost}, g-cost: {self.grid.nodes[x, y].g_cost}, h-cost: {self.grid.nodes[x, y].h_cost}")
        elif pressed3:
            self.grid.place_walkable(x, y)
        elif pressed2:
            if self.special_mode is True:
                self.grid.place_start(x, y)
            else:
                self.grid.place_end(x, y)

    def update(self):
        self.handle_keyboard()
        self.handle_mouse()

        pygame.display.flip()
        self.clock.tick(60)


Viz()

pygame.quit()
