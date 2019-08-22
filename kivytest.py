from array import array
from itertools import chain

from kivy.app import App
from kivy.clock import Clock
from kivy.core.window import Window
from kivy.graphics import Rectangle
from kivy.graphics.texture import Texture
from kivy.properties import NumericProperty, ListProperty
from kivy.uix.widget import Widget

from grid import *

# from kivy.config import Config
# Config.read("myConfig.ini")
# Config.set('kivy', 'log_level', 'warning')
# Config.set('kivy', 'maxfps', '30')
# Config.write()


SHAPE = (399, 399)


class Dedale(Widget):
    cols = NumericProperty(SHAPE[0])
    rows = NumericProperty(SHAPE[1])
    data = ListProperty([])

    # def __init__(self, grid, maze, **kwargs):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.grid = None
        self.maze = None

        self.solver = SolverThread(self)
        self._mazeGenerationThread = MazeGenerationThread(self, max_look_ahead=10)
        self._mazeGenerationThread.start()

        # self.bottom_left = self.maze.shape
        # self.top_right = (0, 0)

        # Window.bind(mouse_pos=lambda w, p: print(p), on_mouse_down=lambda *args: print(args))
        self._keyboard = Window.request_keyboard(self._keyboard_closed, self)
        self._keyboard.bind(on_key_down=self._on_keyboard_down)

        self._build()
        self.bind(
            cols=self._build,  # to rebuild the texture if you change the size
            rows=self._build,
            pos=self._move_rectangle,  # to move the rect when the widget moves
            size=self._move_rectangle,
        )

        self.get_next_maze()

        self._update_texture()

        # Clock.schedule_interval(self._update_data, 1 / 60)
        Clock.schedule_interval(self._update_texture, 1 / 20)

    def _keyboard_closed(self):
        self._keyboard.unbind(on_key_down=self._on_keyboard_down)
        self._keyboard = None

    def _build(self, *args):
        self._texture = Texture.create(size=(self.cols, self.rows))
        self._texture.mag_filter = 'nearest'

        self.data = []
        for row in range(self.rows):
            r = []
            for col in range(self.cols):
                r.append((255, 255, 255))

            self.data.append(r)

        with self.canvas:
            self._rectangle = Rectangle(
                pos=self.pos,
                size=self.size,
                texture=self._texture
            )

    def _update_texture(self, *dt):
        if dt:
            print(f"\r{1 / dt[0]:.2f}", end="")

        # data = [[(
        #     int(255 * ((col * row) / (self.cols * self.rows))), int(255 * ((col * row) / (self.cols * self.rows))),
        #     int(255 * ((col * row) / (self.cols * self.rows)))) for col in range(self.cols)] for row in
        #     range(self.rows)]

        # data = [[(255, 255, 255) if col % 2 == 0 and row % 2 == 0 else (0, 0, 0) for col in range(self.cols)]
        #     for row in range(self.rows)]

        # self.data = data

        # find changed data:
        # size = (self.top_right[0] - self.bottom_left[0], self.top_right[1] - self.bottom_left[1])
        # pos = self.bottom_left
        # data = self.data
        # data = np.array(data)
        # data[self.bottom_left[0]:self.top_right[0], self.bottom_left[1]:self.top_right[1]]

        arr = list(chain(*chain(*self.data)))
        buf = array('B', arr)
        self._texture.blit_buffer(buf, colorfmt='rgb', bufferfmt='ubyte')  # , size=size, pos=pos)

        self.canvas.ask_update()

    def _on_keyboard_down(self, keyboard, keycode, text, modifiers):
        # print(f"keycode: '{keycode}', text: '{text}', modifiers: '{modifiers}'")
        if keycode[0] == 32:
            if not self.solver.started:
                print("solving")
                self.solver.start()
            else:
                self.solver.should_skip = not self.solver.should_skip
        elif keycode[0] == ord("r"):
            if not self.solver.started:
                self.get_next_maze()
                self.solver = SolverThread(self)

    def on_touch_down(self, touch):
        if super().on_touch_down(touch):
            return True
        if self.collide_point(*touch.pos):
            # color used to mark what cell was clicked on
            marker_color = (0, 255, 255)
            # (0-1, 0-1) percentage from lower left corner (1,1)
            rel_pos = touch.spos
            cell_hit = (int(self.rows * rel_pos[1]), int(self.cols * rel_pos[0]))
            # make sure cell hasn't already been marked
            if self.data[cell_hit[0]][cell_hit[1]] != marker_color:
                # print Node info:
                selected_node: Optional[AStarNode, Node] = self.grid.nodes[cell_hit[0]][cell_hit[1]]
                print(f"\n-==Node Info==-")
                print(f" - type: {type(selected_node)}")
                print(f" - pos")
                print(f"   - visual: {cell_hit}")
                print(f"   - node:   ({selected_node.x}, {selected_node.y})")
                print(f" - mode: {selected_node.mode}")
                print(f" - colour: {tuple(self.data[cell_hit[0]][cell_hit[1]])}")

                # highlight it
                original_cell_color = self.data[cell_hit[0]][cell_hit[1]]
                self.set_color(*cell_hit, marker_color)
                # un-highlight it after 0.5s
                Clock.schedule_once(lambda x: self.set_color(*cell_hit, tuple(original_cell_color)), 0.5)

    def _move_rectangle(self, *args):
        self._rectangle.pos = self.pos
        self._rectangle.size = self.size

    def set_color(self, x, y, color=(255, 255, 255)):
        # if (x, y) > self.top_right:
        #     self.top_right = (x, y)
        # if (x, y) < self.bottom_left:
        #     self.bottom_left = (x, y)

        self.data[x][y] = color

    def get_next_maze(self):
        next_maze_and_grid = self._mazeGenerationThread.maze_q.get(block=True)
        self.apply_maze_and_grid(*next_maze_and_grid)

    def apply_maze_and_grid(self, grid, maze):
        self.grid = grid
        self.maze = maze
        # set grid nodes to maze values
        self.cols = int(self.maze.shape[0])
        self.rows = int(self.maze.shape[1])

        self.grid.set_color = self.set_color

        [[self.grid.place_obstacle(x, y) if self.maze[x][y] else self.grid.place_walkable(x, y) for x in
          range(self.maze.shape[0])] for y in range(self.maze.shape[1])]

        self.grid.place_start(*self.grid.start_node_pos)
        self.grid.place_end(*self.grid.end_node_pos)

        self._update_texture()

    def set_maze(self):
        self.cols = int(self.maze.shape[0])
        self.rows = int(self.maze.shape[1])

        self.grid.set_color = self.set_color
        [[self.grid.place_obstacle(x, y) if self.maze[x][y] else self.grid.place_walkable(x, y) for x in
          range(self.maze.shape[0])] for y in range(self.maze.shape[1])]

        self.grid.place_start(*self.grid.start_node_pos)
        self.grid.place_end(*self.grid.end_node_pos)
        self._update_texture()


class SolverThread(threading.Thread):
    def __init__(self, dedale):
        super(SolverThread, self).__init__()
        self.daemon = True
        self.dedale = dedale

        self.started = False
        self.should_skip = False

        self.solver = None

    def solve_maze(self) -> None:
        self.solver = AStar(self.dedale.grid.nodes, self.dedale.grid.start_node_pos, self.dedale.grid.end_node_pos,
                            self.dedale.set_color)  # ,
        # h=AStar.diagonal_distance, neighbors=AStar.neighbors_with_diagonal)
        path, distance = self.get_path(self.solver)
        if path is False:
            print("i can't solve that!!")
            return
        for node in path:
            if not self.started:
                break
            else:
                self.dedale.set_color(node.x, node.y, color=(19, 151, 158))
                if not self.should_skip:
                    time.sleep(.01)

        print(f"took me {distance} steps")

    def get_path(self, solver):
        path = None
        c = 0
        start = time.time()
        # solver.solve(3)
        #
        # while solver.path is False:
        #     time.sleep(.01)
        #
        # print(f"took me {time.time() - start}")
        # solver.should_be_looking = False
        # if type(solver.path) == bool:
        #     return solver.path, -1
        # return solver.path, len(solver.path)

        while path is None:
            path = solver.next_step()
        print(f"took me {time.time() - start}")
        # print(solver.path)

        if type(path) == bool:
            return path, -1
        return path, len(path)

    def run(self) -> None:
        self.started = True
        self.solve_maze()
        self.started = False


class MazeGenerationThread(threading.Thread):
    def __init__(self, parent, max_look_ahead=4):
        super().__init__(daemon=True)
        self.parent = parent
        self.maze_q = queue.Queue(maxsize=max_look_ahead)

    def run(self):
        while True:
            new_grid = Grid(*SHAPE, AStarNode)
            new_maze = Maze(SHAPE).maze

            new_grid.start_node_pos = (new_maze.start[0], new_maze.start[1])
            new_grid.end_node_pos = (new_maze.end[0], new_maze.end[1])

            new_maze = np.array(new_maze.grid, dtype=bool)

            self.maze_q.put(
                (
                    new_grid,
                    new_maze
                ),
                block=True
            )
            print(f"now caching {self.maze_q.qsize()}")


class MyApp(App):
    def __init__(self):
        super().__init__()
        self.shape = (199, 199)
        self.maze = None
        self.grid = None

    def build(self):
        # print("started generating grid")
        # start = time.time()
        # self.grid = Grid(*self.shape, AStarNode)
        #
        # print(f"finished grid after: {time.time() - start}")
        # start = time.time()
        # print("started maze_gen")
        #
        # self.maze = Maze(self.shape).maze
        #
        # print(f"finished maze gen after {time.time() - start}")
        #
        # self.grid.start_node_pos = (self.maze.start[0], self.maze.start[1])
        # self.grid.end_node_pos = (self.maze.end[0], self.maze.end[1])

        # maze_grid = np.array(self.maze.grid, dtype=bool)
        # self.dedale = Dedale(self.grid, maze_grid)
        self.dedale = Dedale()
        return self.dedale


if __name__ == '__main__':
    myApp = MyApp()
    myApp.run()

# timing at 199x199 20fps, no-wait, cardial

# normal
# .29
# .29
# .63
# .33
# .24
# .18
# ==> .327

# threaded_neighbors
# 4.375
# 4.371
# 2.54
# 2.80
# 1.6
# 1
# 2.6
# 5
# 3.94
# 4.7
# ==> 3.29

# diagonal
# 0.9
# 0.1
# 0.1
# 0.3
# 0.05
# 0.06
