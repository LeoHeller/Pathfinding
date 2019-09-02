from random import randint

from array import array
from itertools import chain

import time

from kivy.app import App
from kivy.clock import Clock
from kivy.core.window import Window
from kivy.graphics import Rectangle
from kivy.graphics.texture import Texture
from kivy.properties import NumericProperty, ListProperty
from kivy.uix.widget import Widget


class Dedale(Widget):
    cols = NumericProperty(25)
    rows = NumericProperty(25)
    data = ListProperty([])

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self._keyboard = Window.request_keyboard(self._keyboard_closed, self)
        self._keyboard.bind(on_key_down=self._on_keyboard_down)

        self._build()
        self.bind(
            cols=self._build,  # to rebuild the texture if you change the size
            rows=self._build,
            pos=self._move_rectangle,  # to move the rect when the widget moves
            size=self._move_rectangle,
        )

        self._update_texture()

        Clock.schedule_interval(self._update_data, 1 / 60)
        Clock.schedule_interval(self._update_texture, 1 / 60)

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

    def _update_data(self, *dt):
        data = []
        for row in range(self.rows):
            r = []
            for col in range(self.cols):
                r.append(
                    (
                        randint(0, 255),
                        randint(0, 255),
                        randint(0, 255)
                    )
                )
            data.append(r)

        self.data = data

    def _update_texture(self, *dt):
        if dt:
            print(f"\r{1 / dt[0]:.2f}", end="")

        arr = list(chain(*chain(*self.data)))
        buf = array('B', arr)
        self._texture.blit_buffer(buf, colorfmt='rgb', bufferfmt='ubyte')  # , size=size, pos=pos)

        self.canvas.ask_update()

    def _on_keyboard_down(self, keyboard, keycode, text, modifiers):
        print(f"keycode: '{keycode}', text: '{text}', modifiers: '{modifiers}'")
        if keycode[0] == 32:
            print("spacebar pressed")

    def on_touch_down(self, touch):
        if super().on_touch_down(touch):
            return True
        if self.collide_point(*touch.pos):
            print(touch.pos)

    def _move_rectangle(self, *args):
        self._rectangle.pos = self.pos
        self._rectangle.size = self.size


class MyApp(App):
    def build(self):
        return Dedale()


if __name__ == "__main__":
    myApp = MyApp()
    myApp.run()
