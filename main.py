# Main Script which initializes the app
from kivy.app import App
from kivy.uix.label import Label
from kivy.uix.camera import Camera
from kivy.uix.button import Button
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.image import Image
from kivy.clock import Clock
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.lang import Builder
import numpy as np
import time
from detect_objects import *
from export_data import *


class CameraScreen(Screen):

    def __init__(self, **kwargs):
        super(CameraScreen, self).__init__(**kwargs)
        layout = BoxLayout(orientation='vertical')

        # Camera object
        self.camera = Camera()
        self.camera.resolution = (720, 1280)
        self.camera.play = True

        # Take photo button object
        button = Button(text='Take Photo')
        button.size_hint = (.5, .2)
        button.pos_hint = {'x': .25, 'y': .75}
        button.bind(on_press=self.take_photo)

        # Add to layout
        layout.add_widget(self.camera)
        layout.add_widget(button)
        self.add_widget(layout)

    def take_photo(self, *args):
        self.camera.export_to_png('Coffeelist.png')
        self.manager.current = 'confirm'


class ConfirmScreen(Screen):

    def __init__(self, **kwargs):
        super(ConfirmScreen, self).__init__(**kwargs)
        layout = BoxLayout(orientation='vertical')
        button_layout = BoxLayout(orientation='horizontal')

        # Image object
        self.image = Image(source='Coffeelist.png')
        Clock.schedule_interval(lambda dt: self.image.reload(), 0.2)

        # Confirm button object
        confirm_btn = Button(text='Confirm')
        confirm_btn.size_hint = (.25, .2)
        confirm_btn.pos_hint = {'x': .25, 'y': .75}
        confirm_btn.bind(on_press=self.switch_photo_screen)

        # Cancel button object
        cancel_btn = Button(text='Cancel')
        cancel_btn.size_hint = (.25, .2)
        cancel_btn.pos_hint = {'x': .25, 'y': .75}
        cancel_btn.bind(on_press=self.switch_photo_screen)

        # Add to layout
        layout.add_widget(self.image)
        button_layout.add_widget(confirm_btn)
        button_layout.add_widget(cancel_btn)
        layout.add_widget(button_layout)
        self.add_widget(layout)

    def switch_photo_screen(self, *args):
        self.manager.current = 'camera'


class MainApp(App):
    def build(self):
        sm = ScreenManager()
        sm.add_widget(CameraScreen(name='camera'))
        sm.add_widget(ConfirmScreen(name='confirm'))

        return sm


# Initialize App
if __name__ == '__main__':
    img = 'Tensorflow/workspace/images/collectedimages/tables/CL_DATA_060822 (1).jpeg'
    MainApp().run()

    #     objects = detect_strokes(img)
    #     names = detect_names(img)
    #     names = get_count_for_name(objects, names)
    #     export_data(names)
    #
    #     print(np.matrix(names))
