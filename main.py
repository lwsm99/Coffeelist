# Main Script which initializes the app
from kivy.app import App
from kivy.uix.label import Label
from kivy.uix.camera import Camera
from kivy.uix.button import Button
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.image import Image
from kivy.clock import Clock
from kivy.uix.screenmanager import ScreenManager, Screen
import numpy as np
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
        self.layout = BoxLayout(orientation='vertical')
        self.button_layout = BoxLayout(orientation='horizontal')

        # Image object
        self.image = Image(source='Coffeelist.png')
        Clock.schedule_interval(lambda dt: self.image.reload(), 0.2)

        # Confirm button object
        self.confirm_btn = Button(text='Confirm')
        self.confirm_btn.size_hint = (.25, .2)
        self.confirm_btn.pos_hint = {'x': .25, 'y': .75}
        self.confirm_btn.bind(on_press=lambda widget: self.start_export())

        # Retry button object
        self.retry_btn = Button(text='Retry')
        self.retry_btn.size_hint = (.25, .2)
        self.retry_btn.pos_hint = {'x': .25, 'y': .75}
        self.retry_btn.bind(on_press=self.switch_photo_screen)

        # Success text object
        self.success_label = Label()
        self.success_label.text = 'Your Coffeelist has successfully been exported!'
        self.success_label.size_hint = (.25, .2)
        self.success_label.pos_hint = {'x': .25, 'y': .75}

        # Add to layout
        self.layout.add_widget(self.image)
        self.button_layout.add_widget(self.confirm_btn)
        self.button_layout.add_widget(self.retry_btn)
        self.layout.add_widget(self.button_layout)
        self.add_widget(self.layout)

    def switch_photo_screen(self, *args):
        self.manager.current = 'camera'

    def start_export(self, *args):
        export_to_table('Coffeelist.png')
        self.layout.add_widget(self.success_label)
        self.button_layout.remove_widget(self.confirm_btn)


def export_to_table(img):
    objects = detect_strokes(img)
    names = detect_names(img)
    names = get_count_for_name(objects, names)
    export_data(names)
    print(np.matrix(names))


class MainApp(App):
    def build(self):
        sm = ScreenManager()
        sm.add_widget(CameraScreen(name='camera'))
        sm.add_widget(ConfirmScreen(name='confirm'))

        return sm


# Initialize App
if __name__ == '__main__':
    MainApp().run()
