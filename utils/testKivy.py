from kivy.app import App
from kivy.uix.widget import Widget
from kivy.graphics import Color, Line
from PIL import Image, ImageChops
from kivy.clock import Clock


class MyPaintWidget(Widget):

    # timer event, to track when user has stopped drawing
    event = None

    def on_touch_down(self, touch):
        # cancel any existign timers
        if self.event:
            self.event.cancel()

        # start drawing
        with self.canvas:
            Color(1, 1, 1)
            touch.ud['line'] = Line(points=(touch.x, touch.y), width=10)

    def on_touch_move(self, touch):
        touch.ud['line'].points += [touch.x, touch.y]

    def on_touch_up(self, touch):
        # call my_callback every few seconds
        self.event = Clock.schedule_once(lambda dt:
                                         self.processCanvasImage(), 1)

    def trimImage(self, image):
        bg = Image.new(image.mode, image.size, image.getpixel((0, 0)))
        diff = ImageChops.difference(image, bg)
        diff = ImageChops.add(diff, diff, 2.0, -100)
        bbox = diff.getbbox()
        if bbox:
            return image.crop(bbox)
        else:
            return image

    def processCanvasImage(self):
        # save the canvas to a png  file
        imagefile = 'data/canvas.tmp.png'
        self.export_to_png(imagefile)

        # trim the image
        image = Image.open(imagefile)
        image = self.trimImage(image)
        size = (28, 28)
        image.thumbnail(size, Image.ANTIALIAS)

        background = Image.new('RGB', size, (0, 0, 0))
        background.paste(
             image, (int((size[0] - image.size[0]) / 2),
                     int((size[1] - image.size[1]) / 2))
        )
        image = background

        image.show()
        print image
        self.canvas.clear()


class MyPaintApp(App):

    def build(self):
        return MyPaintWidget()


if __name__ == '__main__':
    MyPaintApp().run()
