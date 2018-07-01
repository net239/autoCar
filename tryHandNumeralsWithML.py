from kivy.app import App
from kivy.uix.widget import Widget
from kivy.graphics import Color, Line
from PIL import Image, ImageChops
from kivy.clock import Clock
from myLittleImageClassifier import myLittleImageClassifier
import argparse
import logging
from logging.handlers import TimedRotatingFileHandler
import numpy as np


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
            touch.ud['line'] = Line(points=(touch.x, touch.y), width=20)

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

        # comvert to greyscale
        image = image.convert('L')

        # convert to MNIST format
        img_array = np.zeros((image.size[0] * image.size[1]))
        w, h = image.size

        n = 0
        for x in xrange(w):
            for y in xrange(h):
                img_array[n] = image.getpixel((y, x))
                n = n + 1

        img_array = img_array / 255

        # now lets prdidct using the ML model
        classifier = App.get_running_app().classifier
        (Y, V) = classifier.predict(img_array)
        print V
        self.canvas.clear()


class MyPaintApp(App):

    classifier = None

    def build(self):
        parser = argparse.ArgumentParser(
                                description='tryHandNumeralsWithML: ' +
                                'Type numerals with hand and use ' +
                                'ML Model to classify',
                                formatter_class=argparse.
                                ArgumentDefaultsHelpFormatter)

        parser.add_argument('-l', '--logfile',
                            help='Log file',
                            required=False,
                            default='log/tryHandNumeralsWithML.log')

        parser.add_argument('-d', '--datadir',
                            help='Data Directory',
                            required=False, default='data/')

        parser.add_argument('-m', '--model',
                            help='Model file',
                            required=False,
                            default='data/myLittleImageClassifier.df')

        args = parser.parse_args()
        print (args)

        # read config file
        logfile = args.logfile
        print "Check all output in TimedRotatingFileHandler: " + logfile

        # set logger
        logHandler = TimedRotatingFileHandler(logfile, when="midnight")
        logFormatter = logging.Formatter('%(asctime)s %(message)s')
        logHandler.setFormatter(logFormatter)
        logger = logging.getLogger('LogWatcherLogger')
        logger.addHandler(logHandler)
        logger.addHandler(logging.StreamHandler())
        logger.setLevel(logging.INFO)

        # create instance of a image classifier
        imageWidth, imageHeight = 28, 28
        self.classifier = myLittleImageClassifier(logger,
                                                  imageWidth,
                                                  imageHeight)

        return MyPaintWidget()


if __name__ == '__main__':
    MyPaintApp().run()
