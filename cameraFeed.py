# class to get camera feed
import numpy as np
import cv2
import urllib


# This class get the camera feed from the andrpid phone
# mounted on top of car - runnign the app 'IP WebCam'
class cameraFeed:

    # url that will have latest snapshot
    LATEST_SNAPSHOT_URL = None

    # logger to put all output ( emails sent, blocked etc.)
    logger = None

    # config
    config = None

    def __init__(self, logger, config):
        self.logger = logger
        self.config = config
        self.LATEST_SNAPSHOT_URL = config['LATEST_SNAPSHOT_URL']

        self.logger.info("Using Camera Feed: " + self.LATEST_SNAPSHOT_URL)

    # get latest scene seen by camera
    def getLatestSnapShot(self, display=False):
        # fetch from url
        req = urllib.urlopen(self.LATEST_SNAPSHOT_URL)
        arr = np.asarray(bytearray(req.read()), dtype=np.uint8)
        img = cv2.imdecode(arr, -1)  # 'Load it as it is'

        if display:
            # show the image
            cv2.imshow("AutoCar Cam", img)
        return img

    # process image - convert to black and white and detect edges
    def processSnapShot(self, img, display=False):

        # convert to grey scale and slightly blur it
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # img = cv2.GaussianBlur(img, (3, 3), 0)

        # detect edges
        # img = self.auto_canny(img)

        if display:
            # show the image
            cv2.imshow("AutoCar Cam", img)

        return img

    # refer : https://www.pyimagesearch.com/2015/04/06/zero-parameter-
    #         automatic-canny-edge-detection-with-python-and-opencv/
    def auto_canny(self, image, sigma=0.33):
        # compute the median of the single channel pixel intensities
        v = np.median(image)

        # apply automatic Canny edge detection using the computed median
        lower = int(max(0, (1.0 - sigma) * v))
        upper = int(min(255, (1.0 + sigma) * v))
        edged = cv2.Canny(image, lower, upper)

        # return the edged image
        return edged
