# class to get camera feed
import numpy as np
import cv2
from skimage import io


class cameraFeed:
    # url that will have latest snapshot
    LATEST_SNAPSHOT_URL = "http://192.168.86.33:8080/shot.jpg"

    def getLatestSnapShot(self):

        # fetch from url
        img = io.imread(self.LATEST_SNAPSHOT_URL)

        # convert to grey and blur
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # detect edges
        img = self.auto_canny(img)

        cv2.imshow("AutoCar Cam", img)

        return img

    def auto_canny(self, image, sigma=0.33):
        # compute the median of the single channel pixel intensities
        v = np.median(image)

        # apply automatic Canny edge detection using the computed median
        lower = int(max(0, (1.0 - sigma) * v))
        upper = int(min(255, (1.0 + sigma) * v))
        edged = cv2.Canny(image, lower, upper)

        # return the edged image
        return edged
