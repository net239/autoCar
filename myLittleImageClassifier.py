import numpy as np
import argparse
import logging
from logging.handlers import TimedRotatingFileHandler


class myLittleImageClassifier:

    # to construct input vector
    imageWidth = None
    imageHeight = None
    batchSize = None

    # logger
    logger = None

    # the input and output vectors
    X = None
    Y = None

    # image width and height define the size of input images
    # to be classified. minibatch size is number of images that
    # will be trained in one go
    def __init__(self, logger, imageWidth=28, imageHeight=28, batchSize=128):
        self.logger = logger
        self.imageWidth = imageWidth
        self.imageHeight = imageHeight
        self.batchSize = batchSize

        # construct the input and output vectors
        # input vector is image of width and height given, converted to a
        # single dimension vector
        X = np.zeros((self.batchSize, self.imageHeight * self.imageWidth))
        Y = np.zeros((self.batchSize, 1))

        self.logger.info("Initialized myLittleImageClassifier. X = %s Y = %s",
                         X.shape, Y.shape)

    # sigmoid activation function and its derivative
    def __nonlin(x, deriv=False):
        if (deriv is True):
            return x*(1-x)
        return 1 / (1 + np.exp(-x))

    # train the model on this batch
    def trainBatch(images, labels):
        return


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='myLittleImageClassifier: ' +
                                     'test utility to train and evaluate',
                                     formatter_class=argparse.
                                     ArgumentDefaultsHelpFormatter)

    parser.add_argument('-l', '--logfile',
                        help='Log file',
                        required=False,
                        default='log/myLittleImageClassifier.log')

    parser.add_argument('-d', '--datadir',
                        help='Data Directory',
                        required=False, default='data/')

    parser.add_argument('-o', '--mode',
                        help='mode: train - train new model/ ' +
                             'show - display model data',
                        required=False, default='train')

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

    if args.mode == 'train':
        logger.info("Operating Mode: " + args.mode)

        imageClassifier = myLittleImageClassifier(logger)
