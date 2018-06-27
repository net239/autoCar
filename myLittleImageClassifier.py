import numpy as np
import pandas as pd
import argparse
import logging
import matplotlib.pyplot as plt
from logging.handlers import TimedRotatingFileHandler
from tensorflow.examples.tutorials.mnist import input_data
import math
import pickle
import os


class myLittleImageClassifier:

    # to construct input vector, image width and height are dimension
    # of the input image - this will be flattened to a single
    # dimension vector
    imageWidth = None
    imageHeight = None

    # number of samples to train in a batch
    batchSize = None

    # logger
    logger = None

    # the input and output vectors
    X = None
    Y = None

    # tracks number of samples currently stored, in the batch
    count_samples = 0

    # weight matrices - two layer network
    syn0 = None
    syn1 = None

    # image width and height define the size of input images
    # to be classified. minibatch size is number of images that
    # will be trained in one go
    # imageWidth, Height, batchSize are explained above
    # numClasses is number of classes in which the images
    # are to be classified. e.g. 0 to 9 is total 10 digits
    # layer2_neurons = number of neurons in layer2
    def __init__(self, logger,
                 imageWidth=28, imageHeight=28,
                 batchSize=16,
                 numClasses=10,
                 layer2_neurons=15):
        self.logger = logger
        self.imageWidth = imageWidth
        self.imageHeight = imageHeight
        self.batchSize = batchSize

        # construct the input and output vectors
        # output vector is  having 10 labels
        self.X = np.zeros((self.batchSize, self.imageHeight * self.imageWidth))
        self.Y = np.zeros((self.batchSize, numClasses))

        # lets define two weight matrices for this 2 layer
        # fully connected network
        # the first weight matrix connects input X (layer0) to layer1
        # and the second weight matrix connects inout from layer1 to layer2
        # the weights are assigned randomly to start with and
        # are between -1 and 1, with mean as zero
        # number fo input features and number of samples
        n_samples, n_features = self.X.shape
        ignore, n_classes = self.Y.shape

        # initialize random seed
        np.random.seed(1)

        ub = 1
        lb = -1
        self.syn0 = (ub-lb)*np.random.random((n_features, layer2_neurons)) + lb
        self.syn1 = (ub-lb)*np.random.random((layer2_neurons, n_classes)) + lb

        self.logger.info("Initialized myLittleImageClassifier " +
                         "X = %s Y = %s syn0 = %s syn1 = %s",
                         self.X.shape, self.Y.shape,
                         self.syn0.shape, self.syn1.shape)

    # sigmoid activation function and its derivative
    def nonlin(self, x, deriv=False):
        if (deriv is True):
            return x*(1.0-x)
        return 1.0 / (1.0 + np.exp(-x))

    def getBatchSize(self):
        return self.batchSize

    def getCountSamples(self):
        return self.count_samples

    # add a single image and label to training batch
    def addToTrainingBatch(self, image, label):
        if (self.count_samples < self.batchSize):
            self.X[self.count_samples] = image
            self.Y[self.count_samples] = label
            self.count_samples = self.count_samples + 1
            return 0
        else:
            self.logger.error("Batch already full: %d", self.count_samples)
            return -1

    # empty any exisrtign training data
    def clearCurrentBatch(self):
        self.X.fill(0)
        self.Y.fill(0)
        self.count_samples = 0

    # train the model on the existing stored batch
    def trainOnExistingBatch(self, iterations, showProgress=False):

        if self.count_samples <= 0:
            self.logger.error("No Samples to train on: %d", self.count_samples)
            return -1

        # use these to see progress of all iterations in the network
        if showProgress:
            track_error1 = pd.DataFrame()
            track_error2 = pd.DataFrame()

        # lets train our network
        for j in xrange(iterations):

            # Lets first apply the weights to the input and forward to layer1
            # and then to layer2
            layer0 = self.X
            layer1 = self.nonlin(np.dot(layer0, self.syn0))
            layer2 = self.nonlin(np.dot(layer1, self.syn1))

            # lets see how far away we are from the actual solution
            # start from the last layer and move backward
            error2 = self.Y - layer2

            # see how much we need to adjust weights
            # error times slope of the output
            delta2 = error2 * self.nonlin(layer2, deriv=True)

            # how much did each l1 value contribute to the l2 error
            # (according to the weights)?
            error1 = np.dot(delta2, self.syn1.T)

            # see how much we need to adjust weights
            # error times slope
            delta1 = error1 * self.nonlin(layer1, deriv=True)

            self.syn1 += np.dot(layer1.T, delta2)
            self.syn0 += np.dot(layer0.T, delta1)

            # store all data, so we can analyze them later
            if showProgress:
                data = error1.T
                track_error1 = track_error1.append(pd.DataFrame(data),
                                                   ignore_index=True)

                data = error2.T
                track_error2 = track_error2.append(pd.DataFrame(data),
                                                   ignore_index=True)

        if showProgress:
            # plot the data generated in above iterations
            fig, axs = plt.subplots(2, 2, sharex=True, sharey=True)
            axs[0, 0].plot(track_error1, label="error1")
            axs[0, 0].set_title("error1")

            axs[0, 1].plot(track_error2, label="error2")
            axs[0, 1].set_title("error2")

            plt.show()

    # predict using the currently stored model
    def predict(self, image):

        # Lets first apply the weights to the input and forward to layer1
        # and then to layer2
        layer0 = image
        layer1 = self.nonlin(np.dot(layer0, self.syn0))
        layer2 = self.nonlin(np.dot(layer1, self.syn1))
        Y = layer2

        # round off
        for n in xrange(numClasses):
            Y[n] = round(Y[n], 2)

        # find index of the maximum value
        maxIndex = np.argmax(Y, axis=0)
        return (Y, maxIndex)

    # display the image
    def displayImage(self, image, title):
        image = image
        image = np.array(image, dtype='float')
        pixels = image.reshape((self.imageWidth, self.imageHeight))
        plt.imshow(pixels, cmap='gray')
        plt.title(title)
        plt.show()

    # train on this given set of images and labels
    # perSampleIterations is number of times the same batch is passed over
    # and over again to adjust and readjust the weights
    # epochs is how many times we go over the training samples all over again
    def trainOnImageSet(self, images, labels,
                        perBatchIterations=1,
                        epochs=5,
                        showProgress=True):

        nImages = images.shape[0]
        for epoch in xrange(epochs):

            self.clearCurrentBatch()
            training_count = 0
            is_last = False
            for i in xrange(nImages):

                # keep track if we are at the last image in the batch
                if i >= nImages-1:
                    is_last = True

                # add to batch
                self.addToTrainingBatch(images[i],
                                        labels[i])

                # time to train?
                if (is_last or self.getCountSamples() >= self.getBatchSize()):
                    self.trainOnExistingBatch(perBatchIterations)
                    self.clearCurrentBatch()
                    training_count = training_count + 1

                    if (is_last or (training_count % 100 == 0)):
                        logger.info("epoch:%d batch: %d total_images:%d " +
                                    "iter:%d training_count:%d",
                                    epoch,
                                    self.getBatchSize(), nImages, i,
                                    training_count)

                        if showProgress:
                            self.checkAccuracy(images, labels)

    # check accuracy against given set of images
    def checkAccuracy(self, images, labels):
        n = images.shape[0]
        correct = 0
        for i in xrange(n):
            (Y, V) = self.predict(images[i])
            maxIndex = np.argmax(labels[i], axis=0)
            if (maxIndex == V):
                correct = correct+1

        logger.info("Correct %d out of %d. percentage=%2.2f%%",
                    correct, n,
                    (correct*1.0)/n * 100)
        return correct

    # save existign model to given file
    def saveModel(self, filename):
        model = {"sync0": self.syn0, "sync1": self.syn1}
        pickle.dump(model, open(filename, "wb"))

    # load model from the given file
    def loadModel(self, filename):
        model = pickle.load(open(filename, "rb"))
        self.syn0 = model["sync0"]
        self.syn1 = model["sync1"]


# to take booleans on command line
def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


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
                        help='mode: train - train new model or ' +
                             'mode: eval  - evaluate existing model',
                        required=False, default='train')

    parser.add_argument('-m', '--model',
                        help='Model file',
                        required=False,
                        default='data/myLittleImageClassifier.df')

    parser.add_argument('-f', '--firstimage', type=int,
                        help='index of the first image to be used in the set',
                        required=False,
                        default=0)

    parser.add_argument('-e', '--lastimage', type=int,
                        help='index of the last image to be used in the set',
                        required=False,
                        default=1000)

    parser.add_argument('-s', '--save', type=str2bool,
                        help='save the trained model',
                        required=False,
                        default=True)

    parser.add_argument('-a', '--load', type=str2bool,
                        help='load the existing trained model',
                        required=False,
                        default=True)

    parser.add_argument('-p', '--epochs', type=int,
                        help='epochs - how many times to train the set',
                        required=False,
                        default=5)

    parser.add_argument('-b', '--batchsize', type=int,
                        help='batch size: number of samples to train together',
                        required=False,
                        default=1)

    parser.add_argument('-y', '--layer2', type=int,
                        help='number of neurons in layer2',
                        required=False,
                        default=100)

    parser.add_argument('-i', '--per_batch_iters', type=int,
                        help='no of times to adjust weights per batch',
                        required=False,
                        default=1)

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

        # read handwritten character images
        mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
        maxImages = mnist.test.images.shape[0]
        imageWidth = int(math.sqrt(mnist.test.images.shape[1]))
        imageHeight = imageWidth
        numClasses = mnist.test.labels[0].shape[0]

        logger.info("MNIST Traing set has %d images. " +
                    "width=%d height=%d classes=%d",
                    maxImages, imageWidth, imageHeight, numClasses
                    )

        # limit our world to first 1000 images
        firstImage = args.firstimage
        lastImage = args.lastimage

        # create instance of a image classifier
        classifier = myLittleImageClassifier(logger,
                                             imageWidth,
                                             imageHeight,
                                             batchSize=args.batchsize,
                                             layer2_neurons=args.layer2)

        # see if we have a saved model
        if (args.load and os.path.isfile(args.model)):
            logger.info("Loading model " + args.model)
            classifier.loadModel(args.model)

        classifier.trainOnImageSet(mnist.test.images[firstImage: lastImage],
                                   mnist.test.labels[firstImage: lastImage],
                                   perBatchIterations=args.per_batch_iters,
                                   epochs=args.epochs
                                   )

        # check accuracy on same image set
        correct = classifier.checkAccuracy(
                                 mnist.test.images[firstImage: lastImage],
                                 mnist.test.labels[firstImage: lastImage])

        if args.save:
            logger.info("Saving model " + args.model)
            classifier.saveModel(args.model)
