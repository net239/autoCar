# Program designed to train and run an autonomous toy RC car

import argparse
import os
import yaml
import logging
import time
from logging.handlers import TimedRotatingFileHandler
import pygame
import pandas as pd

from carController import carController
from cameraFeed import cameraFeed


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='autoCar - train and run ' +
                                     'autonomous toy RC car',
                                     formatter_class=argparse.
                                     ArgumentDefaultsHelpFormatter)

    parser.add_argument('-c', '--config',
                        help='Configuration file',
                        required=False, default='autoCar.yml')

    parser.add_argument('-l', '--logfile',
                        help='Log file',
                        required=False, default='log/autoCar.log')

    parser.add_argument('-d', '--datadir',
                        help='Data Directory',
                        required=False, default='data/')

    parser.add_argument('-m', '--mode',
                        help='mode: train/run',
                        required=False, default='train')

    args = parser.parse_args()
    print (args)

    # read config file
    configfile = args.config
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

    logger.info("Reading configuration: " + configfile)
    config = yaml.safe_load(open(configfile))

    # split path and file name
    path, filename = os.path.split(configfile)

    # add slash at end
    path = os.path.join(path, "")

    pygame.init()
    pygame.display.set_caption('AutoCar')
    window = pygame.display.set_mode((300, 300))

    # Car remote controller
    carController = carController()

    # Camera feed
    cameraFeed = cameraFeed()

    # create a pandas object to store images and corresponding key press
    training_data = pd.DataFrame()

    if args.mode == 'train' or args.mode == 'run':
        logger.info("Operating Mode: " + args.mode)

        running = True
        while running:

            # see what key is pressed
            ev = pygame.event.get()

            for event in ev:

                capture = False
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_w:
                        carController.moveFront()
                        capture = True
                    elif event.key == pygame.K_s:
                        carController.moveBack()
                        # NOTE: do not capture back movement
                        capture = False
                    elif event.key == pygame.K_a:
                        carController.turnLeft()
                        capture = True
                    elif event.key == pygame.K_d:
                        carController.turnRight()
                        capture = True
                    elif event.key == pygame.K_x:
                        carController.stop()

                        # break from this loop
                        running = False
                elif event.type == pygame.KEYUP:
                    carController.stop()

            if capture:

                # store the data set
                df = pd.DataFrame.from_records([{
                                   "frame": cameraFeed.getLatestSnapShot(),
                                   "key": event.key}])

                training_data.append(df)
                logger.info("Saved frame: %d", event.key)

        # stop the car
        carController.stop()

        # save the data frame
        training_data_file = time.strftime(args.datadir +
                                           "autocarmodel-%Y%m%d-%H%M%S.df")
        training_data.to_pickle(training_data_file)
        logger.info("Saved training data to: " + training_data_file)
