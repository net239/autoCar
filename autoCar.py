# Program designed to train and run an autonomous toy RC car

import argparse
import os
import yaml
import logging
import time
from logging.handlers import TimedRotatingFileHandler
import pygame
import pandas as pd
import cv2

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

    parser.add_argument('-o', '--mode',
                        help='mode: train - train new model/ ' +
                             'auto - run using given model/ ' +
                             'manual - run using manual keyboard/ ' +
                             'show - display model data',
                        required=False, default='train')

    parser.add_argument('-m', '--model',
                        help='Model file',
                        required=False, default='data/autocar.df')

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

    # load and show a given set of trainign data
    if args.mode == 'show':
        logger.info("Operating Mode: " + args.mode)

        # load model
        logger.info("Show training data: " + args.model)
        df = pd.read_pickle(args.model)
        for index, row in df.iterrows():

            # write the key on image
            img = row["frame"]
            txt = 'key=' + str(row["key"])
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(img, txt, (30, 30), font, 0.8,
                        (255, 255, 255), 1, cv2.LINE_AA)
            cv2.imshow("AutoCar Cam", img)

            # wait for a key press
            event = pygame.event.wait()
            while True:
                if event.type == pygame.KEYDOWN:
                    break
                else:
                    event = pygame.event.wait()

            if event.key == pygame.K_x:
                break

    # Training mode or runnign mode
elif args.mode == 'train' or args.mode == 'auto' or args.mode == 'manual':
        logger.info("Operating Mode: " + args.mode)

        # Car remote controller
        carController = carController(logger, config)

        if args.mode == 'train' or args.mode == 'auto':
            # Camera feed
            cameraFeed = cameraFeed(logger, config)

            # create a pandas object to store images & corresponding direction
            training_data = pd.DataFrame()

        running = True
        while running:

            # see what key is pressed
            ev = pygame.event.get()

            for event in ev:

                capture = False
                dir = ""
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_w or event.key == pygame.K_UP:
                        carController.moveFront()
                        dir = "F"
                        capture = True
                    elif event.key == pygame.K_s or event.key == pygame.K_DOWN:
                        carController.moveBack()
                        # NOTE: do not capture back movement
                        dir = "B"
                        capture = False
                    elif event.key == pygame.K_a or event.key == pygame.K_LEFT:
                        carController.turnLeft()
                        dir = "L"
                        capture = True
                    elif (event.key == pygame.K_d or
                          event.key == pygame.K_RIGHT):
                        carController.turnRight()
                        dir = "R"
                        capture = True
                    elif event.key == pygame.K_x:
                        carController.stop()

                        # break from this loop
                        running = False
                elif event.type == pygame.KEYUP:
                    carController.stop()

            if args.mode == 'train' and capture is True:

                # store the data set
                img = cameraFeed.getLatestSnapShot(display=False)
                img = cameraFeed.processSnapShot(img, display=True)
                df = pd.DataFrame.from_records([{
                        "frame": img,
                        "key": dir}])

                training_data = training_data.append(df)

        # stop the car
        carController.stop()

        # save the data frame
        training_data_file = time.strftime(args.datadir +
                                           "autocarmodel-%Y%m%d-%H%M%S.df")
        training_data.to_pickle(training_data_file)
        logger.info("Saved training data to: " + training_data_file)
