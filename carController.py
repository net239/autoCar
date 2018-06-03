from pluto import pluto
import time

# class to move the car based as per direction specified
class carController:

    # define the arduino pins that control the actions on RC controller
    PIN_RIGHT = 3
    PIN_LEFT = 9
    PIN_FORWARD = 5
    PIN_REVERSE = 6

    # path of the uno board device
    uno_device = None

    # represents the board - arduino uno
    board = None

    # constants representing car movement and direction
    MOVEMENT_STOP = 0x0000
    MOVEMENT_FORWWARD = 0x0001
    MOVEMENT_REVERSE = 0x0010

    DIRECTION_STRAIGHT = 0x0000
    DIRECTION_LEFT = 0x0100
    DIRECTION_RIGHT = 0x1000

    # represents which way the car is going
    # Forward - F, Reverse  - R or Stop - S
    movement = MOVEMENT_STOP

    # direction - left - L / right - R or straight - S
    direction = DIRECTION_STRAIGHT

    # initialize the board and set all pins to "low"
    def __init__(self, logger, config):
        self.logger = logger
        self.config = config
        self.uno_device = config["uno_device"]

        # set all pins low
        self.board = pluto.Uno(self.uno_device)
        self.stop(wheelsStraight=True)

        self.logger.info("Initialized Uno Controller: " + self.uno_device)

    def turnLeft(self):

        # if the car is to right, make it straight, else left
        if self.direction == self.DIRECTION_RIGHT:
            self.board.pin(self.PIN_RIGHT).low()
            self.board.pin(self.PIN_LEFT).low()
            self.direction = self.DIRECTION_STRAIGHT
        else:
            self.board.pin(self.PIN_RIGHT).low()
            self.board.pin(self.PIN_LEFT).high()
            self.direction = self.DIRECTION_LEFT

    def turnRight(self):
        # if the car is to left, make it straight, else right
        if self.direction == self.DIRECTION_LEFT:
            self.board.pin(self.PIN_RIGHT).low()
            self.board.pin(self.PIN_LEFT).low()
            self.direction = self.DIRECTION_STRAIGHT
        else:
            self.board.pin(self.PIN_LEFT).low()
            self.board.pin(self.PIN_RIGHT).high()
            self.direction = self.DIRECTION_RIGHT

    def moveFront(self):
        self.board.pin(self.PIN_REVERSE).low()
        self.board.pin(self.PIN_FORWARD).high()

        # sleep a few milliseconds
        time.sleep(0.15)

        self.board.pin(self.PIN_REVERSE).low()
        self.board.pin(self.PIN_FORWARD).low()

        self.movement = self.MOVEMENT_FORWWARD

    def moveBack(self):
        self.board.pin(self.PIN_FORWARD).low()
        self.board.pin(self.PIN_REVERSE).high()

        # sleep a few milliseconds
        time.sleep(0.10)

        self.board.pin(self.PIN_REVERSE).low()
        self.board.pin(self.PIN_FORWARD).low()

        self.movement = self.MOVEMENT_REVERSE

    def stop(self, wheelsStraight=False):

        if wheelsStraight:
            self.board.pin(self.PIN_RIGHT).low()
            self.board.pin(self.PIN_LEFT).low()
            self.movement = self.MOVEMENT_STOP

        self.board.pin(self.PIN_REVERSE).low()
        self.board.pin(self.PIN_FORWARD).low()

    def getCurrentDirection(self):
        return self.direction

    def getCurrentMovement(self):
        return self.movement

    def getDirectionAndMovement(self):
        return (self.direction | self.movement)
