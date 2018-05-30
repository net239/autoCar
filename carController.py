from pluto import pluto


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

    # initialize the board and set all pins to "low"
    def __init__(self, logger, config):
        self.logger = logger
        self.config = config
        self.uno_device = config["uno_device"]

        # set all pins low
        self.board = pluto.Uno(self.uno_device)
        self.stop()

        self.logger.info("Initialized Uno Controller: " + self.uno_device)

    def turnLeft(self):
        self.board.pin(self.PIN_RIGHT).low()
        self.board.pin(self.PIN_LEFT).high()

    def turnRight(self):
        self.board.pin(self.PIN_LEFT).low()
        self.board.pin(self.PIN_RIGHT).high()

    def moveFront(self):
        self.board.pin(self.PIN_REVERSE).low()
        self.board.pin(self.PIN_FORWARD).high()

    def moveBack(self):
        self.board.pin(self.PIN_FORWARD).low()
        self.board.pin(self.PIN_REVERSE).high()

    def stop(self):
        self.board.pin(self.PIN_RIGHT).low()
        self.board.pin(self.PIN_LEFT).low()
        self.board.pin(self.PIN_REVERSE).low()
        self.board.pin(self.PIN_FORWARD).low()
