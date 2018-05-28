from pluto import pluto


# class to move the car based as per direction specified
class carController:

    # define the arduino pins that control the actions on RC controller
    PIN_RIGHT = 3
    PIN_LEFT = 9
    PIN_FORWARD = 5
    PIN_REVERSE = 6

    # represents the board - arduino uno
    board = None

    def __init__(self):
        # set all pins low
        self.board = pluto.Uno('/dev/cu.usbmodem1411')
        self.stop()

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
