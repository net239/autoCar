from pluto import pluto
board = pluto.Uno('/dev/cu.usbmodem1411')
board.pin(3).low()
board.pin(5).low()
board.pin(6).low()
board.pin(9).low()

# right
# board.pin(3).high()

# forward
# board.pin(5).high()

# reverse
# board.pin(6).high()

# left
# board.pin(9).high()
