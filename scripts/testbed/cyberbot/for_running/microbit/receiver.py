#File to flash onto reciever microbit. Paste into main.py as needed. Use summer 2023 code documentation

# terminal_controlled_bot_wireless
from microbit import *
from cyberbot import *
from feedback360 import *
from feedback360 import drive
import radio

radio.on()
radio.config(channel=7,length=64)
drive.connect()

while True:
    packet = radio.receive()

    if packet is not None:
        dict = eval(packet)

        vL = dict['vL']
        vR = dict['vR']
        drive.speed(vL, vR)
