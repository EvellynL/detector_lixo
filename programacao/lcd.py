from RPLCD.gpio import CharLCD
import RPi.GPIO as GPIO
from time import sleep

lcd = CharLCD(
    numbering_mode=GPIO.BOARD,
    cols=16, rows=2,
    pin_rs=37,
    pin_e=35,
    pins_data=[33, 31, 29, 11]
)

lcd.clear()
lcd.write_string("Ola Evellyn!")
lcd.cursor_pos = (1, 0)
lcd.write_string("LCD no Raspberry")

sleep(10)
lcd.clear()
