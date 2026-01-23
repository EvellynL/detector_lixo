import RPi.GPIO as GPIO
from time import sleep

SERVO_PIN = 32  # GPIO 18 (modo BOARD)

GPIO.setmode(GPIO.BOARD)
GPIO.setup(SERVO_PIN, GPIO.OUT)

pwm = GPIO.PWM(SERVO_PIN, 50)  # 50 Hz
pwm.start(0)

def mover_servo(angulo):
    duty = 2 + (angulo / 18)
    pwm.ChangeDutyCycle(duty)
    sleep(0.5)
    pwm.ChangeDutyCycle(0)

try:
    while True:
        print("0 graus")
        mover_servo(0)
        sleep(1)

        print("90 graus")
        mover_servo(90)
        sleep(1)

        print("180 graus")
        mover_servo(180)
        sleep(1)

except KeyboardInterrupt:
    pass

pwm.stop()
GPIO.cleanup()
