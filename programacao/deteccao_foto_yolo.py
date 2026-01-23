from ultralytics import YOLO    
import os
import random
from RPLCD.gpio import CharLCD
import RPi.GPIO as GPIO
from time import sleep

# ================= LCD =================
lcd = CharLCD(
    numbering_mode=GPIO.BOARD,
    cols=16, rows=2,
    pin_rs=37,
    pin_e=35,
    pins_data=[33, 31, 29, 11]
)

# ================= SERVOS =================
SERVO_1 = 12   # Seleciona direção
SERVO_2 = 32   # Empurra objeto

GPIO.setmode(GPIO.BOARD)
GPIO.setup(SERVO_1, GPIO.OUT)
GPIO.setup(SERVO_2, GPIO.OUT)

pwm1 = GPIO.PWM(SERVO_1, 50)
pwm2 = GPIO.PWM(SERVO_2, 50)

pwm1.start(0)
pwm2.start(0)

def mover_servo(pwm, angulo, tempo=0.5):
    duty = 2 + (angulo / 18)
    pwm.ChangeDutyCycle(duty)
    sleep(tempo)
    pwm.ChangeDutyCycle(0)

# ================= YOLO =================
model = YOLO('runs/detect/train/weights/best_100epochs.pt')
path = 'programacao/dataset_yolov8/test/images'

nome_img = random.choice(os.listdir(path))
caminho = os.path.join(path, nome_img)

results = model(caminho)
r = results[0]

lcd.clear()

if r.boxes and len(r.boxes) > 0:
    class_id = int(r.boxes.cls[0].item())
    class_name = r.names[class_id]
    print(f"Classe detectada: {class_name}")

    # ---- Mostra no LCD ----
    if class_name == 'PAPER':
        lcd.write_string('     PAPEL')
        angulo_servo1 = 0

    elif class_name == 'METAL':
        lcd.write_string('     METAL')
        angulo_servo1 = 90

    elif class_name == 'GLASS':
        lcd.write_string('     VIDRO')
        angulo_servo1 = 180

    else:
        lcd.write_string(class_name)
        angulo_servo1 = None

    # ---- Servo 1 se posiciona ----
    if angulo_servo1 is not None:
        mover_servo(pwm1, angulo_servo1, tempo=0.7)
        sleep(0.5)  

        # ---- Servo 2 empurra ----
        mover_servo(pwm2, 0, tempo=0.3)
        sleep(0.2)
        mover_servo(pwm2, 180, tempo=0.6)
        sleep(0.2)
        mover_servo(pwm2, 0, tempo=0.3)
        sleep(0.2)

else:
    lcd.write_string("NENHUM OBJETO")
    print("Nenhum objeto detectado")

sleep(5)

pwm1.stop()
pwm2.stop()
GPIO.cleanup()
results[0].show()