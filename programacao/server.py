import requests
import time
from ultralytics import YOLO
import os
import random
from RPLCD.gpio import CharLCD
import RPi.GPIO as GPIO

SERVO_1 = 12 #triagem
SERVO_2 = 32 #push

#SETUP DOS SERVOSMOTORES
GPIO.setmode(GPIO.BOARD)
GPIO.setup(SERVO_1, GPIO.OUT)
GPIO.setup(SERVO_2, GPIO.OUT)

pwm1 = GPIO.PWM(SERVO_1, 50)
pwm2 = GPIO.PWM(SERVO_2, 50)

def mover_servo(pwm, angulo, tempo_movimento=0.6):
    """
    Move o servo para o ângulo desejado e corta o sinal após o movimento
    para evitar trepidações (jitter).
    """
    duty = 2 + (angulo / 18)
    pwm.ChangeDutyCycle(duty)
    time.sleep(tempo_movimento)
    pwm.ChangeDutyCycle(0)

pwm1.start(0)
pwm2.start(0)

SERVER = '192.169.38.77'

CONNECT_URL = f'http://{SERVER}:3000/api/connect'
CLASSIFICATION = f'http://{SERVER}:3000/api/classification'
STATUS_URL = f'http://{SERVER}:3000/api/systemStatus'

def send_status(status):
    requests.post(
        STATUS_URL,

        json={
            'status': status
        }
    )

def iniciar_servos():
    mover_servo(pwm1, 90,tempo_movimento = 0.9)
    mover_servo(pwm2, 0,tempo_movimento = 0.9)
    send_status('Iniciando Servos')

def iniciar_triagem(ang: int):
    if ang is not None:
        send_status('Direcionando Lixeria')
        mover_servo(pwm1, ang, tempo_movimento=0.9)

        send_status('Lixeira posicionada')
        send_status('Iniciando Descarte...')
        time.sleep(0.6)

        mover_servo(pwm2, 150, tempo_movimento=0.9)
        time.sleep(0.5)

        mover_servo(pwm2, 0, tempo_movimento=0.9)
        send_status('Triagem Finalizada!')
        time.sleep(0.6)

model = YOLO('runs/detect/train/weights/best_100epochs.pt')
path = 'programacao/dataset_yolov8/test/images'

while True:
    try:
        requests.post(CONNECT_URL)

        send_status('Iniciando detecçao')

        nome_img = random.choice(os.listdir(path))
        caminho = os.path.join(path, nome_img)

        results = model(caminho)
        r = results[0]

        if r.boxes and len(r.boxes) > 0:
            class_id = int(r.boxes.cls[0].item())
            class_name = r.names[class_id]
            print(f"\n[SUCESSO] Classe detectada: {class_name}")

            with open(caminho, 'rb') as img:
                files = {
                    'image': (
                        nome_img,
                        img,
                        'image/jpeg'
                    )
                }
                data = {
                    'class': class_name
                }

                response = requests.post(
                    CLASSIFICATION, 
                    files=files,
                    data=data,
                )

                if response.status_code == 200:
                    print(
                        'Imagem enviada com sucesso'
                    )
                else: 
                    print(
                        'Erro ao enviar a imagem'
                    )
            
            if class_name == 'PAPER':
                iniciar_triagem(90)
            elif class_name == 'METAL':
                iniciar_triagem(0)
            elif class_name == 'GLASS':
                iniciar_triagem(180)
            else:
                iniciar_triagem(None)
            
            send_status('Aguardando nova detecçcao')
        else:
            print('Nenhuma classe detectada')

            send_status('Nenhuma classe encontrada')

            time.sleep(5)

            send_status('Aguardando nova detecçao')

            time.sleep(10)

         
    except Exception as error:
        print('Erro:', error)
    time.sleep(5)


