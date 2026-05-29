import requests
import time
from ultralytics import YOLO
import os
import random

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
            
            send_status(f'{class_name} detectado')

            time.sleep(5)

            send_status('Realizando a triagem')
            
            time.sleep(5)
            
            send_status('Triagem finzalizada')

            time.sleep(5)

            send_status('Aguardando nova detecçao')

            time.sleep(10)
        else:
            print('Nenhuma classe detectada')

            send_status('Nenhuma classe encontrada')

            time.sleep(5)

            send_status('Aguardando nova detecçao')

            time.sleep(10)

         
    except Exception as error:
        print('Erro:', error)
    time.sleep(5)


