import requests
import time
from ultralytics import YOLO
import os
import random

SERVER = '192.169.38.77'

CONNECT_URL = f'http://{SERVER}:3000/api/connect'
CLASSIFICATION = f'http://{SERVER}:3000/api/classification'

model = YOLO('runs/detect/train/weights/best_100epochs.pt')
path = 'programacao/dataset_yolov8/test/images'

while True:
    try:
        requests.post(CONNECT_URL)

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
        else:
            print('Nenhuma classe detectada')

         
    except Exception as error:
        print('Erro:', error)
    time.sleep(5)


