from ultralytics import YOLO
import cv2
import serial
import time
import os

# ===== MODELO YOLO =====
model = YOLO(r'runs\\detect\\train2\\weights\\best_250epochs.pt')

# ===== CAMINHO DAS IMAGENS =====
path = r'programacao\\dataset_yolov8\\test\\images'

# ===== SERIAL ARDUINO =====
arduino = serial.Serial('COM4', 9600)
time.sleep(2)  # tempo para o Arduino iniciar

# ===== LOOP NAS IMAGENS =====
for nome_img in os.listdir(path):

    img_path = os.path.join(path, nome_img)
    imagem = cv2.imread(img_path)

    if imagem is None:
        print(f'Erro ao ler imagem: {nome_img}')
        continue


    results = model(imagem)
    r = results[0]


    if len(r.boxes) > 0:
        class_id = int(r.boxes.cls[0].item())
        class_name = r.names[class_id]

        print(f'Classe detectada: {class_name}')


        if class_name == 'METAL':
            arduino.write(b'M')
        elif class_name == 'GLASS':
            arduino.write(b'V')
        elif class_name == 'PAPER':
            arduino.write(b'P')

    else:
        print('Nenhum objeto detectado')


    imagem_plot = r.plot()
    cv2.imshow('Detecção YOLO', imagem_plot)

    print("Pressione 'q' para próxima imagem")
    while True:
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


arduino.close()
cv2.destroyAllWindows()

time.sleep(0.2)