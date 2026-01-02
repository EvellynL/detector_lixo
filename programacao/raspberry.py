import os
import cv2
from ultralytics import YOLO

model = YOLO(r'runs\\detect\\train2\\weights\\best_250epochs.pt')
path = r'programacao\\dataset_yolov8\\test\\images'

for images in os.listdir(path):
    img_path = os.path.join(path, images)
    imagem = cv2.imread(img_path)

    if imagem is None:
        print('Ocorreu um erro ao carregar a Imagem')
        continue

    results = model(imagem)
    r = results[0]

    if len(r.boxes) > 0:
        class_id = int(r.boxes.cls[0].item())
        class_name = r.names[class_id]
        print(f'Classe detectada: {class_name}')

    else:
        print('Nenhum objeto foi detectado')

    imagem_plot = r.plot()
    cv2.imshow('Detecção YOLO', imagem_plot)

    print("Pressione 'q' para próxima imagem")
    while True:
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cv2.destroyAllWindows()
