from ultralytics import YOLO
import cv2

# Carregar modelo
model = YOLO("/home/evellyn/Desktop/detector_lixo/runs/detect/train2/weights/best_250epochs.pt")

# Abrir câmera CSI (pipeline correto)
cap = cv2.VideoCapture(
    "libcamerasrc ! video/x-raw,width=640,height=480,framerate=30/1 ! videoconvert ! appsink",
    cv2.CAP_GSTREAMER
)

if not cap.isOpened():
    print("Erro: câmera não abriu")
    exit()

print("Câmera aberta! Pressione Q para sair.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Frame vazio")
        continue

    results = model(frame, imgsz=320, conf=0.5)
    annotated = results[0].plot()

    cv2.imshow("Detector de Lixo", annotated)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
