import subprocess
import numpy as np
import cv2
from ultralytics import YOLO

# ===== CONFIGURAÇÕES =====
MODEL_PATH = "runs/detect/train/weights/best_100epochs.pt"
WIDTH = 640
HEIGHT = 480

# ===== CARREGAR YOLO =====
print("Carregando modelo YOLO...")
model = YOLO(MODEL_PATH)

# ===== COMANDO DA CÂMERA =====
cmd = [
    "rpicam-vid",
    "-t", "0",
    "--inline",
    "--codec", "yuv420",
    "--width", str(WIDTH),
    "--height", str(HEIGHT),
    "-o", "-"
]

pipe = subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=10**8)

print("Câmera iniciada. Pressione Q para sair.")

while True:
    raw = pipe.stdout.read(WIDTH * HEIGHT * 3 // 2)
    if not raw:
        break

    frame = np.frombuffer(raw, dtype=np.uint8).reshape((HEIGHT * 3 // 2, WIDTH))
    frame = cv2.cvtColor(frame, cv2.COLOR_YUV2BGR_I420)

    # ===== YOLO DETECÇÃO =====
    results = model(frame, imgsz=320, conf=0.5, verbose=False)

    annotated = results[0].plot()

    cv2.imshow("YOLO + Raspberry Cam", annotated)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

pipe.terminate()
cv2.destroyAllWindows()
