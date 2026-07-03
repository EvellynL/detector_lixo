import cv2
import threading
import time

from picamera2 import Picamera2
from ultralytics import YOLO

# =============================
# CONFIGURAÇÕES
# =============================

MODEL_PATH = "runs/detect/train5/weights/best.pt"

WIDTH = 640
HEIGHT = 480

CONF = 0.5
IMGSZ = 320

# =============================
# CARREGA O YOLO
# =============================

print("Carregando modelo...")

model = YOLO(MODEL_PATH)

# =============================
# CONFIGURA A CÂMERA
# =============================

picam2 = Picamera2()

config = picam2.create_video_configuration(
    main={
        "size": (WIDTH, HEIGHT),
        "format": "RGB888"
    },
    buffer_count=4
)

picam2.configure(config)
picam2.start()

time.sleep(2)

# =============================
# THREAD DA CÂMERA
# =============================

frame = None
running = True
lock = threading.Lock()


def camera_thread():

    global frame

    while running:

        img = picam2.capture_array()

        with lock:
            frame = img


thread = threading.Thread(target=camera_thread, daemon=True)
thread.start()

# =============================
# FPS
# =============================

fps = 0
counter = 0
start = time.time()

print("Pressione Q para sair.")

# =============================
# LOOP PRINCIPAL
# =============================

while True:

    with lock:

        if frame is None:
            continue

        img = frame.copy()

    # -----------------------------
    # YOLO
    # -----------------------------

    results = model.predict(
        img,
        imgsz=IMGSZ,
        conf=CONF,
        verbose=False
    )

    annotated = results[0].plot()

    # -----------------------------
    # FPS
    # -----------------------------

    counter += 1

    if time.time() - start >= 1:

        fps = counter

        counter = 0

        start = time.time()

    cv2.putText(
        annotated,
        f"FPS: {fps}",
        (10,30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0,255,0),
        2
    )

    cv2.imshow("YOLOv8 Raspberry", annotated)

    if cv2.waitKey(1) == ord("q"):
        break

# =============================
# FINALIZA
# =============================

running = False

thread.join()

picam2.stop()

cv2.destroyAllWindows()