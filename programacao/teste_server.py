import cv2
import time
import threading
import requests
import numpy as np

from ultralytics import YOLO
from picamera2 import Picamera2

from RPLCD.gpio import CharLCD
import RPi.GPIO as GPIO


# =====================================================
# CONFIGURAÇÕES
# =====================================================

MODEL_PATH = "runs/detect/train/weights/best_100epochs.pt"

WIDTH = 640
HEIGHT = 640

CONF = 0.5
IMGSZ = 320

DETECTION_INTERVAL = 0.2  # segundos (5 inferências/s)

SERVER = "192.169.38.77"

CONNECT_URL = f"http://{SERVER}:3000/api/connect"
CLASSIFICATION = f"http://{SERVER}:3000/api/classification"
STATUS_URL = f"http://{SERVER}:3000/api/systemStatus"

SERVO_1 = 12
SERVO_2 = 32


# =====================================================
# GPIO
# =====================================================

GPIO.setmode(GPIO.BOARD)

GPIO.setup(SERVO_1, GPIO.OUT)
GPIO.setup(SERVO_2, GPIO.OUT)

pwm1 = GPIO.PWM(SERVO_1, 50)
pwm2 = GPIO.PWM(SERVO_2, 50)

pwm1.start(0)
pwm2.start(0)


# =====================================================
# SERVOS
# =====================================================

def mover_servo(pwm, angulo, tempo_movimento=0.9):

    duty = 2 + (angulo / 18)

    pwm.ChangeDutyCycle(duty)

    time.sleep(tempo_movimento)

    pwm.ChangeDutyCycle(0)


def send_status(status):

    try:

        requests.post(
            STATUS_URL,
            json={"status": status},
            timeout=3
        )

    except:
        pass


def iniciar_triagem(angulo):

    send_status("Direcionando Lixeira")

    mover_servo(pwm1, angulo)

    send_status("Lixeira posicionada")

    time.sleep(0.6)

    send_status("Iniciando descarte")

    mover_servo(pwm2, 150)

    time.sleep(0.5)

    mover_servo(pwm2, 0)

    send_status("Triagem finalizada")

    time.sleep(0.5)


# =====================================================
# YOLO
# =====================================================

print("Carregando modelo...")

model = YOLO(MODEL_PATH)


# =====================================================
# PICAMERA2
# =====================================================

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


# =====================================================
# THREAD DA CÂMERA
# =====================================================

frame = None

frame_lock = threading.Lock()

running = True


def camera_thread():

    global frame

    while running:

        img = picam2.capture_array()

        with frame_lock:
            frame = img


threading.Thread(
    target=camera_thread,
    daemon=True
).start()


# =====================================================
# FPS
# =====================================================

fps = 0

contador = 0

inicio = time.time()


# =====================================================
# CONTROLE
# =====================================================

busy = False

ultimo_processamento = 0


print("Sistema iniciado.")


# =====================================================
# LOOP PRINCIPAL
# =====================================================

while True:

    requests.post(CONNECT_URL)

    with frame_lock:

        if frame is None:
            continue

        imagem = frame.copy()

    # -------------------------------------------------

    if (not busy) and (time.time() - ultimo_processamento >= DETECTION_INTERVAL):

        ultimo_processamento = time.time()

        results = model.predict(
            imagem,
            imgsz=IMGSZ,
            conf=CONF,
            verbose=False
        )

        r = results[0]

        annotated = r.plot()

        if r.boxes is not None and len(r.boxes) > 0:

            busy = True

            class_id = int(r.boxes.cls[0])

            class_name = r.names[class_id]

            print(f"\nClasse detectada: {class_name}")

            send_status("Objeto detectado")

            # -----------------------------------------
            # ENVIAR IMAGEM
            # -----------------------------------------

            ok, buffer = cv2.imencode(".jpg", imagem)

            if ok:

                files = {

                    "image": (
                        "camera.jpg",
                        buffer.tobytes(),
                        "image/jpeg"
                    )

                }

                data = {

                    "class": class_name

                }

                try:

                    requests.post(
                        CLASSIFICATION,
                        files=files,
                        data=data,
                        timeout=5
                    )

                except Exception as e:

                    print(e)

            # -----------------------------------------
            # TRIAGEM
            # -----------------------------------------

            if class_name == "PAPER":

                iniciar_triagem(90)

            elif class_name == "METAL":

                iniciar_triagem(0)

            elif class_name == "GLASS":

                iniciar_triagem(180)

            else:

                send_status("Classe sem triagem")

            send_status("Aguardando nova detecção")

            busy = False

    else:

        annotated = imagem

    # -------------------------------------------------
    # FPS
    # -------------------------------------------------

    contador += 1

    if time.time() - inicio >= 1:

        fps = contador

        contador = 0

        inicio = time.time()

    cv2.putText(

        annotated,

        f"FPS: {fps}",

        (10, 30),

        cv2.FONT_HERSHEY_SIMPLEX,

        1,

        (0, 255, 0),

        2

    )

    cv2.imshow("Sistema de Triagem", annotated)

    tecla = cv2.waitKey(1)

    if tecla == ord("q"):

        break


# =====================================================
# FINALIZA
# =====================================================

running = False

picam2.stop()

GPIO.cleanup()

cv2.destroyAllWindows()