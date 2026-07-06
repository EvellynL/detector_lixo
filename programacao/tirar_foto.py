import os
from datetime import datetime
import cv2
from picamera2 import Picamera2

# Pasta onde as fotos serão salvas
UPLOADS_DIR = "uploads"
os.makedirs(UPLOADS_DIR, exist_ok=True)

# Inicializa a câmera
picam2 = Picamera2()

# Configuração da câmera
config = picam2.create_preview_configuration(
    main={
        "size": (640, 640),
        "format": "BGR888"   # OpenCV utiliza BGR
    }
)

picam2.configure(config)
picam2.start()

print("===================================")
print("ESPACO -> Tirar foto")
print("Q      -> Sair")
print("===================================")

while True:
    # Captura um frame
    frame = picam2.capture_array()
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    # Exibe o vídeo
    cv2.imshow("Raspberry Pi Camera", frame_bgr)
   

    tecla = cv2.waitKey(1) & 0xFF

    # Espaço = tira foto
    if tecla == ord(' '):
        nome = datetime.now().strftime("foto_%Y%m%d_%H%M%S.jpg")
        caminho = os.path.join(UPLOADS_DIR, nome)

        cv2.imwrite(caminho, frame_bgr)
        print(f"Foto salva em: {caminho}")

    # Q = sair
    elif tecla == ord('q'):
        break

picam2.stop()
cv2.destroyAllWindows()