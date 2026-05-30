from ultralytics import YOLO    
import os
import random
from RPLCD.gpio import CharLCD
import RPi.GPIO as GPIO
from time import sleep

# ================= CONFIGURAÇÕES =================
SERVO_1 = 12   # Seleciona direção (Maior) -> Inicia em 90°
SERVO_2 = 32   # Empurra objeto (Menor)   -> Inicia em 0°

GPIO.setmode(GPIO.BOARD)
GPIO.setup(SERVO_1, GPIO.OUT)
GPIO.setup(SERVO_2, GPIO.OUT)

pwm1 = GPIO.PWM(SERVO_1, 50)
pwm2 = GPIO.PWM(SERVO_2, 50)

# ================= FUNÇÃO DE MOVIMENTAÇÃO =================
def mover_servo(pwm, angulo, tempo_movimento=0.6):
    """
    Move o servo para o ângulo desejado e corta o sinal após o movimento
    para evitar trepidações (jitter).
    """
    duty = 2 + (angulo / 18)
    pwm.ChangeDutyCycle(duty)
    sleep(tempo_movimento)  # Tempo necessário para o braço físico chegar ao destino
    pwm.ChangeDutyCycle(0)  # Desliga o sinal para o servo ficar imóvel e silencioso

# ================= INICIALIZAÇÃO DOS SERVOS =================
print("Inicializando e calibrando servos...")
pwm1.start(0)
pwm2.start(0)

# Define as posições iniciais solicitadas
mover_servo(pwm1, 90, tempo_movimento=0.8)   # Servo maior inicia no meio (90°)
mover_servo(pwm2, 0, tempo_movimento=0.8)    # Servo menor inicia fechado (0°)
sleep(0.5)

# ================= LCD =================
#lcd = CharLCD(
#    numbering_mode=GPIO.BOARD,
#    cols=16, rows=2,
#    pin_rs=37, pin_e=35,
#    pins_data=[33, 31, 29, 11]
#)

# ================= YOLO (REDE NEURAL) =================
print("Carregando modelo YOLO...")
model = YOLO('runs/detect/train/weights/best_100epochs.pt')
path = 'programacao/dataset_yolov8/test/images'

nome_img = random.choice(os.listdir(path))
caminho = os.path.join(path, nome_img)

results = model(caminho)
r = results[0]

#lcd.clear()

if r.boxes and len(r.boxes) > 0:
    class_id = int(r.boxes.cls[0].item())
    class_name = r.names[class_id]
    print(f"\n[SUCESSO] Classe detectada: {class_name}")

    # ---- Classificação e Seleção de Ângulo (Servo 1) ----
    if class_name == 'PAPER':
        #lcd.write_string('     PAPEL')
        angulo_servo1 = 90
    elif class_name == 'METAL':
        #lcd.write_string('     METAL')
        angulo_servo1 = 0
    elif class_name == 'GLASS':
        #lcd.write_string('     VIDRO')
        angulo_servo1 = 180
    else:
        #lcd.write_string(class_name)
        angulo_servo1 = None

    # ---- Execução da Triagem Automatizada ----
    if angulo_servo1 is not None:
        # 1. Servo 1 vai para a lixeira correta
        print(f"Direcionando lixeira (Servo 1) para {angulo_servo1}°...")
        mover_servo(pwm1, angulo_servo1, tempo_movimento=0.8)
        
        # 2. Confirmação física de posicionamento concluído
        print("Confirmação: Lixeira posicionada. Iniciando descarte...")
        sleep(0.5)  # Margem de segurança mecânica

        # 3. Servo 2 abre (180°) para empurrar o objeto
        print("Empurrando objeto (Servo 2 -> 180°)...")
        mover_servo(pwm2, 150, tempo_movimento=0.6)
        
        # 4. Aguarda o objeto cair/escorregar
        print("Aguardando 3 segundos com a rampa aberta...")
        sleep(3.0)
        
        # 5. Servo 2 retorna para a posição fechada (0°)
        print("Retornando empurrador (Servo 2 -> 0°)...")
        mover_servo(pwm2, 0, tempo_movimento=0.6)
        
        print("Processo de triagem concluído com sucesso!")

else:
    #lcd.write_string("NENHUM OBJETO")
    print("\n[AVISO] Nenhum objeto detectado na imagem.")

# Tempo de espera antes de encerrar o script
sleep(2)

# ================= FINALIZAÇÃO DO SISTEMA =================
print("Encerrando sistemas e liberando pinos...")
pwm1.stop()
pwm2.stop()
GPIO.cleanup()

# Opcional: exibe a imagem com a bbounding box gerada pela IA
results[0].show()