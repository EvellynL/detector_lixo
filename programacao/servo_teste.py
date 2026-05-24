import RPi.GPIO as GPIO
import time

# 1. Configuração dos pinos do Raspberry Pi
GPIO.setmode(GPIO.BCM)  # Usa o mapeamento Broadcom (GPIO xx) e não o número físico do pino
GPIO_PINO_SERVO = 12    # GPIO 18 (Pino físico 12)

GPIO.setup(GPIO_PINO_SERVO, GPIO.OUT)

# 2. Configura o PWM no pino escolhido para a frequência de 50Hz (padrão de servos)
servo = GPIO.PWM(GPIO_PINO_SERVO, 50) 

# Inicia o PWM com o servo na posição de 0 graus (Ciclo de trabalho de 2.5%)
servo.start(2.5)
print("Servo iniciado na posição 0°")
time.sleep(1)

def definir_angulo(angulo):
    if angulo < 0 or angulo > 180:
        print("Ângulo inválido!")
        return
        
    duty_cycle = 2.5 + (angulo / 18.0)
    servo.ChangeDutyCycle(duty_cycle)
    print(f"Movendo para {angulo}°")
    
    # IMPORTANTE: Espera o servo chegar lá (ex: 0.5 segundos)
    time.sleep(1)
    
    # O TRUQUE: Envia sinal ZERO. Isso "desliga" o motor e mata a trepidação!
    servo.ChangeDutyCycle(0)

# 3. Loop principal de teste
try:
    while True:
        # Move o servo para as posições principais
        definir_angulo(0)     # 0 Graus
        time.sleep(1)
        
        definir_angulo(90)    # 90 Graus (Centro)
        time.sleep(1)
        
        definir_angulo(180)   # 180 Graus
        time.sleep(1)
        
        definir_angulo(90)    # Volta para o centro
        time.sleep(1)

except KeyboardInterrupt:
    # Captura o Ctrl+C para encerrar o programa de forma segura
    print("\nInterrompido pelo usuário. Finalizando...")

finally:
    # 4. Limpeza e encerramento dos pinos (Muito importante!)
    servo.stop()
    GPIO.cleanup()
    print("Pinos GPIO liberados com sucesso.")