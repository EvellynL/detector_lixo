import subprocess
import numpy as np
import cv2

cmd = [
    "rpicam-vid",
    "-t", "0",
    "--inline",
    "--codec", "yuv420",
    "-o", "-"
]

pipe = subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=10**8)

width, height = 640, 480

while True:
    raw = pipe.stdout.read(width * height * 3 // 2)
    if not raw:
        break

    frame = np.frombuffer(raw, dtype=np.uint8).reshape((height * 3 // 2, width))
    frame = cv2.cvtColor(frame, cv2.COLOR_YUV2BGR_I420)

    cv2.imshow("Camera", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

pipe.terminate()
cv2.destroyAllWindows()
