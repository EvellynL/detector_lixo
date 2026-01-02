from ultralytics import YOLO    
import cv2
import numpy as np


model = YOLO('runs\detect\\train2\weights\\best_250epochs.pt')

results = model('programacao\image.png')
r = results[0]

classe_id = int(r.boxes.cls[0].item())
class_name = r.names[classe_id]

print(class_name)


# results[0].show()