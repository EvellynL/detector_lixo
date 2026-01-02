from ultralytics import YOLO

def main():
    model = YOLO("yolov8n.pt")
    model.train(
        data="programacao\dataset_yolov8\data.yaml",
        epochs=250,
        imgsz=416,
        batch=16,
        workers=0,
        lr0= 0.005,
        cos_lr= True,
        iou= 0.6
    )

if __name__ == "__main__":
    main()
