from ultralytics import YOLO

def main():
    model = YOLO("yolov8n.pt")
    model.train(
        data="programacao\dataset_yolov8\data.yaml",
        epochs=1000,
        imgsz=320,
        batch=16,
        workers=4,
        lr0= 0.005,
        cos_lr= True,
        iou= 0.6,
        optimizer = 'AdamW',
        weight_decay=0.0005,
        conf = 0.5

    )

if __name__ == "__main__":
    main()
