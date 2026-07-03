from ultralytics import YOLO

def main():
    model = YOLO("yolov8n.pt")
    model.train(
        data="programacao\dataset_yolov8_640\data.yaml",
        epochs=300,
        imgsz=640,
        batch=10,
        workers=8,
        lr0= 0.01,
        cos_lr= True,
        iou= 0.5,
        optimizer = 'AdamW',
        lrf = 0.1,
        weight_decay=0.0005,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        mosaic=1.0,
        mixup=0.1,

    )


if __name__ == "__main__":
    main()
