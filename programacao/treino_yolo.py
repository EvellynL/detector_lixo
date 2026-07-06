from ultralytics import YOLO

def main():
    model = YOLO("yolov8n.pt")
    model.train(
        data="programacao\dataset_yolo-640\data.yaml",
        epochs=150,
        imgsz=640,
        batch=10,
        workers=8,
        lr0= 0.01,
        cos_lr= True,
        iou= 0.5,
        optimizer = 'AdamW',
        lrf = 0.1,
        weight_decay=0.0005,
       
       ##augmeniton

       fliplr=0.5,
       flipud=0.0,

       degrees=15,

       translate=0.05,
       scale=0.10,
       shear=0.0,
       perspective=0.0,

       hsv_h=0.015,
       hsv_s=0.4,
       hsv_v=0.2,

       mosaic=0.5,
       mixup=0.0,
       copy_paste=0.0

    )


if __name__ == "__main__":
    main()
