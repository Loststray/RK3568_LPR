from ultralytics import YOLO

model = YOLO("weights/best_local.pt")

results = model.train(
    data="../YOLO_Data/data.yaml",
    epochs=5,
    imgsz=640,
    batch=-1,
    workers=8,
    cache=False,
    device=0,
    project="runs/train",
    name="yolo26_ccpd",
    save=True
)