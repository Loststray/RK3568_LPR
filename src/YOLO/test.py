from ultralytics import YOLO

model = YOLO("yolo26n.pt")
results = model("images/image.png")
results[0].show()