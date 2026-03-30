from ultralytics import YOLO
import os

import cv2

model = YOLO("weights/best.pt")
source = ["test/1.jpg", "test/2.jpg", "test/3.jpg"]

results = model(source)

for i, r in enumerate(results, 1):
    im = r.plot()  # 带检测框的图
    cv2.imshow(f"result_{i}", im)
    key = cv2.waitKey(0)  # 按任意键看下一张
    if key == 27:         # ESC 退出
        break

cv2.destroyAllWindows()
