import os

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

from LPRNet.model.LPRNet import build_lprnet

CHARS = [
    "京", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑",
    "苏", "浙", "皖", "闽", "赣", "鲁", "豫", "鄂", "湘", "粤",
    "桂", "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁",
    "新", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
    "A", "B", "C", "D", "E", "F", "G", "H", "J", "K", "L", "M",
    "N", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z",
    "I", "O", "-",
]

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_IMAGE_ROOT = "YOLO/test"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
]


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
yolo_weight = os.path.join(BASE_DIR, "YOLO/weights/best.pt")
lpr_weight = os.path.join(BASE_DIR, "LPRNet/weights/final.pth")

yolo_model = YOLO(yolo_weight)
lpr_model = build_lprnet(lpr_max_len=8, phase=False, class_num=len(CHARS), dropout_rate=0)
lpr_model.load_state_dict(torch.load(lpr_weight, map_location=device))
lpr_model.to(device)
lpr_model.eval()


def resolve_local_path(path):
    if os.path.isabs(path):
        return path
    candidate = os.path.join(BASE_DIR, path)
    if os.path.exists(candidate):
        return candidate
    return path


def cv2ImgAddText(img, text, pos, textColor=(0, 255, 0), textSize=20):
    """在图像上添加中文文本"""
    if isinstance(img, np.ndarray):
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img)

    font = None
    for font_path in FONT_CANDIDATES:
        if os.path.exists(font_path):
            try:
                font = ImageFont.truetype(font_path, textSize, encoding="utf-8")
                break
            except Exception:
                continue

    if font is None:
        font = ImageFont.load_default()

    draw.text(pos, text, textColor, font=font)
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)


def decode_res(preds, chars):
    """CTC Greedy 解码：去空白符、去重复"""
    if len(preds) == 0:
        return ""

    res = []
    blank_idx = len(chars) - 1

    for i in range(len(preds)):
        if preds[i] == blank_idx:
            continue
        if i > 0 and preds[i] == preds[i - 1]:
            continue
        res.append(chars[preds[i]])

    return "".join(res)


def recognize_plate(crop_img):
    if crop_img.size == 0:
        return ""

    tmp_img = cv2.resize(crop_img, (94, 24))
    tmp_img = tmp_img.astype("float32")
    tmp_img -= 127.5
    tmp_img *= 0.0078125
    tmp_img = np.transpose(tmp_img, (2, 0, 1))
    tmp_img = torch.from_numpy(tmp_img).unsqueeze(0).to(device)

    with torch.no_grad():
        preds = lpr_model(tmp_img)
        preds = preds.cpu().numpy()
        arg_max_preds = np.argmax(preds, axis=1)
        plate_no = decode_res(arg_max_preds[0], CHARS)
    return plate_no


def detect_and_recognize(frame, conf_thres=0.25):
    results = yolo_model(frame, verbose=False)
    h, w = frame.shape[:2]
    detections = []

    for box in results[0].boxes:
        conf = float(box.conf[0])
        if conf < conf_thres:
            continue

        xyxy = box.xyxy[0].cpu().numpy().astype(int)
        x1, y1, x2, y2 = xyxy
        x1 = max(0, min(x1, w - 1))
        y1 = max(0, min(y1, h - 1))
        x2 = max(0, min(x2, w))
        y2 = max(0, min(y2, h))
        if x2 <= x1 or y2 <= y1:
            continue

        crop_img = frame[y1:y2, x1:x2]
        plate_no = recognize_plate(crop_img)
        detections.append({"bbox": (x1, y1, x2, y2), "plate": plate_no, "conf": conf})

    return detections


def draw_detections(frame, detections):
    output = frame.copy()
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        plate_no = det["plate"] if det["plate"] else "未识别"
        conf = det["conf"]
        cv2.rectangle(output, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"{plate_no} ({conf:.2f})"
        text_y = max(0, y1 - 30)
        output = cv2ImgAddText(output, label, (x1, text_y), textColor=(0, 255, 0), textSize=24)
    return output


def process_images(img_paths, conf_thres=0.25):
    for img_path in img_paths:
        resolved = resolve_local_path(img_path)
        img_ori = cv2.imread(resolved)
        if img_ori is None:
            print(f"\n图片: {img_path}")
            print("  读取失败，请检查路径")
            continue

        detections = detect_and_recognize(img_ori, conf_thres=conf_thres)
        print(f"\n图片: {resolved}")
        if not detections:
            print("  未检测到车牌")
        else:
            for det in detections:
                print(f"  车牌 : {det['plate']} (置信度: {det['conf']:.2f})")

        vis_img = draw_detections(img_ori, detections)
        cv2.imshow("Recognition Result", vis_img)
        key = cv2.waitKey(0)
        if key in (27, ord("q"), ord("Q")):
            break


def collect_image_files(image_root, max_size=5000):
    resolved_root = resolve_local_path(image_root)
    if not os.path.isdir(resolved_root):
        raise ValueError(f"图片根目录不存在或不是目录: {resolved_root}")

    image_files = []
    for root, _, files in os.walk(resolved_root):
        for filename in files:
            if len(image_files) >= max_size:
                break
            ext = os.path.splitext(filename)[1].lower()
            if ext in IMAGE_EXTENSIONS:
                image_files.append(os.path.join(root, filename))

    image_files.sort()
    return image_files
