import os

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

from LPRNet.model.LPRNet import build_lprnet
from LPRNet.model.STNet import build_STNet

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

EXPOSURE_PREPROCESS_DEFAULTS = {
    "judge": {
        "low_clip_pixel": 10,      # 小于等于该亮度算暗部截断
        "high_clip_pixel": 245,    # 大于等于该亮度算高光截断
        "low_clip_thr": 0.08,      # 暗部截断比例阈值
        "high_clip_thr": 0.015,     # 高光截断比例阈值
        "low_p50_thr": 90,         # 中位亮度过低阈值
        "high_p50_thr": 110,       # 中位亮度过高阈值
    },
    "transform": {
        "过度曝光": {"alpha": 0.80, "beta": -50.0, "gamma": 2},
        "曝光不足": {"alpha": 1.25, "beta": 20.0, "gamma": 0.65},
        "曝光正常": {"alpha": 1.00, "beta": 0.0, "gamma": 1.00},
    },
    "max_iters": 2,               # 自动纠偏最大迭代次数
}


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
yolo_weight = os.path.join(BASE_DIR, "YOLO/weights/best.pt")
lpr_weight = os.path.join(BASE_DIR, "LPRNet/weights/Final_LPRNet_model.pth")
stn_weight = os.path.join(BASE_DIR, "LPRNet/weights/Final_STNet_model.pth")

yolo_model = YOLO(yolo_weight)

lpr_model = build_lprnet(lpr_max_len=8, phase=False, class_num=len(CHARS), dropout_rate=0)
lpr_model.load_state_dict(torch.load(lpr_weight, map_location=device))
lpr_model.to(device)
lpr_model.eval()

stn_model = build_STNet(False)
stn_model.load_state_dict(torch.load(stn_weight,map_location=device))
stn_model.to(device)
stn_model.eval()


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
        tmp_img = stn_model(tmp_img)
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

def transform_img(img, alpha=1.0, beta=0.0, gamma=1.0):
    if img is None:
        raise ValueError("img 不能为空")

    img_float = img.astype(np.float32)
    linear_img = np.clip(img_float * float(alpha) + float(beta), 0.0, 255.0)

    safe_gamma = max(float(gamma), 1e-8)
    gamma_img = np.power(linear_img / 255.0, safe_gamma) * 255.0
    return np.clip(gamma_img, 0.0, 255.0).astype(np.uint8)


def judge_exposure(
    img_bgr,
    low_clip_pixel=10,
    high_clip_pixel=245,
    low_clip_thr=0.08,
    high_clip_thr=0.02,
    low_p50_thr=90,
    high_p50_thr=160,
):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    low_clip = np.mean(gray <= low_clip_pixel)
    high_clip = np.mean(gray >= high_clip_pixel)
    p1, p50, p99 = np.percentile(gray, [1, 50, 99])

    print(f"high={high_clip},p50={p50}\n")
    if high_clip > high_clip_thr and p50 > high_p50_thr:
        return "过度曝光", {
            "low_clip": float(low_clip), "high_clip": float(high_clip),
            "p1": float(p1), "p50": float(p50), "p99": float(p99)
        }
    if low_clip > low_clip_thr and p50 < low_p50_thr:
        return "曝光不足", {
            "low_clip": float(low_clip), "high_clip": float(high_clip),
            "p1": float(p1), "p50": float(p50), "p99": float(p99)
        }

    return "曝光正常", {
        "low_clip": float(low_clip), "high_clip": float(high_clip),
        "p1": float(p1), "p50": float(p50), "p99": float(p99)
    }


def _merge_preprocess_params(params):
    cfg = {
        "judge": EXPOSURE_PREPROCESS_DEFAULTS["judge"].copy(),
        "transform": {
            key: value.copy()
            for key, value in EXPOSURE_PREPROCESS_DEFAULTS["transform"].items()
        },
        "max_iters": EXPOSURE_PREPROCESS_DEFAULTS["max_iters"],
    }

    if not params:
        return cfg

    if "judge" in params and isinstance(params["judge"], dict):
        cfg["judge"].update(params["judge"])

    if "transform" in params and isinstance(params["transform"], dict):
        for exposure_type, trans_cfg in params["transform"].items():
            if not isinstance(trans_cfg, dict):
                continue
            if exposure_type not in cfg["transform"]:
                cfg["transform"][exposure_type] = {}
            cfg["transform"][exposure_type].update(trans_cfg)

    if "max_iters" in params:
        cfg["max_iters"] = int(params["max_iters"])

    return cfg


def preprocess(img, alpha=None, beta=None, gamma=None, params=None):
    # 输入: img,alpha,beta,gamma
    # 逻辑: 自动判断曝光类型，再按类型套用 transform；也支持手工 alpha/beta/gamma 覆盖
    # 输出: new_img
    if img is None:
        raise ValueError("img 不能为空")

    # 手动模式：你显式给了参数时，优先按手动参数执行一次变换
    if alpha is not None or beta is not None or gamma is not None:
        manual_alpha = 1.0 if alpha is None else alpha
        manual_beta = 0.0 if beta is None else beta
        manual_gamma = 1.0 if gamma is None else gamma
        return transform_img(img, manual_alpha, manual_beta, manual_gamma)

    cfg = _merge_preprocess_params(params)
    out = img.copy()

    for _ in range(max(1, cfg["max_iters"])):
        exposure_type, _ = judge_exposure(out, **cfg["judge"])
        if exposure_type == "曝光正常":
            break

        trans_cfg = cfg["transform"].get(
            exposure_type,
            cfg["transform"]["曝光正常"],
        )
        out = transform_img(
            out,
            alpha=trans_cfg.get("alpha", 1.0),
            beta=trans_cfg.get("beta", 0.0),
            gamma=trans_cfg.get("gamma", 1.0),
        )

    return out
