import argparse
import os
import time

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

from LPRNet.model.LPRNet import build_lprnet

# 配置字符集
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
DEFAULT_VIDEO_SOURCE = "0"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
yolo_weight = os.path.join(BASE_DIR, "YOLO/weights/best.pt")
lpr_weight = os.path.join(BASE_DIR, "LPRNet/weights/Final_LPRNet_model.pth")

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
            for i, det in enumerate(detections, 1):
                print(f"  车牌 : {det['plate']} (置信度: {det['conf']:.2f})")

        vis_img = draw_detections(img_ori, detections)
        cv2.imshow("Recognition Result", vis_img)
        key = cv2.waitKey(0)
        if key in (27, ord("q"), ord("Q")):
            break


def collect_image_files(image_root):
    resolved_root = resolve_local_path(image_root)
    if not os.path.isdir(resolved_root):
        raise ValueError(f"图片根目录不存在或不是目录: {resolved_root}")

    image_files = []
    for root, _, files in os.walk(resolved_root):
        for filename in files:
            ext = os.path.splitext(filename)[1].lower()
            if ext in IMAGE_EXTENSIONS:
                image_files.append(os.path.join(root, filename))

    image_files.sort()
    return image_files


def is_key_frame(curr_gray, prev_gray, motion_threshold):
    if prev_gray is None:
        return True, 0.0
    motion_score = float(np.mean(cv2.absdiff(curr_gray, prev_gray)))
    return motion_score >= motion_threshold, motion_score


def parse_video_source(source):
    source = str(source)
    if source.isdigit():
        return int(source)
    return resolve_local_path(source)


def process_video_stream(
    source,
    conf_thres=0.25,
    frame_interval=3,
    motion_threshold=6.0,
    max_skip=30,
    display_scale=1.0,
    save_keyframes_dir=None,
):
    video_source = parse_video_source(source)
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频源: {source}")

    if save_keyframes_dir:
        os.makedirs(save_keyframes_dir, exist_ok=True)

    frame_interval = max(frame_interval, 1)
    max_skip = max(max_skip, 1)
    display_scale = max(display_scale, 0.1)

    prev_gray = None
    frame_idx = 0
    last_key_frame_idx = -max_skip
    last_detections = []

    prev_time = time.time()
    fps = 0.0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame_idx += 1
        curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        motion_trigger, motion_score = is_key_frame(curr_gray, prev_gray, motion_threshold)
        prev_gray = curr_gray

        interval_trigger = (frame_idx % frame_interval == 0)
        timeout_trigger = (frame_idx - last_key_frame_idx >= max_skip)
        key_frame = (interval_trigger and motion_trigger) or timeout_trigger

        if key_frame:
            last_detections = detect_and_recognize(frame, conf_thres=conf_thres)
            last_key_frame_idx = frame_idx
            if save_keyframes_dir:
                keyframe_name = f"keyframe_{frame_idx:06d}.jpg"
                cv2.imwrite(os.path.join(save_keyframes_dir, keyframe_name), frame)

        display = draw_detections(frame, last_detections)
        now = time.time()
        dt = now - prev_time
        prev_time = now
        if dt > 0:
            fps = fps * 0.9 + (1.0 / dt) * 0.1

        status_text = f"frame={frame_idx} key={int(key_frame)} motion={motion_score:.2f} fps={fps:.1f}"
        display = cv2ImgAddText(display, status_text, (10, 10), textColor=(255, 255, 0), textSize=20)

        if abs(display_scale - 1.0) > 1e-6:
            display = cv2.resize(display, None, fx=display_scale, fy=display_scale)

        cv2.imshow("Video LPR (q/ESC to quit)", display)
        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord("q"), ord("Q")):
            break

    cap.release()
    cv2.destroyAllWindows()


def get_parser():
    parser = argparse.ArgumentParser(description="车牌检测与识别（单图/视频流关键帧）")
    parser.add_argument("--mode", choices=["image", "video"], default="image", help="运行模式")
    parser.add_argument("--source", default=None, help="image 模式传图片根目录，video 模式传视频路径或摄像头编号")
    parser.add_argument("--conf-thres", type=float, default=0.25, help="YOLO 检测阈值")
    parser.add_argument("--frame-interval", type=int, default=3, help="视频模式：每隔 N 帧触发一次关键帧判定")
    parser.add_argument("--motion-threshold", type=float, default=6.0, help="视频模式：帧间差分阈值")
    parser.add_argument("--max-skip", type=int, default=30, help="视频模式：最长跳过帧数，超过后强制做一次识别")
    parser.add_argument("--display-scale", type=float, default=1.0, help="显示缩放比例")
    parser.add_argument("--save-keyframes", default=None, help="视频模式：可选，关键帧保存目录")
    return parser


if __name__ == "__main__":
    args = get_parser().parse_args()
    try:
        if args.mode == "image":
            image_root = args.source if args.source else DEFAULT_IMAGE_ROOT
            image_paths = collect_image_files(image_root)
            if not image_paths:
                print(f"目录内未找到图片: {resolve_local_path(image_root)}")
            else:
                print(f"共找到 {len(image_paths)} 张图片，开始识别...")
                process_images(image_paths, conf_thres=args.conf_thres)
        else:
            source = args.source if args.source else DEFAULT_VIDEO_SOURCE
            process_video_stream(
                source=source,
                conf_thres=args.conf_thres,
                frame_interval=args.frame_interval,
                motion_threshold=args.motion_threshold,
                max_skip=args.max_skip,
                display_scale=args.display_scale,
                save_keyframes_dir=args.save_keyframes,
            )
    finally:
        cv2.destroyAllWindows()
