import os
import time
from urllib.parse import urlparse

import cv2
import numpy as np

from image_process import cv2ImgAddText, detect_and_recognize, draw_detections, resolve_local_path

DEFAULT_VIDEO_SOURCE = "0"
NETWORK_STREAM_SCHEMES = {"rtsp", "rtmp", "http", "https", "udp", "tcp"}


def is_key_frame(curr_gray, prev_gray, motion_threshold):
    if prev_gray is None:
        return True, 0.0
    motion_score = float(np.mean(cv2.absdiff(curr_gray, prev_gray)))
    return motion_score >= motion_threshold, motion_score


def parse_video_source(source):
    if source is None:
        raise ValueError("视频源不能为空")

    source = str(source)
    if source.isdigit():
        return int(source)

    scheme = urlparse(source).scheme.lower()
    if scheme in NETWORK_STREAM_SCHEMES:
        return source

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
    if isinstance(video_source, str) and urlparse(video_source).scheme.lower() == "rtsp":
        cap = cv2.VideoCapture(video_source, cv2.CAP_FFMPEG)
    else:
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

        interval_trigger = frame_idx % frame_interval == 0
        timeout_trigger = frame_idx - last_key_frame_idx >= max_skip
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
