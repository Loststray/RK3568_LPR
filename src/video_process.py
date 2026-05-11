import os
import shutil
import subprocess
import time
from collections import Counter
from urllib.parse import urlparse

import cv2
import numpy as np

from image_process import cv2ImgAddText, detect_and_recognize, draw_detections, resolve_local_path

DEFAULT_VIDEO_SOURCE = "0"
NETWORK_STREAM_SCHEMES = {"rtsp", "rtmp", "http", "https", "udp", "tcp"}
DEFAULT_VIDEO_OUTPUT_DIR = "video_outputs"
DEFAULT_VIDEO_OUTPUT_FPS = 25.0
H264_ENCODER_CANDIDATES = ("libx264", "libopenh264", "h264_v4l2m2m")


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


def build_output_video_path(video_source):
    timestamp = time.strftime("%Y%m%d_%H%M%S")

    if isinstance(video_source, int):
        os.makedirs(DEFAULT_VIDEO_OUTPUT_DIR, exist_ok=True)
        return os.path.join(DEFAULT_VIDEO_OUTPUT_DIR, f"camera_{video_source}_{timestamp}.mp4")

    scheme = urlparse(video_source).scheme.lower()
    if scheme in NETWORK_STREAM_SCHEMES:
        os.makedirs(DEFAULT_VIDEO_OUTPUT_DIR, exist_ok=True)
        return os.path.join(DEFAULT_VIDEO_OUTPUT_DIR, f"{scheme}_stream_{timestamp}.mp4")

    source_dir = os.path.dirname(os.path.abspath(video_source)) or "."
    source_name = os.path.splitext(os.path.basename(video_source))[0] or "video"
    output_path = os.path.join(source_dir, f"{source_name}_lpr.mp4")
    if not os.path.exists(output_path):
        return output_path

    return os.path.join(source_dir, f"{source_name}_lpr_{timestamp}.mp4")


def build_temp_output_video_path(output_video_path):
    root, ext = os.path.splitext(output_video_path)
    return f"{root}.tmp{ext or '.mp4'}"


def resolve_h264_encoder():
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("未找到 ffmpeg，无法将视频重新编码为 H.264")

    try:
        result = subprocess.run(
            ["ffmpeg", "-hide_banner", "-encoders"],
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr.strip() if exc.stderr else ""
        raise RuntimeError(f"无法查询 ffmpeg 编码器列表: {stderr or exc}") from exc

    encoder_text = "\n".join(filter(None, [result.stdout, result.stderr]))
    for encoder in H264_ENCODER_CANDIDATES:
        if encoder in encoder_text:
            return encoder

    raise RuntimeError("当前 ffmpeg 未提供可用的 H.264 编码器")


def reencode_video_to_h264(input_path, output_path, encoder=None):
    encoder = encoder or resolve_h264_encoder()

    command = [
        "ffmpeg",
        "-y",
        "-i",
        input_path,
        "-c:v",
        encoder,
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        output_path,
    ]

    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr.strip() if exc.stderr else ""
        raise RuntimeError(f"ffmpeg H.264 转码失败: {stderr or exc}") from exc
    return encoder


def finalize_output_video(temp_output_video_path, output_video_path):
    encoder = None
    h264_ready = False

    try:
        encoder = resolve_h264_encoder()
        print(f"开始使用 ffmpeg 转码为 H.264 ({encoder}): {output_video_path}")
        reencode_video_to_h264(temp_output_video_path, output_video_path, encoder=encoder)
        h264_ready = True
        print(f"检测结果视频已保存: {output_video_path}")
    except RuntimeError as exc:
        if os.path.exists(output_video_path):
            os.remove(output_video_path)
        os.replace(temp_output_video_path, output_video_path)
        print(f"H.264 转码不可用，保留原始 MP4 输出: {exc}")
        print(f"检测结果视频已保存: {output_video_path}")
        return output_video_path, encoder, h264_ready, str(exc)

    if os.path.exists(temp_output_video_path):
        os.remove(temp_output_video_path)
    return output_video_path, encoder, h264_ready, None


def process_video_stream(
    source,
    conf_thres=0.25,
    frame_interval=3,
    motion_threshold=6.0,
    max_skip=30,
    display_scale=1.0,
    save_keyframes_dir=None,
    save_video=False,
    output_path=None,
    return_details=False,
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
    writer = None
    output_video_path = output_path if output_path else build_output_video_path(video_source) if save_video else None
    if output_video_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_video_path)), exist_ok=True)
    temp_output_video_path = build_temp_output_video_path(output_video_path) if save_video else None
    output_fps = cap.get(cv2.CAP_PROP_FPS)
    if output_fps <= 0 or np.isnan(output_fps):
        output_fps = DEFAULT_VIDEO_OUTPUT_FPS

    prev_gray = None
    frame_idx = 0
    last_key_frame_idx = -max_skip
    last_detections = []

    prev_time = time.time()
    fps = 0.0
    recognized_plates = Counter()
    transcode_warning = None
    encoder = None
    h264_ready = False

    try:
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
                for det in last_detections:
                    plate = det["plate"].strip() if det["plate"] else ""
                    if plate:
                        recognized_plates[plate] += 1
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

            if save_video:
                if writer is None:
                    height, width = display.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    writer = cv2.VideoWriter(temp_output_video_path, fourcc, output_fps, (width, height))
                    if not writer.isOpened():
                        raise RuntimeError(f"无法创建临时输出视频: {temp_output_video_path}")
                    print(f"开始保存临时检测结果视频: {temp_output_video_path}")
                writer.write(display)
            else:
                cv2.imshow("Video LPR (q/ESC to quit)", display)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q"), ord("Q")):
                    break
    finally:
        cap.release()
        if writer is not None:
            writer.release()

    print(f"视频处理结束，共处理 {frame_idx} 帧")
    if writer is not None:
        output_video_path, encoder, h264_ready, transcode_warning = finalize_output_video(
            temp_output_video_path,
            output_video_path,
        )
    else:
        output_video_path = None

    recognized_plate_list = [
        {"plate": plate, "count": count}
        for plate, count in recognized_plates.most_common()
    ]
    if return_details:
        return {
            "output_path": output_video_path,
            "frames_processed": frame_idx,
            "recognized_plates": recognized_plate_list,
            "h264_ready": h264_ready,
            "encoder": encoder,
            "transcode_warning": transcode_warning,
        }
    return output_video_path
