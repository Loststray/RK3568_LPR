import argparse
import os
from collections import Counter

import cv2

from image_process import (
    DEFAULT_IMAGE_ROOT,
    collect_image_files,
    detect_and_recognize,
    draw_detections,
    process_images,
    resolve_local_path,
)
from video_process import DEFAULT_VIDEO_SOURCE, process_video_stream
from dataset_process import test_dataset


def _ensure_output_dir(output_path):
    output_dir = os.path.dirname(os.path.abspath(output_path))
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)


def _summarize_plates(detections):
    counter = Counter()
    for det in detections:
        plate = det["plate"].strip() if det["plate"] else ""
        if plate:
            counter[plate] += 1
    return [{"plate": plate, "count": int(count)} for plate, count in counter.most_common()]


def _serialize_detections(detections):
    serialized = []
    for det in detections:
        serialized.append(
            {
                "bbox": [int(value) for value in det["bbox"]],
                "plate": det["plate"] or "",
                "conf": float(det["conf"]),
            }
        )
    return serialized


def run_image_file(image_path, output_path, conf_thres=0.5):
    resolved_image_path = resolve_local_path(image_path)
    image = cv2.imread(resolved_image_path)
    if image is None:
        raise RuntimeError(f"无法读取图片: {resolved_image_path}")

    detections = detect_and_recognize(image, conf_thres=conf_thres)
    visualized = draw_detections(image, detections)
    _ensure_output_dir(output_path)
    if not cv2.imwrite(output_path, visualized):
        raise RuntimeError(f"无法保存检测结果图片: {output_path}")

    return {
        "output_path": output_path,
        "detections": _serialize_detections(detections),
        "recognized_plates": _summarize_plates(detections),
    }


def run_video_file(
    video_path,
    output_path,
    conf_thres=0.5,
    frame_interval=3,
    motion_threshold=6.0,
    max_skip=30,
    display_scale=1.0,
):
    _ensure_output_dir(output_path)
    result = process_video_stream(
        source=video_path,
        conf_thres=conf_thres,
        frame_interval=frame_interval,
        motion_threshold=motion_threshold,
        max_skip=max_skip,
        display_scale=display_scale,
        save_video=True,
        output_path=output_path,
        return_details=True,
    )
    if not result or not result.get("output_path"):
        raise RuntimeError("视频检测未生成结果文件")
    return result


def get_parser():
    parser = argparse.ArgumentParser(description="车牌检测与识别（单图/视频流关键帧）")
    parser.add_argument("--mode", choices=["image", "video", "dataset"], default="image", help="运行模式")
    parser.add_argument("--source", default=None, help="image 模式传图片根目录，video 模式传视频路径或摄像头编号")
    parser.add_argument("--conf-thres", type=float, default=0.5, help="YOLO 检测阈值")
    parser.add_argument("--frame-interval", type=int, default=3, help="视频模式：每隔 N 帧触发一次关键帧判定")
    parser.add_argument("--motion-threshold", type=float, default=6.0, help="视频模式：帧间差分阈值")
    parser.add_argument("--max-skip", type=int, default=30, help="视频模式：最长跳过帧数，超过后强制做一次识别")
    parser.add_argument("--display-scale", type=float, default=1.0, help="视频模式：输出缩放比例")
    parser.add_argument("--save-keyframes", default=None, help="视频模式：可选，关键帧保存目录")
    parser.add_argument("--save-video", action="store_true", help="视频模式：保存带识别结果的视频到本地，不弹出显示窗口")
    return parser


def run_image_mode(image_root, conf_thres):
    image_paths = collect_image_files(image_root)
    if not image_paths:
        print(f"目录内未找到图片: {resolve_local_path(image_root)}")
        return

    print(f"共找到 {len(image_paths)} 张图片，开始识别...")
    process_images(image_paths, conf_thres=conf_thres)


if __name__ == "__main__":
    args = get_parser().parse_args()
    try:
        if args.mode == "video":
            source = args.source if args.source else DEFAULT_VIDEO_SOURCE
            process_video_stream(
                source=source,
                conf_thres=args.conf_thres,
                frame_interval=args.frame_interval,
                motion_threshold=args.motion_threshold,
                max_skip=args.max_skip,
                display_scale=args.display_scale,
                save_keyframes_dir=args.save_keyframes,
                save_video=args.save_video,
            )
        elif args.mode == 'image':
            image_root = args.source if args.source else DEFAULT_IMAGE_ROOT
            run_image_mode(image_root=image_root, conf_thres=args.conf_thres)
        else:
            image_root = args.source if args.source else DEFAULT_IMAGE_ROOT
            test_dataset(image_root=image_root,conf_thresh=args.conf_thres)
    finally:
        cv2.destroyAllWindows()
