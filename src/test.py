import argparse

import cv2

from image_process import DEFAULT_IMAGE_ROOT, collect_image_files, process_images, resolve_local_path
from video_process import DEFAULT_VIDEO_SOURCE, process_video_stream
from dataset_process import test_dataset

def get_parser():
    parser = argparse.ArgumentParser(description="车牌检测与识别（单图/视频流关键帧）")
    parser.add_argument("--mode", choices=["image", "video", "dataset"], default="image", help="运行模式")
    parser.add_argument("--source", default=None, help="image 模式传图片根目录，video 模式传视频路径或摄像头编号")
    parser.add_argument("--conf-thres", type=float, default=0.5, help="YOLO 检测阈值")
    parser.add_argument("--frame-interval", type=int, default=3, help="视频模式：每隔 N 帧触发一次关键帧判定")
    parser.add_argument("--motion-threshold", type=float, default=6.0, help="视频模式：帧间差分阈值")
    parser.add_argument("--max-skip", type=int, default=30, help="视频模式：最长跳过帧数，超过后强制做一次识别")
    parser.add_argument("--display-scale", type=float, default=1.0, help="显示缩放比例")
    parser.add_argument("--save-keyframes", default=None, help="视频模式：可选，关键帧保存目录")
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
            )
        elif args.mode == 'image':
            image_root = args.source if args.source else DEFAULT_IMAGE_ROOT
            run_image_mode(image_root=image_root, conf_thres=args.conf_thres)
        else:
            image_root = args.source if args.source else DEFAULT_IMAGE_ROOT
            test_dataset(image_root=image_root,conf_thresh=args.conf_thres)
    finally:
        cv2.destroyAllWindows()
