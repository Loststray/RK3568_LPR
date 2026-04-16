import argparse
import math
import os
import time
from pathlib import Path

from ultralytics import YOLO

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DEFAULT_WEIGHTS = "YOLO/runs/detect/runs/train/yolo26_ccpd2/weights/best.pt"
DEFAULT_SOURCE = "dataset/CCPD_test"


def collect_image_files(source):
    source_path = Path(source)
    if source_path.is_file():
        return [str(source_path)]
    if not source_path.is_dir():
        raise ValueError(f"source 不存在或不是目录: {source}")

    image_files = []
    for root, _, files in os.walk(source_path):
        for filename in files:
            if Path(filename).suffix.lower() in IMAGE_EXTENSIONS:
                image_files.append(str(Path(root) / filename))

    image_files.sort()
    return image_files


def chunk_list(items, batch_size):
    for i in range(0, len(items), batch_size):
        yield items[i:i + batch_size]


def get_time(
    model,
    source=DEFAULT_SOURCE,
    batch_size=1,
    imgsz=640,
    conf=0.25,
    iou=0.7,
    warmup=1,
):
    """
    统计推理时间:
    - batch_size == 1: 输出每张图片推理时间
    - batch_size > 1: 输出每个 batch 推理时间
    """
    image_files = collect_image_files(source)
    if not image_files:
        print(f"未找到可用图片: {source}")
        return

    batch_size = max(1, int(batch_size))
    warmup = max(0, int(warmup))

    print(f"找到 {len(image_files)} 张图片, batch_size={batch_size}")

    # 预热，避免首轮推理包含额外初始化开销
    warmup_samples = image_files[: min(len(image_files), warmup)]
    for sample in warmup_samples:
        _ = model.predict(source=sample, imgsz=imgsz, conf=conf, iou=iou, verbose=False)

    total_time = 0.0
    total_images = 0
    total_batches = int(math.ceil(len(image_files) / batch_size))

    for batch_idx, batch_paths in enumerate(chunk_list(image_files, batch_size), start=1):
        t1 = time.perf_counter()
        _ = model.predict(source=batch_paths, imgsz=imgsz, conf=conf, iou=iou, verbose=False)
        elapsed = time.perf_counter() - t1

        batch_count = len(batch_paths)
        total_time += elapsed
        total_images += batch_count

        if batch_size == 1:
            print(f"[{batch_idx}/{total_batches}] {batch_paths[0]}: {elapsed * 1000:.2f} ms")
        else:
            print(
                f"[{batch_idx}/{total_batches}] batch={batch_count}: "
                f"{elapsed * 1000:.2f} ms ({elapsed / batch_count * 1000:.2f} ms/img)"
            )

    avg_ms_per_img = (total_time / total_images * 1000.0) if total_images else 0.0
    fps = (total_images / total_time) if total_time > 0 else 0.0
    print("-" * 70)
    print(f"总图片数: {total_images}")
    print(f"总耗时  : {total_time:.4f} s")
    print(f"平均耗时: {avg_ms_per_img:.2f} ms/img")
    print(f"吞吐(FPS): {fps:.2f}")


def build_parser():
    parser = argparse.ArgumentParser(description="YOLO 推理时间评测")
    parser.add_argument("--weights", default=DEFAULT_WEIGHTS, help="模型权重路径")
    parser.add_argument("--source", default=DEFAULT_SOURCE, help="图片路径或图片目录")
    parser.add_argument("--batch-size", type=int, default=1, help="batch 大小")
    parser.add_argument("--imgsz", type=int, default=640, help="推理尺寸")
    parser.add_argument("--conf", type=float, default=0.5, help="置信度阈值")
    parser.add_argument("--iou", type=float, default=0.7, help="NMS IoU 阈值")
    parser.add_argument("--warmup", type=int, default=1, help="预热轮次")
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    model = YOLO(args.weights)
    get_time(
        model=model,
        source=args.source,
        batch_size=args.batch_size,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        warmup=args.warmup,
    )
