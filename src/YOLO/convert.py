import argparse
import shutil
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_WEIGHTS = BASE_DIR / "weights" / "best.pt"
DEFAULT_OUTPUT = BASE_DIR / "weights" / "best.onnx"


def parse_args():
    parser = argparse.ArgumentParser(description="将 YOLO 的 .pt 权重导出为 ONNX")
    parser.add_argument(
        "--weights",
        type=str,
        default=str(DEFAULT_WEIGHTS),
        help="输入 .pt 权重路径",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(DEFAULT_OUTPUT),
        help="输出 .onnx 文件路径",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        nargs="+",
        default=[640],
        help="导出输入尺寸，支持 `--imgsz 640` 或 `--imgsz 640 640`",
    )
    parser.add_argument("--batch", type=int, default=1, help="导出时使用的 batch size")
    parser.add_argument("--opset", type=int, default=12, help="ONNX opset 版本")
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="导出设备，如 cpu、0、0,1",
    )
    parser.add_argument(
        "--dynamic",
        action="store_true",
        help="启用动态 batch / shape",
    )
    parser.add_argument(
        "--half",
        action="store_true",
        help="使用 FP16 导出，仅建议在 CUDA 设备上使用",
    )
    parser.add_argument(
        "--simplify",
        action="store_true",
        help="导出后尝试简化 ONNX 图",
    )
    return parser.parse_args()


def resolve_path(path_str):
    path = Path(path_str)
    if path.is_absolute():
        return path
    return (Path.cwd() / path).resolve()


def normalize_imgsz(imgsz_values):
    if len(imgsz_values) == 1:
        return imgsz_values[0]
    if len(imgsz_values) == 2:
        return tuple(imgsz_values)
    raise ValueError("--imgsz 只支持传入 1 个或 2 个整数")


def export_onnx(weights_path, output_path, imgsz, batch, opset, device, dynamic, half, simplify):
    try:
        from ultralytics import YOLO
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "缺少依赖，请先安装 ultralytics 和 torch，例如：`pip install ultralytics torch onnx`"
        ) from exc

    if not weights_path.is_file():
        raise FileNotFoundError(f"权重文件不存在: {weights_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(weights_path))
    exported_path = model.export(
        format="onnx",
        imgsz=imgsz,
        batch=batch,
        opset=opset,
        device=device,
        dynamic=dynamic,
        half=half,
        simplify=simplify,
    )

    exported_path = Path(exported_path).resolve()
    target_path = output_path.resolve()

    if not exported_path.is_file():
        raise RuntimeError(f"ONNX 导出失败，未找到输出文件: {exported_path}")

    if exported_path != target_path:
        if target_path.exists():
            target_path.unlink()
        shutil.move(str(exported_path), str(target_path))

    if not target_path.is_file() or target_path.stat().st_size == 0:
        raise RuntimeError(f"ONNX 文件无效: {target_path}")

    return target_path


def main():
    args = parse_args()

    weights_path = resolve_path(args.weights)
    output_path = resolve_path(args.output)
    imgsz = normalize_imgsz(args.imgsz)

    exported_path = export_onnx(
        weights_path=weights_path,
        output_path=output_path,
        imgsz=imgsz,
        batch=args.batch,
        opset=args.opset,
        device=args.device,
        dynamic=args.dynamic,
        half=args.half,
        simplify=args.simplify,
    )

    print(f"ONNX 导出成功: {exported_path}")


if __name__ == "__main__":
    main()
