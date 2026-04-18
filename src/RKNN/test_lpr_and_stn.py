import sys
import time
from pathlib import Path

import cv2
import numpy as np
from rknn.api import RKNN

BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "dataset_lpr.txt"
DEFAULT_QUANT = True

INPUT_SIZE = (94, 24)  # (w, h)
MODEL_MEAN = [[127.5, 127.5, 127.5]]
MODEL_STD = [[128.0, 128.0, 128.0]]
NORM_MEAN = 127.5
NORM_SCALE = 0.0078125  # 1 / 128

DEFAULT_STN_RKNN_PATH = BASE_DIR / "weights" / "stnet.rknn"
DEFAULT_LPR_RKNN_PATH = BASE_DIR / "weights" / "lprnet.rknn"

CHARS = [
    "京",
    "沪",
    "津",
    "渝",
    "冀",
    "晋",
    "蒙",
    "辽",
    "吉",
    "黑",
    "苏",
    "浙",
    "皖",
    "闽",
    "赣",
    "鲁",
    "豫",
    "鄂",
    "湘",
    "粤",
    "桂",
    "琼",
    "川",
    "贵",
    "云",
    "藏",
    "陕",
    "甘",
    "青",
    "宁",
    "新",
    "0",
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9",
    "A",
    "B",
    "C",
    "D",
    "E",
    "F",
    "G",
    "H",
    "J",
    "K",
    "L",
    "M",
    "N",
    "P",
    "Q",
    "R",
    "S",
    "T",
    "U",
    "V",
    "W",
    "X",
    "Y",
    "Z",
    "I",
    "O",
    "-",
]


def parse_arg():
    if len(sys.argv) < 4:
        print(
            "Usage: python3 {} stn_onnx_model_path lpr_onnx_model_path [platform] [dtype(optional)] "
            "[stn_rknn_path(optional)] [lpr_rknn_path(optional)]".format(sys.argv[0])
        )
        print(
            "       platform choose from [rk3562, rk3566, rk3568, rk3576, rk3588, rv1126b, rv1109, rv1126, rk1808]"
        )
        print("       dtype choose from [i8, fp] for [rk3562, rk3566, rk3568, rk3576, rk3588, rv1126b]")
        print("       dtype choose from [u8, fp] for [rv1109, rv1126, rk1808]")
        exit(1)

    stn_model_path = sys.argv[1]
    lpr_model_path = sys.argv[2]
    platform = sys.argv[3]

    do_quant = DEFAULT_QUANT
    if len(sys.argv) > 4:
        model_type = sys.argv[4]
        if model_type not in ["i8", "u8", "fp"]:
            print("ERROR: Invalid model type: {}".format(model_type))
            exit(1)
        elif model_type in ["i8", "u8"]:
            do_quant = True
        else:
            do_quant = False

    stn_rknn_path = Path(sys.argv[5]) if len(sys.argv) > 5 else DEFAULT_STN_RKNN_PATH
    lpr_rknn_path = Path(sys.argv[6]) if len(sys.argv) > 6 else DEFAULT_LPR_RKNN_PATH

    return stn_model_path, lpr_model_path, platform, do_quant, stn_rknn_path, lpr_rknn_path


def load_image_paths(dataset_file):
    if not dataset_file.is_file():
        raise FileNotFoundError(f"Dataset file not found: {dataset_file}")

    paths = []
    with dataset_file.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            path = Path(line)
            if not path.is_absolute():
                path = (BASE_DIR.parent / line).resolve()
            paths.append(path)
    return paths


def build_model(model_path, platform, do_quant, model_name):
    model_path = str(model_path)
    rknn = RKNN(verbose=False)

    print(f"--> Config {model_name} model")
    rknn.config(mean_values=MODEL_MEAN, std_values=MODEL_STD, target_platform=platform)
    print("done")

    print(f"--> Loading {model_name} model")
    ret = rknn.load_onnx(model=model_path)
    if ret != 0:
        print(f"Load {model_name} model failed!")
        exit(ret)
    print("done")

    print(f"--> Building {model_name} model")
    ret = rknn.build(do_quantization=do_quant, dataset=str(DATASET_PATH))
    if ret != 0:
        print(f"Build {model_name} model failed!")
        exit(ret)
    print("done")

    print(f"--> Init {model_name} runtime")
    ret = rknn.init_runtime(perf_debug=True)
    if ret != 0:
        print(f"Init {model_name} runtime failed!")
        exit(ret)
    print("done")
    return rknn


def preprocess_input(image):
    resized = cv2.resize(image, INPUT_SIZE)
    chw = np.transpose(resized, (2, 0, 1))
    tensor = np.expand_dims(chw, axis=0)
    return np.ascontiguousarray(tensor, dtype=np.float32)


def prepare_lpr_input(stn_output):
    output = np.asarray(stn_output)
    if output.ndim == 3:
        output = np.expand_dims(output, axis=0)
    if output.ndim != 4:
        raise ValueError(f"Unexpected STN output shape: {output.shape}")

    # STN output can be NCHW or NHWC depending on conversion settings.
    if output.shape[1] == 3:
        output = np.ascontiguousarray(output, dtype=np.float32)
    elif output.shape[-1] == 3:
        output = np.transpose(output, (0, 3, 1, 2))
        output = np.ascontiguousarray(output, dtype=np.float32)
    else:
        raise ValueError(f"Unexpected STN output shape: {output.shape}")

    # STN output is in normalized space:
    # normalized = (pixel - 127.5) * 0.0078125.
    # Convert it back to pixel space so LPR preprocess applies exactly once.
    output = output / NORM_SCALE + NORM_MEAN
    output = np.clip(output, 0.0, 255.0)
    return np.ascontiguousarray(output, dtype=np.float32)


def ctc_greedy_decode(indices):
    blank_idx = len(CHARS) - 1
    decoded = []
    prev = None
    for idx in indices:
        idx = int(idx)
        if idx == blank_idx:
            prev = idx
            continue
        if prev == idx:
            continue
        if 0 <= idx < len(CHARS):
            decoded.append(CHARS[idx])
        prev = idx
    return "".join(decoded)


def decode_lpr_logits(logits):
    logits = np.asarray(logits)
    if logits.ndim == 3:
        logits = logits[0]
    elif logits.ndim > 3:
        logits = np.squeeze(logits)

    if logits.ndim != 2:
        raise ValueError(f"Unexpected LPR output shape: {logits.shape}")

    if logits.shape[0] == len(CHARS):
        indices = np.argmax(logits, axis=0)
    elif logits.shape[1] == len(CHARS):
        indices = np.argmax(logits, axis=1)
    else:
        # Fallback: prefer class dimension as the larger axis.
        if logits.shape[0] >= logits.shape[1]:
            indices = np.argmax(logits, axis=0)
        else:
            indices = np.argmax(logits, axis=1)
    return ctc_greedy_decode(indices)


if __name__ == "__main__":
    stn_model_path, lpr_model_path, platform, do_quant, stn_rknn_path, lpr_rknn_path = parse_arg()

    stn_rknn = None
    lpr_rknn = None
    try:
        stn_rknn = build_model(stn_model_path, platform, do_quant, "STN")
        lpr_rknn = build_model(lpr_model_path, platform, do_quant, "LPR")

        image_paths = load_image_paths(DATASET_PATH)
        if not image_paths:
            raise RuntimeError(f"No image paths found in {DATASET_PATH}")

        stn_total_ms = 0.0
        lpr_total_ms = 0.0
        total_ms = 0.0
        valid_count = 0
        match_count = 0

        print(f"--> Testing with {len(image_paths)} images from {DATASET_PATH}")

        for idx, image_path in enumerate(image_paths, start=1):
            if not image_path.is_file():
                print(f"[{idx}/{len(image_paths)}] skip missing file: {image_path}")
                continue

            image = cv2.imread(str(image_path))
            if image is None:
                print(f"[{idx}/{len(image_paths)}] skip unreadable image: {image_path}")
                continue

            input_tensor = preprocess_input(image)
            if idx == 1:
                print(f"Input tensor shape (STN): {input_tensor.shape}")

            t0 = time.perf_counter()
            stn_outputs = stn_rknn.inference(inputs=[input_tensor], data_format=["nchw"])
            stn_ms = (time.perf_counter() - t0) * 1000.0

            if not stn_outputs:
                print(f"[{idx}/{len(image_paths)}] STN output is empty: {image_path.name}")
                continue

            stn_input_for_lpr = prepare_lpr_input(stn_outputs[0])

            t1 = time.perf_counter()
            lpr_outputs = lpr_rknn.inference(inputs=[stn_input_for_lpr], data_format=["nchw"])
            lpr_ms = (time.perf_counter() - t1) * 1000.0

            if not lpr_outputs:
                print(f"[{idx}/{len(image_paths)}] LPR output is empty: {image_path.name}")
                continue

            plate = decode_lpr_logits(lpr_outputs[0])
            gt_plate = image_path.stem
            is_match = plate == gt_plate

            valid_count += 1
            if is_match:
                match_count += 1

            infer_total = stn_ms + lpr_ms
            stn_total_ms += stn_ms
            lpr_total_ms += lpr_ms
            total_ms += infer_total

            print(
                f"[{idx}/{len(image_paths)}] {image_path.name} "
                f"stn={stn_ms:.2f} ms lpr={lpr_ms:.2f} ms total={infer_total:.2f} ms "
                f"pred={plate} gt={gt_plate} match={'Y' if is_match else 'N'}"
            )

        if valid_count > 0:
            avg_stn = stn_total_ms / valid_count
            avg_lpr = lpr_total_ms / valid_count
            avg_total = total_ms / valid_count
            fps = 1000.0 / avg_total if avg_total > 0 else 0.0
            acc = match_count / valid_count

            print(f"Average STN inference: {avg_stn:.2f} ms/image")
            print(f"Average LPR inference: {avg_lpr:.2f} ms/image")
            print(f"Average end-to-end inference: {avg_total:.2f} ms/image, FPS={fps:.2f}")
            print(f"Accuracy (filename GT): {acc:.4f} ({match_count}/{valid_count})")
        else:
            print("No valid images were processed.")

        print("--> Export STN rknn model")
        ret = stn_rknn.export_rknn(str(stn_rknn_path))
        if ret != 0:
            print("Export STN rknn model failed!")
            exit(ret)
        print("done")

        print("--> Export LPR rknn model")
        ret = lpr_rknn.export_rknn(str(lpr_rknn_path))
        if ret != 0:
            print("Export LPR rknn model failed!")
            exit(ret)
        print("done")
    finally:
        if stn_rknn is not None:
            stn_rknn.release()
        if lpr_rknn is not None:
            lpr_rknn.release()
