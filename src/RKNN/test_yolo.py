import sys
import time
from pathlib import Path

import cv2
import numpy as np
from rknn.api import RKNN
from copy import copy

BASE_DIR = Path(__file__).resolve().parent
QUANT_DATASET_PATH = BASE_DIR / "dataset.txt"
TEST_DATASET_PATH = BASE_DIR / "dataset_ccpd_test.txt"
DEFAULT_QUANT = True
INPUT_SIZE = (640, 640)  # (h, w)
CONF_THRES = 0.25
IOU_THRES = 0.45
METRIC_IOU_THRES = 0.50
CLASS_NAMES = ["plate"]
OUTPUT_DIR = BASE_DIR / "runs" / "yolov26_rknn_test"

CLASSES = ("licence plate",)
def parse_arg():
    if len(sys.argv) < 3:
        print("Usage: python3 {} onnx_model_path [platform] [dtype(optional)]".format(sys.argv[0]))
        print("       platform choose from [rk3562, rk3566, rk3568, rk3576, rk3588, rv1126b, rv1109, rv1126, rk1808]")
        print("       dtype choose from [i8, fp] for [rk3562, rk3566, rk3568, rk3576, rk3588, rv1126b]")
        print("       dtype choose from [u8, fp] for [rv1109, rv1126, rk1808]")
        exit(1)

    model_path = sys.argv[1]
    platform = sys.argv[2]

    do_quant = DEFAULT_QUANT
    if len(sys.argv) > 3:
        model_type = sys.argv[3]
        if model_type not in ["i8", "u8", "fp"]:
            print("ERROR: Invalid model type: {}".format(model_type))
            exit(1)
        elif model_type in ["i8", "u8"]:
            do_quant = True
        else:
            do_quant = False

    return model_path, platform, do_quant


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


def draw(image, boxes, scores, classes):
    """Draw the boxes on the image.

    # Argument:
        image: original image.
        boxes: ndarray, boxes of objects.
        classes: ndarray, classes of objects.
        scores: ndarray, scores of objects.
        all_classes: all classes name.
    """
    print("{:^12} {:^12}  {}".format('class', 'score', 'xmin, ymin, xmax, ymax'))
    print('-' * 50)
    for box, score, cl in zip(boxes, scores, classes):
        top, left, right, bottom = box
        top = int(top)
        left = int(left)
        right = int(right)
        bottom = int(bottom)
        cv2.rectangle(image, (top, left), (right, bottom), (255, 0, 0), 2)
        cv2.putText(image, '{0} {1:.2f}'.format(CLASSES[int(cl)], score),
                    (top, left - 6),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 0, 255), 2)

        print("{:^12} {:^12.3f} [{:>4}, {:>4}, {:>4}, {:>4}]".format(CLASSES[int(cl)], score, top, left, right, bottom))

def letterbox(im, new_shape=(640, 640), color=(0, 0, 0)):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])

    # Compute padding
    ratio = r  # ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im, ratio, (dw, dh)

def get_real_box(src_shape, box, dw, dh, ratio):
    bbox = copy(box)
    # unletter_box result
    bbox[:,0] -= dw
    bbox[:,0] /= ratio
    bbox[:,0] = np.clip(bbox[:,0], 0, src_shape[1])

    bbox[:,1] -= dh
    bbox[:,1] /= ratio
    bbox[:,1] = np.clip(bbox[:,1], 0, src_shape[0])

    bbox[:,2] -= dw
    bbox[:,2] /= ratio
    bbox[:,2] = np.clip(bbox[:,2], 0, src_shape[1])

    bbox[:,3] -= dh
    bbox[:,3] /= ratio
    bbox[:,3] = np.clip(bbox[:,3], 0, src_shape[0])
    return bbox

def postprocess_yolo26(outputs):
    """
    后处理 - 三尺度输出解码
    
    参数:
        outputs: (1, 84, 80, 80), (1, 84, 40, 40), (1, 84, 20, 20)
    
    返回:
        boxes: (N, 4) - [x1, y1, x2, y2] 归一化到原图
        scores: (N,) - 置信度
        classes: (N,) - 类别索引
    """

    all_boxes, all_scores, all_classes = [], [], []
    
    # strides for 3 scales
    strides = [8, 16, 32]
    
    for i, output in enumerate(outputs):
        # output shape: (1, 84, h, w) -> (84, h*w)
        pred = output[0].reshape(5, -1)
        
        h, w = output.shape[2], output.shape[3]
        stride = strides[i]
        
        # anchor_points
        y = np.arange(h) * stride + stride // 2
        x = np.arange(w) * stride + stride // 2
        xx, yy = np.meshgrid(x, y)
        anchor_points = np.stack([xx.ravel(), yy.ravel()], axis=0)  # (2, N)
        
        #box cls_scores
        box_dist = pred[:4, :]  # (4, N)
        cls_scores = pred[4:, :]  # (80, N)
        
        # dist2bbox
        x1y1 = anchor_points - box_dist[:2, :] * stride
        x2y2 = anchor_points + box_dist[2:, :] * stride
        boxes = np.concatenate([x1y1, x2y2], axis=0)  # (4, N)
        
        # max_cls_scores
        max_cls_scores = cls_scores.max(axis=0)  # (N,)
        
        mask = max_cls_scores > CONF_THRES
        if not mask.any():
            continue
        
        # classes
        classes = cls_scores.argmax(axis=0)

        all_boxes.append(boxes[:, mask])
        all_scores.append(max_cls_scores[mask])
        all_classes.append(classes[mask])
    
    if not all_boxes:
        return np.empty((0, 4)), np.empty(0), np.empty(0)
    
    boxes = np.concatenate(all_boxes, axis=1).T  # (N, 4)
    scores = np.concatenate(all_scores)
    classes = np.concatenate(all_classes)

    return boxes, scores, classes


def parse_ccpd_gt_bbox(image_path):
    """
    从 CCPD 文件名的第 3 个字段中解析 GT 框:
    xxx-xxx-x1&y1_x2&y2-...
    """
    stem = image_path.stem
    fields = stem.split("-")
    if len(fields) < 3:
        return None

    try:
        left_top, right_bottom = fields[2].split("_")
        x1, y1 = map(float, left_top.split("&"))
        x2, y2 = map(float, right_bottom.split("&"))
    except (ValueError, IndexError):
        return None

    x_min, x_max = (x1, x2) if x1 <= x2 else (x2, x1)
    y_min, y_max = (y1, y2) if y1 <= y2 else (y2, y1)
    return np.array([x_min, y_min, x_max, y_max], dtype=np.float32)


def box_iou_xyxy(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter_area

    if union <= 0.0:
        return 0.0
    return inter_area / union


def evaluate_image_predictions(pred_boxes, pred_scores, gt_box, iou_thresh=0.5):
    """
    单类、单 GT 场景下的匹配:
    - 置信度降序逐个匹配
    - 首个 IoU>=阈值 且未匹配过 GT 的预测记 TP，其余记 FP
    """
    if len(pred_boxes) == 0:
        return [], 0, 0, 1

    order = np.argsort(-pred_scores)
    matched = False
    records = []
    tp_count = 0
    fp_count = 0

    for idx in order:
        pred_box = pred_boxes[idx]
        score = float(pred_scores[idx])
        iou = box_iou_xyxy(pred_box, gt_box)

        if (not matched) and (iou >= iou_thresh):
            tp = 1
            fp = 0
            matched = True
            tp_count += 1
        else:
            tp = 0
            fp = 1
            fp_count += 1
        records.append((score, tp, fp))

    fn_count = 0 if matched else 1
    return records, tp_count, fp_count, fn_count


def compute_ap50(records, total_gt):
    if total_gt <= 0 or not records:
        return 0.0

    records = sorted(records, key=lambda x: x[0], reverse=True)
    tp = np.array([item[1] for item in records], dtype=np.float32)
    fp = np.array([item[2] for item in records], dtype=np.float32)

    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(fp)

    recall = tp_cum / (total_gt + 1e-9)
    precision = tp_cum / (tp_cum + fp_cum + 1e-9)

    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))

    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])

    idx = np.where(mrec[1:] != mrec[:-1])[0]
    ap = np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1])
    return float(ap)

if __name__ == '__main__':
    model_path, platform, do_quant = parse_arg()

    # Create RKNN object
    rknn = RKNN(verbose=False)

    # Pre-process config
    print('--> Config model')
    rknn.config(mean_values=[[0, 0, 0]], std_values=[[255, 255, 255]], target_platform=platform)
    print('done')

    # Load model
    print('--> Loading model')
    ret = rknn.load_onnx(model=model_path)
    # ret = rknn.load_pytorch(model=model_path,input_size_list=[[1,3,24,94]])
    if ret != 0:
        print('Load model failed!')
        exit(ret)
    print('done')

    # Build model
    print('--> Building model')
    ret = rknn.build(do_quantization=do_quant, dataset=str(QUANT_DATASET_PATH))
    if ret != 0:
        print('Build model failed!')
        exit(ret)
    print('done')

    ret = rknn.init_runtime(perf_debug=True)
    if ret != 0:
        print('Init runtime failed!')
        exit(ret)

    # test model
    image_paths = load_image_paths(TEST_DATASET_PATH)
    if not image_paths:
        raise RuntimeError(f'No image paths found in {TEST_DATASET_PATH}')

    total_time = 0.0
    valid_count = 0
    eval_images = 0
    skipped_gt = 0
    total_gt = 0
    total_tp = 0
    total_fp = 0
    total_fn = 0
    ap_records = []

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f'--> Testing with {len(image_paths)} images from {TEST_DATASET_PATH}')

    for idx, image_path in enumerate(image_paths, start=1):
        if not image_path.is_file():
            print(f'[{idx}/{len(image_paths)}] skip missing file: {image_path}')
            continue

        image = cv2.imread(str(image_path))
        if image is None:
            print(f'[{idx}/{len(image_paths)}] skip unreadable image: {image_path}')
            continue
        src_shape = image.shape[:2]
        input_image, ratio, (dw, dh) = letterbox(image, INPUT_SIZE)
        input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
        input_tensor = np.expand_dims(input_image, axis=0)  # (1, 640, 640, 3)

        if idx == 1:
            print(f'Input tensor shape: {input_tensor.shape}')

        t_start = time.perf_counter()
        outputs = rknn.inference(inputs=[input_tensor])
        infer_ms = (time.perf_counter() - t_start) * 1000.0
        total_time += infer_ms
        valid_count += 1

        print(f'[{idx}/{len(image_paths)}] {image_path.name} infer={infer_ms:.2f} ms')

        input_data = [outputs[0], outputs[1], outputs[2]]
        boxes, scores, classes = postprocess_yolo26(input_data)

        if len(boxes) == 0:
            print('no object found')
            pred_boxes = np.empty((0, 4), dtype=np.float32)
            pred_scores = np.empty((0,), dtype=np.float32)
        else:
            boxes = get_real_box(src_shape, boxes, dw, dh, ratio)
            # draw(image, boxes, scores, classes)
            # cv2.imwrite(str(OUTPUT_DIR / f'{idx}.jpg'), image)
            print('Save results to result.jpg!')
            pred_boxes = boxes.astype(np.float32)
            pred_scores = scores.astype(np.float32)

        gt_box = parse_ccpd_gt_bbox(image_path)
        if gt_box is None:
            skipped_gt += 1
            print(f'[{idx}/{len(image_paths)}] skip metric for unparsable GT: {image_path.name}')
            continue

        eval_images += 1
        total_gt += 1
        records, tp_count, fp_count, fn_count = evaluate_image_predictions(
            pred_boxes=pred_boxes,
            pred_scores=pred_scores,
            gt_box=gt_box,
            iou_thresh=METRIC_IOU_THRES,
        )
        ap_records.extend(records)
        total_tp += tp_count
        total_fp += fp_count
        total_fn += fn_count


    if valid_count > 0:
        avg_ms = total_time / valid_count
        fps = 1000.0 / avg_ms if avg_ms > 0 else 0.0
        print(f'Average inference: {avg_ms:.2f} ms/image, FPS={fps:.2f}')
        print(f'Results saved to: {OUTPUT_DIR}')
    else:
        print('No valid images were processed.')

    if total_gt > 0:
        precision = total_tp / (total_tp + total_fp + 1e-9)
        recall = total_tp / (total_tp + total_fn + 1e-9)
        map50 = compute_ap50(ap_records, total_gt)

        print(f'------ Detection Metrics (IoU={METRIC_IOU_THRES:.2f}) ------')
        print(f'Evaluated images: {eval_images}, GT boxes: {total_gt}, skipped GT: {skipped_gt}')
        print(f'TP={total_tp}, FP={total_fp}, FN={total_fn}')
        print(f'Precision: {precision:.4f}')
        print(f'Recall:    {recall:.4f}')
        print(f'mAP50:     {map50:.4f}')
    else:
        print('No valid GT boxes were found, skip metric calculation.')


    ret = rknn.export_rknn('RKNN/weights/yolo.rknn')
    # Release
    rknn.release()
