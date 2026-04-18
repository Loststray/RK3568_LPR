import sys
import time
from pathlib import Path

import cv2
import numpy as np
from rknn.api import RKNN
from copy import copy

BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "dataset.txt"
DEFAULT_QUANT = True
INPUT_SIZE = (640, 640)  # (h, w)
CONF_THRES = 0.25
IOU_THRES = 0.45
CLASS_NAMES = ["plate"]
OUTPUT_DIR = BASE_DIR / "runs" / "yolov26_rknn_test"

CLASSES = ("licence plate")
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
    ret = rknn.build(do_quantization=do_quant, dataset=str(DATASET_PATH))
    if ret != 0:
        print('Build model failed!')
        exit(ret)
    print('done')

    ret = rknn.init_runtime(perf_debug=True)
    if ret != 0:
        print('Init runtime failed!')
        exit(ret)

    # test model
    image_paths = load_image_paths(DATASET_PATH)
    if not image_paths:
        raise RuntimeError(f'No image paths found in {DATASET_PATH}')

    total_time = 0.0
    valid_count = 0
    print(f'--> Testing with {len(image_paths)} images from {DATASET_PATH}')

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
        else:
            boxes = get_real_box(src_shape, boxes, dw, dh, ratio)
            draw(image, boxes, scores, classes)
            cv2.imwrite(f'RKNN/yolotest/{idx}.jpg',image)
            print('Save results to result.jpg!')


    if valid_count > 0:
        avg_ms = total_time / valid_count
        fps = 1000.0 / avg_ms if avg_ms > 0 else 0.0
        print(f'Average inference: {avg_ms:.2f} ms/image, FPS={fps:.2f}')
        print(f'Results saved to: {OUTPUT_DIR}')
    else:
        print('No valid images were processed.')


    ret = rknn.export_rknn('RKNN/weights/yolo.rknn')
    # Release
    rknn.release()
