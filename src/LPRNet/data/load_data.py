import os
import random
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np
from imutils import paths
from torch.utils.data import Dataset

CHARS = ['京', '沪', '津', '渝', '冀', '晋', '蒙', '辽', '吉', '黑',
         '苏', '浙', '皖', '闽', '赣', '鲁', '豫', '鄂', '湘', '粤',
         '桂', '琼', '川', '贵', '云', '藏', '陕', '甘', '青', '宁',
         '新',
         '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
         'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K',
         'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V',
         'W', 'X', 'Y', 'Z', 'I', 'O', '-'
         ]

CHARS_DICT = {char: i for i, char in enumerate(CHARS)}

provinces = ["皖", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", "苏", "浙", "京", "闽", "赣", "鲁", "豫", "鄂", "湘", "粤", "桂", "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁", "新", "警", "学", "O"]
alphabets = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W',
             'X', 'Y', 'Z', 'O']
ads = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X',
       'Y', 'Z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'O']


def _normalize_img_dirs(img_dir: Sequence[str]) -> List[str]:
    if isinstance(img_dir, str):
        img_dirs = [img_dir]
    else:
        img_dirs = list(img_dir)
    return [os.path.expanduser(p) for p in img_dirs if p]


def parse_ccpd_plate_from_path(image_path: str) -> Optional[str]:
    """
    从 CCPD 标准文件名中提取车牌字符。

    CCPD 文件名格式（按 '-' 分割为 7 个字段）：
    025-95_113-154&383_386&473-386&473_177&454_154&383_363&402-0_0_22_27_27_33_16-37-15.jpg

    字段5是车牌索引。索引映射：
    indexes[0] -> provinces
    indexes[1] -> alphabets
    indexes[2:] -> ads
    """
    basename = os.path.splitext(os.path.basename(image_path))[0]
    fields = basename.split("-")
    if len(fields) < 5:
        return None

    plate_index_str = fields[4]
    index_parts = plate_index_str.split("_")
    if len(index_parts) not in (7, 8):
        return None

    try:
        indexes = [int(part) for part in index_parts]
    except ValueError:
        return None

    if not (0 <= indexes[0] < len(provinces)):
        return None
    if not (0 <= indexes[1] < len(alphabets)):
        return None
    if any(idx < 0 or idx >= len(ads) for idx in indexes[2:]):
        return None

    plate = provinces[indexes[0]] + alphabets[indexes[1]]
    plate += "".join(ads[idx] for idx in indexes[2:])
    return plate


def parse_ccpd_bbox_from_path(image_path: str) -> Optional[Tuple[int, int, int, int]]:
    """
    从 CCPD 标准文件名中提取边界框 (x1, y1, x2, y2)。
    字段3格式: left-up_x&left-up_y_right-bottom_x&right-bottom_y
    """
    basename = os.path.splitext(os.path.basename(image_path))[0]
    fields = basename.split("-")
    if len(fields) < 3:
        return None

    bbox_str = fields[2]
    points = bbox_str.split("_")
    if len(points) != 2:
        return None

    try:
        x1, y1 = [int(v) for v in points[0].split("&")]
        x2, y2 = [int(v) for v in points[1].split("&")]
    except (ValueError, TypeError):
        return None

    # 规范化为左上到右下
    x_min, x_max = min(x1, x2), max(x1, x2)
    y_min, y_max = min(y1, y2), max(y1, y2)
    if x_max <= x_min or y_max <= y_min:
        return None
    return x_min, y_min, x_max, y_max


def plate_to_char_indices(plate_text: str) -> Optional[List[int]]:
    label = []
    for c in plate_text:
        if c not in CHARS_DICT:
            return None
        label.append(CHARS_DICT[c])
    return label


class LPRDataLoader(Dataset):
    def __init__(self, img_dir, imgSize, lpr_max_len, PreprocFun=None):
        self.img_size = imgSize
        self.lpr_max_len = lpr_max_len
        self.img_paths = []
        for directory in _normalize_img_dirs(img_dir):
            self.img_paths += [el for el in paths.list_images(directory)]
        random.shuffle(self.img_paths)
        self.PreprocFun = PreprocFun if PreprocFun is not None else self.transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        filename = self.img_paths[index]
        image = cv2.imread(filename)
        if image is None:
            raise ValueError(f"无法读取图像: {filename}")
        height, width, _ = image.shape
        if height != self.img_size[1] or width != self.img_size[0]:
            image = cv2.resize(image, self.img_size)
        image = self.PreprocFun(image)

        basename = os.path.basename(filename)
        imgname, _ = os.path.splitext(basename)
        imgname = imgname.split("-")[0].split("_")[0]
        label = []
        for c in imgname:
            label.append(CHARS_DICT[c])

        if len(label) == 8 and self.check(label) is False:
            print(imgname)
            raise AssertionError("Error label ^~^!!!")

        return image, label, len(label)

    def transform(self, img):
        img = img.astype('float32')
        img -= 127.5
        img *= 0.0078125
        img = np.transpose(img, (2, 0, 1))
        return img

    def check(self, label):
        if label[2] != CHARS_DICT['D'] and label[2] != CHARS_DICT['F'] \
                and label[-1] != CHARS_DICT['D'] and label[-1] != CHARS_DICT['F']:
            print("Error label, Please check!")
            return False
        return True


class CCPDDataloader(Dataset):
    """
    适用于 CCPD 标准命名数据集的训练数据加载器。
    返回格式与 LPRDataLoader 保持一致：(image, label, len(label))。
    """
    def __init__(self, img_dir, imgSize, lpr_max_len, PreprocFun=None, strict=False):
        self.img_size = imgSize
        self.lpr_max_len = lpr_max_len
        self.strict = strict
        self.PreprocFun = PreprocFun if PreprocFun is not None else self.transform

        self.samples = []
        skipped = 0
        for directory in _normalize_img_dirs(img_dir):
            for image_path in paths.list_images(directory):
                plate_text = parse_ccpd_plate_from_path(image_path)
                if plate_text is None:
                    skipped += 1
                    if self.strict:
                        raise ValueError(f"CCPD 文件名解析失败: {image_path}")
                    continue

                label = plate_to_char_indices(plate_text)
                if label is None or len(label) == 0 or len(label) > self.lpr_max_len:
                    skipped += 1
                    if self.strict:
                        raise ValueError(f"标签无效或超长: {image_path} -> {plate_text}")
                    continue

                bbox = parse_ccpd_bbox_from_path(image_path)
                if bbox is None:
                    skipped += 1
                    if self.strict:
                        raise ValueError(f"CCPD 边界框解析失败: {image_path}")
                    continue

                self.samples.append((image_path, label, bbox))

        random.shuffle(self.samples)
        if len(self.samples) == 0:
            raise ValueError("未找到可用的 CCPD 样本，请检查数据路径和文件名格式。")
        if skipped > 0:
            print(f"[CCPDDataloader] 有效样本: {len(self.samples)}, 跳过样本: {skipped}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        filename, label, bbox = self.samples[index]
        image = cv2.imread(filename)
        if image is None:
            raise ValueError(f"无法读取图像: {filename}")

        height, width, _ = image.shape
        x1, y1, x2, y2 = bbox
        x1 = max(0, min(x1, width - 1))
        y1 = max(0, min(y1, height - 1))
        x2 = max(1, min(x2, width))
        y2 = max(1, min(y2, height))

        if x2 <= x1 or y2 <= y1:
            if self.strict:
                raise ValueError(f"裁剪边界框越界: {filename} -> {bbox}")
            plate_img = image
        else:
            plate_img = image[y1:y2, x1:x2]
            if plate_img.size == 0:
                if self.strict:
                    raise ValueError(f"裁剪结果为空: {filename} -> {bbox}")
                plate_img = image
        crop_h, crop_w = plate_img.shape[:2]
        if crop_h != self.img_size[1] or crop_w != self.img_size[0]:
            plate_img = cv2.resize(plate_img, self.img_size)
        # image = plate_img
        image = self.PreprocFun(plate_img)

        return image, label, len(label)

    def transform(self, img):
        img = img.astype('float32')
        img -= 127.5
        img *= 0.0078125
        img = np.transpose(img, (2, 0, 1))
        return img
