import os
import time

import cv2
import numpy as np

from image_process import cv2ImgAddText, detect_and_recognize, draw_detections,resolve_local_path, collect_image_files

provinces = ["皖", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", "苏", "浙", "京", "闽", "赣", "鲁", "豫", "鄂", "湘", "粤", "桂", "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁", "新", "警", "学", "O"]
alphabets = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W',
             'X', 'Y', 'Z', 'O']
ads = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X',
       'Y', 'Z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'O']

def get_license_plate_number(image_path):
    try:
        resolved_path = resolve_local_path(image_path)
        filename = os.path.basename(resolved_path)

        # 去掉文件扩展名，仅保留文件名主体
        basename = os.path.splitext(filename)[0]

        # CCPD 标准文件名至少包含 5 个字段，字段5是车牌号索引
        fields = basename.split("-")
        if len(fields) < 5:
            return None

        plate_index_str = fields[4]
        index_parts = plate_index_str.split("_")

        # 常规车牌: 7 段索引(省份+字母+5位)
        # 新能源车牌: 8 段索引(省份+字母+6位)
        if len(index_parts) not in (7, 8):
            return None

        indexes = [int(part) for part in index_parts]

        if not (0 <= indexes[0] < len(provinces)):
            return None
        if not (0 <= indexes[1] < len(alphabets)):
            return None
        if any(idx < 0 or idx >= len(ads) for idx in indexes[2:]):
            return None

        plate_number = provinces[indexes[0]] + alphabets[indexes[1]]
        plate_number += "".join(ads[idx] for idx in indexes[2:])
        return plate_number

    except Exception as e:
        print(f"解析CCPD标准格式失败: {image_path}, 错误: {e}")
        return None

def process_images(img_paths, conf_thres=0.25):
    total,correct = 0,0
    for img_path in img_paths:
        total += 1
        resolved = resolve_local_path(img_path)
        img_ori = cv2.imread(resolved)
        if img_ori is None:
            print(f"\n图片: {img_path}")
            print("  读取失败，请检查路径")
            continue
        
        detections = detect_and_recognize(img_ori, conf_thres=conf_thres)
        lp_str = get_license_plate_number(img_path)
        # print(f"\n图片: {resolved}")
        if not detections:
            print("  未检测到车牌")
        else:
            for det in detections:
                # print(f"  车牌 : {det['plate']} (置信度: {det['conf']:.2f}) 实际车牌 : {lp_str}")
                if det['plate'] == lp_str:
                    correct += 1
                else:
                    print(f"\n图片: {resolved}\n 车牌 = {det['plate']} (置信度: {det['conf']:.2f}) 实际车牌 : {lp_str}")
                    # vis_img = draw_detections(img_ori, detections)
                    # cv2.imshow("Recognition Result", vis_img)
                    # key = cv2.waitKey(0)
                    # if key in (27, ord("q"), ord("Q")):
                    #     break
    print(f"准确率={correct / total}")
        

def test_dataset(
        image_root
):
    images = collect_image_files(image_root,100)
    if not images:
        print(f"目录内未找到图片: {resolve_local_path(image_root)}")
        return

    print(f"共找到 {len(images)} 张图片，开始识别...")
    
    process_images(images)
