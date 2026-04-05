from data.load_data import CCPDDataloader,LPRDataLoader
import numpy as np
import cv2
CHARS = ['京', '沪', '津', '渝', '冀', '晋', '蒙', '辽', '吉', '黑',
         '苏', '浙', '皖', '闽', '赣', '鲁', '豫', '鄂', '湘', '粤',
         '桂', '琼', '川', '贵', '云', '藏', '陕', '甘', '青', '宁',
         '新',
         '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
         'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K',
         'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V',
         'W', 'X', 'Y', 'Z', 'I', 'O', '-'
         ]
ccpd = CCPDDataloader("/home/fiatiustitia/RK3568_LPR/src/dataset/CCPD2019/ccpd_base", [94, 24], 8)
generic = LPRDataLoader("/home/fiatiustitia/RK3568_LPR/src/dataset/test", [94, 24], 8)

for i in range(0,10):
    s = ""
    for it in ccpd[i][1]:
        if (it > 30):
            s += CHARS[it]
    cv2.imshow(f"LP={s}",ccpd[i][0])
    key = cv2.waitKey(0)
    if key == 'q':
        break
for i in range(0,10):
    s = ""
    for it in generic[i][1]:
        if (it > 30):
            s += CHARS[it]
    cv2.imshow(f"LP={s}",generic[i][0])
    key = cv2.waitKey(0)
    if key == 'q':
        break