import cv2
import numpy as np
from model.STNet import build_STNet
import torch
from data.load_data import CCPDDataloader

CHARS = ['京', '沪', '津', '渝', '冀', '晋', '蒙', '辽', '吉', '黑',
         '苏', '浙', '皖', '闽', '赣', '鲁', '豫', '鄂', '湘', '粤',
         '桂', '琼', '川', '贵', '云', '藏', '陕', '甘', '青', '宁',
         '新',
         '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
         'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K',
         'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V',
         'W', 'X', 'Y', 'Z', 'I', 'O', '-'
         ]


def detransform(img, dtype=np.uint8):
    img = np.transpose(img, (1, 2, 0)).astype(np.float32, copy=True)
    img = img / 0.0078125
    img = img + 127.5
    img = np.rint(img)
    if np.issubdtype(dtype, np.integer):
        img = np.clip(img, np.iinfo(dtype).min, np.iinfo(dtype).max)
    return img.astype(dtype)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = build_STNet(False).to(device)
net.load_state_dict(torch.load("LPRNet/weights/Final_STNet_model.pth", map_location=device))
net.eval()

ccpd = CCPDDataloader("/home/fiatiustitia/RK3568_LPR/src/dataset/CCPD_test", [94, 24], 8)


for i in range(10):
    img, label, _ = ccpd[i]
    plate = ""
    for idx in label:
        if idx > 30:
            plate += CHARS[idx]

    img_tensor = torch.from_numpy(img).unsqueeze(0).to(device)
    with torch.no_grad():
        warped = net(img_tensor).detach().cpu().numpy()[0]

    img_ori = detransform(img)
    img_warped = detransform(warped)
    img_concat = np.hstack((img_ori, img_warped))
    cv2.imshow(f"LP={plate} | left=raw right=stnet", img_concat)
    key = cv2.waitKey(0)
    cv2.destroyAllWindows()
    if key == ord('q'):
        break
