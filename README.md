# RK3568_LPR 车牌识别项目

基于 `YOLO + LPRNet` 的两阶段车牌识别项目：先检测车牌位置，再对车牌区域做字符识别。

- 车牌检测：`src/YOLO`
- 车牌字符识别：`src/LPRNet`
- 端到端推理入口：`src/test.py`
- 数据集转换脚本（CCPD -> YOLO）：`src/YOLO/convert_ccpd_to_yolo.py`

## 1. 项目功能

- 单图/批量图片车牌检测与识别（可视化窗口展示）
- 视频文件/摄像头实时识别，支持关键帧触发与运动检测
- CCPD 数据集自动转换为 YOLO 训练格式
- YOLO 模型训练与快速测试脚本
- LPRNet 模型训练与测试脚本
- 项目内提供预训练权重，可直接跑通推理

## 2. 目录结构

```text
src/
├── test.py                         # 端到端推理（YOLO + LPRNet）
├── YOLO/
│   ├── convert_ccpd_to_yolo.py     # CCPD 转 YOLO 格式
│   ├── train_yolo.py               # YOLO 训练
│   ├── test_yolo.py                # YOLO 测试
│   ├── weights/
│   │   ├── best.pt
│   │   └── best_local.pt
│   └── test/                       # 示例测试图
├── LPRNet/
│   ├── train_LPRNet.py             # LPRNet 训练
│   ├── test_LPRNet.py              # LPRNet 测试
│   ├── model/LPRNet.py
│   ├── data/load_data.py
│   └── weights/Final_LPRNet_model.pth
├── YOLO_Data/
│   └── data.yaml                   # YOLO 数据配置
└── dataset/
    ├── CCPD2019/
    └── CCPD2020/
```

## 3. 环境依赖

建议 Python 3.9+，并使用虚拟环境。

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install torch torchvision torchaudio ultralytics opencv-python pillow numpy imutils tqdm
```

说明：
- 若使用 GPU，请按你的 CUDA 版本安装对应的 PyTorch。
- `src/test.py` 和 `src/YOLO/test_yolo.py` 会调用 `cv2.imshow`，需要图形界面环境。

## 4. 快速开始（端到端识别）

### 4.1 图片模式（默认）

```bash
python src/test.py --mode image --source YOLO/test
```

- `--source`：图片目录（相对路径会优先按 `src/` 解析）
- `--conf-thres`：检测阈值，默认 `0.25`

### 4.2 视频/摄像头模式

```bash
# 摄像头（0）
python src/test.py --mode video --source 0

# 本地视频文件
python src/test.py --mode video --source /path/to/demo.mp4
```

可用参数示例：

```bash
python src/test.py --mode video --source 0 \
  --conf-thres 0.30 \
  --frame-interval 3 \
  --motion-threshold 6.0 \
  --max-skip 30 \
  --display-scale 1.0 \
  --save-keyframes ./keyframes
```

参数含义：
- `--frame-interval`：每隔 N 帧执行一次关键帧判定
- `--motion-threshold`：帧差阈值，越大越不敏感
- `--max-skip`：最长跳过帧数，超出后强制识别
- `--save-keyframes`：可选，保存关键帧目录

## 5. 数据集转换（CCPD -> YOLO）

### 5.1 使用已划分数据（如 CCPD2020）

```bash
python src/YOLO/convert_ccpd_to_yolo.py \
  --source src/dataset/CCPD2020/ccpd_green \
  --target src/YOLO_Data \
  --preserve-splits
```

### 5.2 自动划分数据（如 CCPD2019 某子集）

```bash
python src/YOLO/convert_ccpd_to_yolo.py \
  --source src/dataset/CCPD2019/ccpd_base \
  --target src/YOLO_Data \
  --val-ratio 0.2 \
  --test-ratio 0.1
```

转换后会在目标目录生成 `images/`、`labels/` 和 `data.yaml`。

## 6. YOLO 训练与测试

```bash
cd src/YOLO
python train_yolo.py
python test_yolo.py
```

当前 `train_yolo.py` 默认配置：
- 初始权重：`weights/best_local.pt`
- 数据配置：`../YOLO_Data/data.yaml`
- 训练轮数：`epochs=5`

如需自定义训练参数，直接修改 `src/YOLO/train_yolo.py` 中的 `model.train(...)`。

## 7. LPRNet 训练与测试

```bash
# 训练
python src/LPRNet/train_LPRNet.py \
  --train_img_dirs /path/to/train \
  --test_img_dirs /path/to/val

# 测试
python src/LPRNet/test_LPRNet.py \
  --test_img_dirs /path/to/test \
  --pretrained_model src/LPRNet/weights/Final_LPRNet_model.pth
```

注意：`LPRNet/data/load_data.py` 默认按文件名解析标签，需保证你的数据命名规则与脚本逻辑一致。

## 8. 预训练权重

项目中已包含可直接使用的权重：

- `src/YOLO/weights/best.pt`
- `src/LPRNet/weights/Final_LPRNet_model.pth`

端到端推理脚本 `src/test.py` 默认加载以上两个权重。

## 9. 常见问题

- 打不开窗口：请确认当前环境支持 GUI（远程服务器可用 X11 转发或改为保存图片）。
- 摄像头无法打开：检查 `--source`（如 `0`、`1`）和系统摄像头权限。
- 识别为空：可先降低 `--conf-thres`，并确认车牌区域清晰、分辨率足够。

