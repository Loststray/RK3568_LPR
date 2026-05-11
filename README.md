# RK3568_LPR

基于 `YOLO + STNet + LPRNet` 的车牌识别项目。仓库内同时提供了桌面端推理脚本、Web Demo、CCPD 数据集处理、YOLO/LPRNet 训练与测试、ONNX 导出，以及面向 RK3568 等 Rockchip 平台的 RKNN 部署脚本。

## 功能概览

- 图片批量识别：检测车牌并输出识别结果，默认从 `src/YOLO/test` 读取示例图片。
- 视频/摄像头识别：支持关键帧触发、运动检测、保存关键帧、导出带结果的视频。
- 数据集模式：对 CCPD 风格测试集做整集评测。
- Web Demo：上传图片或视频，在浏览器内查看检测结果。
- 数据准备：将 CCPD 数据集转换为 YOLO 训练格式，并可抽样生成测试集。
- 模型训练与评测：包含 YOLO 检测模型和 STNet + LPRNet 识别模型。
- 模型导出与部署：支持导出 ONNX，并进一步转换/测试 RKNN 模型。

## 目录结构

```text
src/
├── test.py                      # 端到端入口：image / video / dataset
├── image_process.py             # 图片推理与绘制
├── video_process.py             # 视频推理、关键帧逻辑、视频保存
├── dataset_process.py           # CCPD 风格测试集评测
├── make_test_dataset.py         # 从 CCPD2019 抽样生成测试集
├── YOLO/
│   ├── train_yolo.py            # YOLO 训练
│   ├── test_yolo.py             # YOLO 快速可视化测试
│   ├── eval_yolo.py             # YOLO 推理耗时评测
│   ├── convert.py               # YOLO .pt -> ONNX
│   ├── convert_ccpd_to_yolo.py  # CCPD -> YOLO 数据集格式
│   └── weights/
├── LPRNet/
│   ├── train_LPRNet.py          # STNet + LPRNet 训练
│   ├── test_LPRNet.py           # STNet + LPRNet 测试
│   ├── convert.py               # LPRNet / STNet -> ONNX
│   ├── data/load_data.py        # CCPD / 通用数据加载
│   ├── model/
│   └── weights/
├── RKNN/
│   ├── convert.py               # ONNX -> RKNN
│   ├── test_yolo.py             # YOLO RKNN 量化、推理与指标评测
│   ├── test_lpr_and_stn.py      # STN/LPR RKNN 量化、推理与识别评测
│   ├── video_sender.py          # FFmpeg 推流示例
│   └── weights/
├── web/
│   ├── app.py                   # Flask Web 服务
│   ├── templates/
│   ├── static/
│   └── runtime/                 # 运行时上传目录
├── YOLO_Data/                   # YOLO 训练数据目录
├── dataset/                     # CCPD2019 / CCPD2020 / CCPD_test
└── videos/                      # 示例视频
```

## 环境依赖

建议使用 Python 3.10+ 和虚拟环境。

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

补充说明：

- `requirements.txt` 中的 `torch==2.10.0+cu130` 是带 CUDA 的固定版本；如果你的环境不使用 CUDA 13.0，需要先按实际平台安装合适的 PyTorch，再安装其余依赖。
- `src/web/app.py` 依赖 `Flask`。
- `src/video_process.py` 在保存视频时会尝试调用 `ffmpeg` 转码到 H.264；如果系统没有可用编码器，会回退为原始 MP4 输出。
- `src/RKNN/*` 依赖 Rockchip 的 `rknn-toolkit2` 或兼容环境，这部分通常需要在目标开发环境单独安装。
- 图片/视频本地可视化依赖 `cv2.imshow`，需要图形界面环境。
- 中文绘制优先使用 `/usr/share/fonts/truetype/wqy/wqy-microhei.ttc`，没有该字体时会回退到默认字体。

## 快速开始

### 1. 图片识别

```bash
python src/test.py --mode image --source YOLO/test --conf-thres 0.5
```

说明：

- `--source` 支持单个目录，脚本会递归读取图片。
- 相对路径会优先按 `src/` 目录解析，所以 `YOLO/test` 会映射到 `src/YOLO/test`。
- 图片模式会逐张弹窗显示结果，按 `q` 或 `Esc` 退出。

### 2. 视频或摄像头识别

```bash
# 摄像头
python src/test.py --mode video --source 0

# 本地视频
python src/test.py --mode video --source videos/1106041631-1-208.mp4
```

保存检测结果视频：

```bash
python src/test.py --mode video --source videos/1106041631-1-208.mp4 \
  --save-video \
  --conf-thres 0.5 \
  --frame-interval 3 \
  --motion-threshold 6.0 \
  --max-skip 30 \
  --display-scale 1.0 \
  --save-keyframes ./keyframes
```

主要参数：

- `--conf-thres`：检测阈值，默认 `0.5`。
- `--frame-interval`：每隔多少帧做一次关键帧判定，默认 `3`。
- `--motion-threshold`：帧间差分阈值，默认 `6.0`。
- `--max-skip`：最长跳过帧数，默认 `30`。
- `--display-scale`：显示或输出视频缩放比例，默认 `1.0`。
- `--save-keyframes`：保存关键帧到指定目录。
- `--save-video`：保存结果视频到本地，启用后不会弹出实时窗口。

### 3. 数据集评测模式

```bash
python src/test.py --mode dataset --source dataset/CCPD_test --conf-thres 0.5
```

该模式会读取整套测试图片并输出识别准确率。当前 GT 解析依赖 CCPD 标准文件名。

## Web Demo

启动服务：

```bash
python src/web/app.py
```

默认监听 `0.0.0.0:5000`，打开浏览器访问 `http://127.0.0.1:5000` 即可上传图片或视频。

Web 接口特性：

- 图片上传后会返回带检测框的结果图。
- 视频上传后会调用 `run_video_file()` 生成结果视频。
- 上传文件大小限制为 `512MB`。
- 运行时文件默认保存在 `src/web/runtime/uploads` 和 `src/web/static/generated/results`。

## 数据集准备

### 1. CCPD 转 YOLO 格式

自动划分训练/验证/测试集：

```bash
python src/YOLO/convert_ccpd_to_yolo.py \
  --source src/dataset/CCPD2019/ccpd_base \
  --target src/YOLO_Data \
  --val-ratio 0.2 \
  --test-ratio 0.1 \
  --max-size 10000
```

保留原始 `train/val/test` 划分：

```bash
python src/YOLO/convert_ccpd_to_yolo.py \
  --source src/dataset/CCPD2020/ccpd_green \
  --target src/YOLO_Data \
  --preserve-splits
```

常用参数：

- `--dataset-type`: `auto` / `ccpd2019` / `ccpd2020`
- `--preserve-splits`: 使用原始划分
- `--all`: 自动查找并批量转换多个 CCPD 数据集
- `--no-yaml`: 不生成 `data.yaml`
- `--max-size`: 限制处理图片数量

转换结果会写入 `src/YOLO_Data/images/*`、`src/YOLO_Data/labels/*` 和 `src/YOLO_Data/data.yaml`。

### 2. 生成测试集

```bash
python src/make_test_dataset.py
```

该脚本会从 `src/dataset/CCPD2019` 中按预设比例抽样，输出到 `src/dataset/CCPD_test`。

## YOLO 检测模型

### 1. 训练

```bash
cd src/YOLO
python train_yolo.py
```

`train_yolo.py` 当前默认配置：

- 初始权重：`./weights/best_local.pt`
- 数据配置：`../YOLO_Data/data.yaml`
- 训练轮数：`50`
- 输入尺寸：`640`
- 结果输出：`runs/train/yolo26_ccpd`

### 2. 快速测试

```bash
cd src/YOLO
python test_yolo.py
```

脚本默认读取 `test/1.jpg`、`test/2.jpg`、`test/3.jpg`，并弹窗显示检测框。

### 3. 推理耗时评测

```bash
python src/YOLO/eval_yolo.py \
  --weights src/YOLO/weights/best.pt \
  --source src/dataset/CCPD_test \
  --batch-size 1 \
  --imgsz 640 \
  --conf 0.5
```

## STNet + LPRNet 识别模型

### 1. 训练

```bash
python src/LPRNet/train_LPRNet.py \
  --train_img_dirs dataset/CCPD2019,dataset/CCPD2020 \
  --test_img_dirs dataset/CCPD_test \
  --dataset_type ccpd
```

说明：

- 默认会同时加载 `Final_LPRNet_model.pth` 和 `Final_STNet_model.pth` 继续训练。
- `--dataset_type ccpd` 使用 CCPD 标准文件名解析标签和边界框。
- `--dataset_type generic` 使用通用加载器，此时文件名需要以车牌字符串作为前缀。

### 2. 测试

```bash
python src/LPRNet/test_LPRNet.py \
  --test_img_dirs dataset/CCPD_test \
  --dataset_type ccpd \
  --decode_method greedy
```

可选解码方式：

- `--decode_method greedy`
- `--decode_method beam --topk 3`

## ONNX 导出

### 1. 导出 YOLO

```bash
python src/YOLO/convert.py \
  --weights src/YOLO/weights/best.pt \
  --output src/YOLO/weights/best.onnx \
  --imgsz 640 \
  --batch 1
```

### 2. 导出 LPRNet

```bash
python src/LPRNet/convert.py \
  --model LPRNet \
  --weights src/LPRNet/weights/Final_LPRNet_model.pth \
  --output src/LPRNet/weights/LPRNet.onnx
```

### 3. 导出 STNet

```bash
python src/LPRNet/convert.py \
  --model STNet \
  --weights src/LPRNet/weights/Final_STNet_model.pth \
  --output src/LPRNet/weights/STNet.onnx
```

## RKNN 部署与测试

以下脚本面向 Rockchip NPU 环境，通常需要在安装了 `rknn-toolkit2` 的机器上执行。

### 1. 单模型 ONNX -> RKNN

```bash
python src/RKNN/convert.py src/LPRNet/weights/STNet.onnx rk3568 i8 src/RKNN/weights/stnet.rknn
```

### 2. YOLO RKNN 测试与导出

```bash
python src/RKNN/test_yolo.py src/YOLO/weights/best.onnx rk3568 i8
```

该脚本会：

- 使用 `src/RKNN/dataset.txt` 做量化
- 使用 `src/RKNN/dataset_ccpd_test.txt` 做推理与检测指标评测
- 导出 `src/RKNN/weights/yolo.rknn`

### 3. STNet + LPRNet RKNN 测试与导出

```bash
python src/RKNN/test_lpr_and_stn.py \
  src/LPRNet/weights/STNet.onnx \
  src/LPRNet/weights/LPRNet.onnx \
  rk3568 \
  i8
```

该脚本会：

- 使用 `src/RKNN/dataset_lpr.txt` 做量化
- 使用 `src/RKNN/dataset_lpr_full.txt` 做识别评测
- 导出 `src/RKNN/weights/stnet.rknn` 和 `src/RKNN/weights/lprnet.rknn`

### 4. RTSP 推流示例

```bash
python src/RKNN/video_sender.py
```

脚本会读取 `/dev/video0`，通过 `ffmpeg` 将原始帧推送到 `rtsp://127.0.0.1:8554/lpr`。

## 预置权重与模型文件

仓库中已包含一组可直接使用的模型文件：

- YOLO：`src/YOLO/weights/best.pt`、`src/YOLO/weights/best.onnx`
- STNet/LPRNet：`src/LPRNet/weights/Final_STNet_model.pth`、`src/LPRNet/weights/Final_LPRNet_model.pth`
- RKNN：`src/RKNN/weights/yolo.rknn`、`src/RKNN/weights/stnet.rknn`、`src/RKNN/weights/lprnet.rknn`

端到端推理默认会加载：

- `src/YOLO/weights/best.pt`
- `src/LPRNet/weights/Final_STNet_model.pth`
- `src/LPRNet/weights/Final_LPRNet_model.pth`

## 常见问题

- `cv2.imshow` 无法显示：请在带桌面的环境运行，或使用 `--save-video` / Web Demo 替代本地窗口。
- 视频保存后浏览器无法直接播放：脚本会优先用 `ffmpeg` 转码为 H.264；如果本机没有可用 H.264 编码器，会保留原始 MP4。
- 中文车牌文字显示异常：安装 `wqy-microhei` 等中文字体，或修改 `src/image_process.py` 中的 `FONT_CANDIDATES`。
- RKNN 脚本无法运行：确认当前环境已安装 `rknn-toolkit2`，并且目标平台参数如 `rk3568` 与实际设备一致。
