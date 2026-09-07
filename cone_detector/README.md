# ROS1 车载实时锥桶检测系统 — `cone_detector`

基于 **YOLO11n** 的车载锥桶三色检测 ROS1 功能包，支持 **TensorRT / ONNX Runtime / YOLOv5-PyTorch 三后端自动回退** 的 GPU 加速推理、自定义数据集训练与全链路模型导出部署。

> 目标平台：NVIDIA Jetson（AGX Xavier / Orin，Ubuntu 18.04 + ROS Melodic）
> 计算平台：512 CUDA 核心 Volta 架构，目标帧率 ≥ 20 fps（FP16），INT8 量化后 ≥ 40 fps
> 决策语义：红锥桶 → 左转 / 蓝锥桶 → 右转 / 黄锥桶 → 停止

---

## 目录

1. [项目结构](#1-项目结构)
2. [锥桶颜色 ↔ 决策映射](#2-锥桶颜色--决策映射)
3. [ROS 接口](#3-ros-接口)
4. [快速开始](#4-快速开始)
   - [4.1 环境依赖](#41-环境依赖)
   - [4.2 编译](#42-编译)
   - [4.3 模型导出全链路](#43-模型导出全链路)
   - [4.4 启动检测](#44-启动检测)
   - [4.5 部署环境变量](#45-部署环境变量)
5. [训练模块](#5-训练模块)
6. [参数配置](#6-参数配置)
7. [性能基准](#7-性能基准)
8. [下游节点与推理 API 示例](#8-下游节点与推理-api-示例)
9. [轻量化部署（Jetson）](#9-轻量化部署jetson)
   - [9.1 INT8 量化（核心路线）](#91-int8-量化核心路线)
   - [9.2 动态 batch 双图合并](#92-动态-batch-双图合并)
   - [9.3 代码层时延优化（无需重建 engine）](#93-代码层时延优化无需重建-engine)
   - [9.4 INT8 验收（必做）](#94-int8-验收必做)
10. [常见问题](#10-常见问题)

---

## 1. 项目结构

```
cone_detector/                          ← ROS 功能包根目录
├── CMakeLists.txt                      ← 构建配置（含消息生成 + check 脚本注册）
├── package.xml                         ← 包描述与依赖声明
├── README.md                           ← 本文档
│
├── msg/                                ← 自定义 ROS 消息
│   ├── ConeDetection.msg              ← 单锥桶检测结果
│   └── ConeDetectionArray.msg         ← 帧级检测结果数组
│
├── inference/                         ← 【统一推理后端（无 rospy 依赖，可被 camera 包复用）】
│   ├── __init__.py
│   └── trt_yolo_detector.py           ← ConeDetector / TensorRTEngine / ONNXEngine 封装
│
├── scripts/
│   ├── cone_detector_node.py          ← 【主检测节点】TensorRT/ONNX 推理 + ROS 通信
│   ├── cone_decision_consumer.py      ← 决策订阅示例节点
│   ├── benchmark_detector.py           ← 推理性能基准测试（无需 ROS）
│   ├── check_trt_env.py                ← Jetson 部署前环境校验（依赖/GPU/模型文件）
│   ├── export_to_onnx.py               ← PT → ONNX 导出（YOLO11/8/6 通用）
│   ├── convert_to_tensorrt.py          ← ONNX → TensorRT .engine（FP16/INT8 + 动态 batch）
│   ├── verify_int8.py                  ← INT8 精度验收（FP16 vs INT8 一致性 + 速度）
│   │
│   └── train/                         ← 【训练模块】
│       ├── train_cone.py              ← YOLO 训练主脚本
│       ├── prepare_dataset.py         ← VOC/COCO → YOLO 数据集格式转换
│       ├── dataset_checker.py         ← 数据集格式校验
│       └── dataset/                   ← 【预留数据集导入目录】
│
├── launch/
│   └── cone_detector.launch            ← 主启动文件
│
├── config/
│   ├── detector_params.yaml           ← 检测器运行时参数
│   └── cone_data.yaml                ← YOLO 训练数据集配置模板
│
├── docs/
│   ├── LIGHTWEIGHT_GUIDE.md           ← 轻量化部署全流程（INT8 / 动态 batch / 代码优化）
│   └── NRT_TRT_INTEGRATION.md         ← camera 包接入 cone_detector 推理后端说明
│
└── models/                            ← 模型文件（手动放入，见 models/README.md）
    ├── yolo11n_cone.pt               ← 原始 PyTorch 权重（训练产出）
    ├── yolo11n_cone.onnx             ← ONNX 中间模型
    └── yolo11n_cone.engine           ← TensorRT engine（Jetson 本机构建）
```

> 📌 **命名说明**：历史脚本/默认路径曾用 `yolo26n_cone.*`（误命名），实际训练权重为 **YOLO11n**。新文档统一以 `yolo11n_cone.*` 为准，旧命名文件在设置 `NRT_TRT_ENGINE` 指向后同样可用。

---

## 2. 锥桶颜色 ↔ 决策映射

| 锥桶颜色 | 类别名称    | YOLO class id | 决策信号     | 行为   |
|----------|-------------|:-----------:|--------------|--------|
| 红色 🔴  | red_cone    | 0            | `TURN_LEFT`  | 左转   |
| 蓝色 🔵  | blue_cone   | 1            | `TURN_RIGHT` | 右转   |
| 黄色 🟡  | yellow_cone | 2            | `STOP`       | 停止   |
| —        | —           | —            | `default`    | 无锥桶，维持当前状态 |

> 当同一帧检测到多个锥桶时，以**置信度最高**的锥桶决策为主导信号发布到 `/cone_detector/decision`。

无检测时 → `default`（默认状态）。

---

## 3. ROS 接口

### 订阅话题

| 话题 | 类型 | 说明 |
|------|------|------|
| `/camera/image_raw` | `sensor_msgs/Image` | 车载摄像头原始图像（默认） |
| 可配置为任意图像话题 | | 通过 `image_topic` 参数覆盖 |

### 发布话题

| 话题 | 类型 | 说明 |
|------|------|------|
| `/cone_detector/detections` | `ConeDetectionArray` | 本帧所有检测结果 |
| `/cone_detector/decision` | `std_msgs/String` | 本帧主导决策（`TURN_LEFT` / `TURN_RIGHT` / `STOP` / `default`） |
| `/cone_detector/image` | `sensor_msgs/Image` | 可视化图像（含检测框，可关闭） |

### 消息结构

**ConeDetectionArray**（`/cone_detector/detections`）:
```
Header  header
ConeDetection[]  detections
uint32  num_detections
float32 inference_time
bool    has_cones
string  dominant_decision
```

**ConeDetection**（单锥桶）:
```
string  color          # "red" / "yellow" / "blue"
string  decision       # "TURN_LEFT" / "TURN_RIGHT" / "STOP"
float32 confidence     # 0.0 ~ 1.0
uint32  x_min, y_min, x_max, y_max
float32 center_x, center_y  # 归一化坐标 [0,1]
```

---

## 4. 快速开始

### 4.1 环境依赖

```bash
# ROS Melodic（Ubuntu 18.04）
sudo apt install -y \
  ros-melodic-rospy \
  ros-melodic-std-msgs \
  ros-melodic-sensor-msgs \
  ros-melodic-cv-bridge \
  ros-melodic-image-transport \
  ros-melodic-message-generation \
  ros-melodic-message-runtime

# Python 依赖
pip install --upgrade pip
pip install numpy opencv-python pyyaml

# YOLO 训练（开发机）
pip install torch torchvision ultralytics onnx onnxsim

# 推理加速（Jetson 已预装；其他平台按需安装）
pip install onnxruntime-gpu        # ONNX CUDA 加速
pip install tensorrt pycuda        # TensorRT（Jetson 已预装）
```

### 4.2 编译

```bash
cd ~/catkin_ws/src
# 将 cone_detector 整个文件夹放入此处

cd ~/catkin_ws
catkin_make
source devel/setup.bash

# 验证消息生成
rosmsg show cone_detector/ConeDetectionArray
# 应输出 ConeDetectionArray 消息结构
```

### 4.3 模型导出全链路

```
PyTorch (.pt)
    ↓  scripts/export_to_onnx.py
ONNX (.onnx)
    ↓  scripts/convert_to_tensorrt.py  [仅 Jetson 上执行]
TensorRT Engine (.engine)     ← 最终部署文件
```

**Step 1 — PT → ONNX（开发机或 Jetson 均可）**

```bash
python scripts/export_to_onnx.py \
    --weights runs/train/cone_exp/weights/best.pt \
    --output models/yolo11n_cone.onnx \
    --imgsz 640 \
    --simplify
```

**Step 2 — ONNX → TensorRT（仅目标 Jetson 平台）**

```bash
# FP16 半精度（推荐，Volta+ GPU 通用，速度快，精度损失极小）
python scripts/convert_to_tensorrt.py \
    --onnx    models/yolo11n_cone.onnx \
    --output  models/yolo11n_cone.engine \
    --fp16 \
    --imgsz   640 \
    --batch   1

# INT8 8位量化（最高性能，需校准数据，见 §9.1）
python scripts/convert_to_tensorrt.py \
    --onnx        models/yolo11n_cone.onnx \
    --output      models/yolo11n_cone_int8.engine \
    --int8 \
    --calib-dir   scripts/train/dataset/images/val \
    --calib-num   100 \
    --imgsz       640
```

> ⚠️ `.engine` 文件必须在目标 GPU 架构上构建，**不可跨平台迁移**。

### 4.4 启动检测

```bash
# 启动（TensorRT，推荐）
roslaunch cone_detector cone_detector.launch \
    model_path:=$(rospack find cone_detector)/models/yolo11n_cone.engine \
    confidence_thresh:=0.45 \
    enable_vis:=true

# 查看检测画面
rosrun image_view image_view image:=/cone_detector/image

# 调试决策话题
rostopic echo /cone_detector/decision
```

### 4.5 部署环境变量

推理后端支持通过环境变量覆盖，**无需改代码**（`camera/yolo_camera.py` 与 `ConeDetector` 共用）：

| 环境变量 | 取值 | 说明 |
|----------|------|------|
| `NRT_DETECTOR` | `trt` / `onnx` / `yolov5` | 强制指定后端（默认 `trt`，失败自动回退） |
| `NRT_TRT_ENGINE` | 路径 | TensorRT `.engine` 路径（指向 INT8/FP16 任意版本） |
| `NRT_ONNX_MODEL` | 路径 | ONNX `.onnx` 路径 |
| `NRT_BATCH_MERGE` | `1` / `0` | 双目双图合并 batch=2 推理（默认 `0` 关，需 engine 按 `--batch 2` 构建） |

```bash
# 例：启用 INT8 engine
export NRT_TRT_ENGINE=$(rospack find cone_detector)/models/yolo11n_cone_int8.engine
python src/camera/yolo_camera.py
# 日志应见：检测后端已加载: tensorrt (.../yolo11n_cone_int8.engine)
```

部署前建议先跑环境校验：

```bash
conda activate yolov5
python scripts/check_trt_env.py
# 校验：Python 库 / GPU 可用性 / 模型文件存在 / 环境变量覆盖提示
```

---

## 5. 训练模块

### 5.1 数据集准备

**方案 A — 直接放入 YOLO 格式**

```
scripts/train/dataset/
├── images/
│   ├── train/IMG_001.jpg ...
│   └── val/IMG_201.jpg ...
└── labels/
    ├── train/IMG_001.txt ...   ← class x_center y_center width height
    └── val/IMG_201.txt ...
```

**方案 B — VOC/COCO 自动转换**

```bash
# VOC XML → YOLO
python scripts/train/prepare_dataset.py \
    --input-dir /path/to/VOCdevkit/VOC2007 \
    --format voc \
    --output-dir scripts/train/dataset \
    --class-names red_cone blue_cone yellow_cone

# COCO JSON → YOLO
python scripts/train/prepare_dataset.py \
    --input-dir /path/to/coco_dataset \
    --format coco \
    --coco-anno annotations/instances_train.json \
    --output-dir scripts/train/dataset \
    --class-names red_cone blue_cone yellow_cone
```

### 5.2 数据集校验

```bash
python scripts/train/dataset_checker.py --data-yaml config/cone_data.yaml
```

### 5.3 启动训练

```bash
# 基础训练
python scripts/train/train_cone.py \
    --data-yaml config/cone_data.yaml \
    --epochs 100 --imgsz 640 --batch 16 --device 0

# Jetson 轻量训练（FP16 混合精度）
python scripts/train/train_cone.py \
    --data-yaml config/cone_data.yaml \
    --epochs 80 --imgsz 416 --batch 8 --device 0 --half

# 断点续训
python scripts/train/train_cone.py \
    --model runs/train/cone_exp/weights/last.pt \
    --data-yaml config/cone_data.yaml --epochs 50 --resume
```

| 参数 | 默认 | 说明 |
|------|------|------|
| `--data-yaml` | `config/cone_data.yaml` | 数据集配置 |
| `--epochs` | 100 | 训练轮数 |
| `--imgsz` | 640 | 输入尺寸（实时可降至 416） |
| `--batch` | 16 | 批次大小（Jetson 建议 8） |
| `--device` | auto | 设备：auto/cpu/GPU id |
| `--half` | False | FP16 混合精度 |
| `--lr0` | 0.001 | 初始学习率 |
| `--patience` | 50 | 早停轮数 |

### 5.4 训练 → 部署流程

```
train_cone.py (训练)
    ↓ best.pt / last.pt
export_to_onnx.py (导出 ONNX)
    ↓ *.onnx
convert_to_tensorrt.py (构建 engine，Jetson 上执行)
    ↓ *.engine
cone_detector.launch (部署推理)
```

---

## 6. 参数配置

`config/detector_params.yaml`:

```yaml
# 模型路径（.engine 或 .onnx）
model_path: "$(find cone_detector)/models/yolo11n_cone.engine"

# 推理参数
confidence_thresh: 0.45    # 置信度阈值
nms_iou_thresh: 0.45       # NMS IOU 阈值
input_size: 640            # 输入分辨率（416/640/320）；运行时从 engine 自动推导，无需手动改
device: "cuda"             # cuda / cpu

# ROS 话题
image_topic: "/camera/image_raw"
enable_vis: true           # 是否发布可视化图像
```

运行时覆盖参数：

```bash
roslaunch cone_detector cone_detector.launch \
    confidence_thresh:=0.35 \
    enable_vis:=false \
    input_size:=416
```

---

## 7. 性能基准

测试平台：Jetson AGX Xavier（512 CUDA 核心 / 32GB）。数值为**目标/设计值**，实际以 Jetson 实测为准。

| 精度模式 | 输入尺寸 | 帧率（目标） | 模型大小 | 推荐场景 |
|----------|:---:|:---:|:---:|------|
| **TensorRT FP16** | 640 | **≥25 fps** | ~3 MB | **生产推荐** |
| TensorRT FP16 | 416 | ≥35 fps | ~3 MB | 低功耗/节能模式 |
| TensorRT INT8 | 640 | ≥40 fps | ~2 MB | 极致性能（见 §9） |
| TensorRT INT8 | 416 | ≥45 fps | ~2 MB | 极限节能模式 |
| ONNX + CUDA | 640 | ≥18 fps | ~7 MB | 无 TensorRT 时备选 |
| ONNX + CPU | 640 | ~5 fps | ~7 MB | 仅调试用 |

> 代码层时延优化（§9.3）在不重建 engine、不损失精度前提下进一步降低单帧开销，与 INT8 / 动态 batch 正交叠加。

---

## 8. 下游节点与推理 API 示例

### 8.1 ROS 订阅示例

```python
#!/usr/bin/env python3
"""cone_decision_consumer.py — 决策信号订阅示例"""
import rospy
from cone_detector.msg import ConeDetectionArray

def on_detections(msg: ConeDetectionArray):
    if not msg.has_cones:
        rospy.logwarn_throttle(5, "[决策] 无锥桶检测，输出默认状态")
        return

    rospy.loginfo(f"[决策] 本帧检测 {msg.num_detections} 个锥桶，"
                  f"主导决策: {msg.dominant_decision}，"
                  f"推理耗时: {msg.inference_time*1000:.1f}ms")

    for det in msg.detections:
        rospy.loginfo(f"  -> {det.color} ({det.confidence:.2f}) -> 决策 {det.decision}")

if __name__ == "__main__":
    rospy.init_node("cone_decision_consumer")
    rospy.Subscriber("/cone_detector/detections", ConeDetectionArray, on_detections)
    rospy.loginfo("[决策节点] 已订阅 /cone_detector/detections")
    rospy.spin()
```

### 8.2 Python 直接调用 `ConeDetector`（无 ROS 依赖）

`inference/trt_yolo_detector.py` 已去除 rospy 依赖，可被任意 Python 程序直接调用：

```python
import sys, cv2
sys.path.insert(0, "/path/to/src")          # src 目录（含 cone_detector 包）
from cone_detector.inference import ConeDetector, COLOR_MAP

det = ConeDetector(
    engine_path="models/yolo11n_cone.engine",  # 或设 NRT_TRT_ENGINE 环境变量
    conf_thresh=0.45, iou_thresh=0.45, input_size=640,
)

frame = cv2.imread("frame.jpg")
results = det.detect(frame)                   # list[{label, center, bbox, confidence}]

for r in results:
    # label -> 决策 由 COLOR_MAP 统一映射
    cls_id = [k for k, v in COLOR_MAP.items() if v[0] == r["label"]][0]
    decision = COLOR_MAP[cls_id][1]           # TURN_LEFT / TURN_RIGHT / STOP
    print(f"{r['label']}  conf={r['confidence']:.2f}  -> {decision}")

# 双图合并推理（需 engine 按 --batch 2 构建）
# batch_results = det.detect_batch([left_img, right_img])
```

### 8.3 基准测试工具

```bash
# 测试模型推理速度（无需 ROS 环境）
python scripts/benchmark_detector.py \
    --model models/yolo11n_cone.engine \
    --imgsz 640 --iterations 200 --warmup 20
```

---

## 9. 轻量化部署（Jetson）

> **重要前提**：本机（Windows 开发机）无法构建 TensorRT engine（无 CUDA/TensorRT，且 `.engine` 与 GPU 架构绑定不可跨平台）。所有 **engine 构建、INT8 校准、精度验证** 都必须在 **Jetson 目标机** 上执行。本仓库交付的是完整的「执行物料」——脚本、流程、验证工具，到 Jetson 上按顺序跑即可。

完整流程见 [`docs/LIGHTWEIGHT_GUIDE.md`](docs/LIGHTWEIGHT_GUIDE.md)。核心三板斧：

### 9.1 INT8 量化（核心路线）

收益最大、最经典：FP16 25 → **40 fps**（≈1.6×），模型 3 → **2 MB**。

- 校准器已内置（`convert_to_tensorrt.py` 的 `YOLOInt8Calibrator`），无需改代码；
- 需 ~100–200 张代表性锥桶图像（覆盖红/蓝/黄、不同光照、远近、遮挡）；
- 构建命令见 §4.3 Step 2 的 `--int8` 示例；
- 推理侧通过 `NRT_TRT_ENGINE` 指向 INT8 engine 即可，加载逻辑与 FP16 一致。

### 9.2 动态 batch 双图合并

双目左右图合并为 batch=2 一次推理，省 ~5–15% 推理开销（仅 TRT 动态 batch engine 支持）。

```bash
# 1) engine 必须按 --batch 2 构建
python scripts/convert_to_tensorrt.py \
    --onnx models/yolo11n_cone.onnx --output models/yolo11n_cone_int8_b2.engine \
    --int8 --calib-dir /opt/cone_calib/calib_images --calib-num 150 \
    --imgsz 640 --batch 2

# 2) 启用（默认关闭，零风险回退）
export NRT_TRT_ENGINE=models/yolo11n_cone_int8_b2.engine
export NRT_BATCH_MERGE=1
python src/camera/yolo_camera.py
```

- batch=2 engine 也能处理 batch=1（动态 shape min=1），单图推理正常；
- 后端为 onnx/yolov5 时 `NRT_BATCH_MERGE` 自动忽略；
- 异常时 `NRT_BATCH_MERGE=0` 即可回退。

### 9.3 代码层时延优化（无需重建 engine）

与 INT8 / 动态 batch **正交**，对 FP16 / INT8 / ONNX 全部生效，**不依赖量化、无需重建 engine**：

| 优化点 | 文件 | 效果 |
|--------|------|------|
| 去掉后处理重复 resize | `inference/trt_yolo_detector.py`（`letterbox_params` 复用 ratio/pad） | 每帧省 1 次全分辨率缩放（双目×2），CPU 开销下降 |
| TensorRT 锁页缓冲 + 异步流 | `inference/trt_yolo_detector.py`（`TensorRTEngine`） | 预分配 pinned 缓冲，H2D→推理→D2H 异步流水线，消除每帧 `np.empty` |
| 引擎预热 | `TensorRTEngine._warmup()` | 加载时跑 3 次 dummy，消除首帧 kernel JIT 长时延 |
| 输入尺寸自动推导 | `ConeDetector._sync_input_size()` | 从 engine 读空间尺寸，416/640 自动适配，免改 `TRT_INPUT_SIZE` 常量、防配错 |

### 9.4 INT8 验收（必做）

`scripts/verify_int8.py` 在同一批图上分别用 FP16 与 INT8 engine 推理，对比**检测一致性**（同类别检测数、IOU 重叠度、漏检/误检率）与**速度**：

```bash
conda activate yolov5
cd ~/catkin_ws/src/cone_detector
python scripts/verify_int8.py \
    --fp16   models/yolo11n_cone.engine \
    --int8   models/yolo11n_cone_int8.engine \
    --images /opt/cone_calib/calib_images \
    --imgsz  640
```

判定标准：平均 IOU ≥ 0.8 且漏检率 ≤ 5% 视为精度可接受；否则需增加校准图或检查分布，必要时回退 FP16（`NRT_TRT_ENGINE` 重新指向 FP16 engine）。

---

## 10. 常见问题

### Q1: `ModuleNotFoundError: No module named 'cone_detector.msg'`
消息未生成，重新编译：
```bash
cd ~/catkin_ws && catkin_make && source devel/setup.bash
```

### Q2: TensorRT 推理报错 `engine is nullptr`
`.engine` 文件与当前 GPU 架构不匹配。在目标 Jetson 上重新运行 `convert_to_tensorrt.py`。

### Q3: Jetson 显存不足（CUDA OOM）
```bash
roslaunch cone_detector cone_detector.launch input_size:=416
# 或降低 TensorRT workspace：convert_to_tensorrt.py 加 --workspace 2048
```

### Q4: 置信度阈值调整
```bash
roslaunch cone_detector cone_detector.launch confidence_thresh:=0.5
```

### Q5: 误检率高
- 提高 `confidence_thresh`（如 0.6~0.7）；
- 使用更难负样本重新训练；
- 检查数据集是否混入非锥桶图片。

### Q6: INT8 精度明显下降
增加校准图数量/覆盖度，或回退 FP16：将 `NRT_TRT_ENGINE` 重新指向 FP16 engine。

### Q7: 动态 batch 报错
`NRT_BATCH_MERGE=0` 关闭合并，或重建 engine 时去掉 `--batch 2`。

### Q8: 训练收敛慢
- 检查数据集是否足够（建议 ≥300 张/类）；
- 确认 `data.yaml` 中 `path` 为绝对路径；
- 使用 `--half` 启用混合精度加速。

---

## License

MIT License — 可自由使用、修改和商业化。
