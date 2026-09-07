# models/ —— 推理权重放置目录

本目录存放 **YOLO11n** 锥桶检测的推理模型文件（NRT_WS 相机包 TRT 后端默认引用）：

| 文件 | 说明 | 默认引用方 |
|---|---|---|
| `yolo11n_cone.engine` | TensorRT FP16 引擎（**推荐**，需在目标 Jetson 上构建） | 通过 `NRT_TRT_ENGINE` 环境变量指向；`yolo_camera.py` 默认 `TRT_ENGINE_PATH` 仍为 `yolo26n_cone.engine`（历史命名） |
| `yolo11n_cone_int8.engine` | TensorRT INT8 引擎（**轻量化**，需在目标 Jetson 上构建 + 校准，详见 `docs/LIGHTWEIGHT_GUIDE.md`） | 通过 `NRT_TRT_ENGINE` 环境变量覆盖指向 |
| `yolo11n_cone.onnx` | ONNX Runtime 备选模型 | `yolo_camera.py` → `ONNX_MODEL_PATH` |
| `best.pt` | YOLOv5 PyTorch 回退权重（可选，原 camera 包已有） | `yolo_camera.py` → `weight_path` |

> 📌 **命名说明**：历史脚本/默认路径曾用 `yolo26n_cone.*`（误命名），实际训练权重为 **YOLO11n**
> （class 序：0=red → `TURN_LEFT`，1=blue → `TURN_RIGHT`，2=yellow → `STOP`）。
> 新文件统一以 `yolo11n_cone.*` 命名；`yolo_camera.py` 的 `TRT_ENGINE_PATH` 默认值仍是 `yolo26n_cone.engine`，
> 因此推荐直接 `export NRT_TRT_ENGINE=.../yolo11n_cone.engine` 指向新命名文件。

## 生成方式（在装有 PyTorch + ultralytics 的机器上）

```bash
# 1) 导出 ONNX
python scripts/export_to_onnx.py \
    --weights /path/to/yolo11n_cone.pt \
    --output models/yolo11n_cone.onnx \
    --imgsz 640

# 2) 转换为 TensorRT FP16 引擎（必须在目标 Jetson 上执行，engine 不可跨平台复制）
python scripts/convert_to_tensorrt.py \
    --onnx models/yolo11n_cone.onnx \
    --output models/yolo11n_cone.engine \
    --fp16 --imgsz 640

# 3) 轻量化：转换为 TensorRT INT8 引擎（FP16→INT8 速度 ~1.6×、模型 ~2MB）
#    需校准数据目录（~150 张代表性锥桶图），校准准备与精度验证见 docs/LIGHTWEIGHT_GUIDE.md
python scripts/convert_to_tensorrt.py \
    --onnx        models/yolo11n_cone.onnx \
    --output      models/yolo11n_cone_int8.engine \
    --int8 \
    --calib-dir   /path/to/calibration_images \
    --calib-num   150 \
    --imgsz       640
# 启用：export NRT_TRT_ENGINE=models/yolo11n_cone_int8.engine
```

## 部署步骤（Jetson）

```bash
# 1) 环境校验（conda yolov5 环境）
conda activate yolov5
python src/cone_detector/scripts/check_trt_env.py

# 2) 缺依赖时安装
#    Jetson 通常预装 TensorRT / pycuda：
sudo apt install tensorrt python3-pycuda
#    或 conda 环境内：
conda install -c conda-forge tensorrt pycuda
pip install onnxruntime

# 3) 启动相机节点（自动选择 TRT 后端）
export NRT_TRT_ENGINE=$(rospack find cone_detector)/models/yolo11n_cone.engine
python src/camera/yolo_camera.py
```
