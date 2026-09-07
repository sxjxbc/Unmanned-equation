# 锥桶检测轻量化指南（Jetson 部署）

> 目标平台：Jetson AGX Xavier / Orin（512 CUDA Volta，Ubuntu 18.04 + ROS Melodic）
> 当前默认部署：TensorRT **FP16**（基准 ~25fps / ~3MB）
> 轻量化目标：**INT8 为主 + 动态 batch 辅助**，FP16→INT8 目标 ≥40fps / ~2MB

本文档解决一个问题：**本机（Windows）无法构建 TensorRT engine**（无 CUDA/TensorRT，且 `.engine` 与 GPU 架构绑定不可跨平台）。所有 engine 构建、校准、精度验证都需要在 **Jetson 目标机** 上执行。本仓库交付的是完整的「执行物料」——脚本、流程、验证工具——你到 Jetson 上按顺序跑即可。

---

## 0. 轻量化手段对比（先看清取舍）

| 手段 | 速度收益 | 模型大小 | 精度风险 | 是否需重建 engine | 本机可改代码 | 推荐度 |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **INT8 量化** | 25→**40**fps (~1.6×) | 3→**2**MB | 中（需校准+验证） | ✅ 必须（Jetson） | ❌ 推理侧已支持 | ⭐⭐⭐⭐⭐ |
| **输入分辨率 640→416** | FP16 25→**35**fps | 几乎不变 | 低（小目标略掉） | ✅ 必须（Jetson） | ❌ imgsz 已参数化 | ⭐⭐⭐⭐ |
| **动态 batch 双图合并** | 双目成对推理省 ~5-15% 开销 | 不变 | 无 | ✅ 必须（--batch 2） | ✅ `detect_batch`（默认关） | ⭐⭐⭐ |

**结论**：INT8 是性价比最高的轻量化，收益最大、最经典、脚本已就绪（`convert_to_tensorrt.py` 内置 `YOLOInt8Calibrator`）。分辨率降维作为「顺手叠加」（改 `--imgsz` 即可）。动态 batch 是代码增强，默认关闭、按需开启。

---

## 1. INT8 量化（核心路线）

### 1.1 前置条件

- ✅ `convert_to_tensorrt.py` 已内置 INT8 校准器（无需改代码）
- ✅ 推理侧已支持 INT8 engine：`ConeDetector` 加载 `.engine` 时 INT8/FP16 无差异，自动调用
- ⚠️ 需要 **~100-200 张代表性锥桶图像** 用于校准（分布要覆盖红/蓝/黄锥桶、不同光照、远近、遮挡）
- ⚠️ 需要 `.onnx` 模型（先由 `.pt` 导出，见 1.4）

### 1.2 准备校准数据

校准数据不需要标签，只需要「看起来像真实场景」的图片。推荐来源（任选其一）：

**方案 A — 从训练/验证集取（最省事，若 Jetson 上有 dataset）**
```bash
# 在 Jetson 上，从训练集 val 随机抽 150 张作为校准集
mkdir -p /opt/cone_calib/calib_images
find /path/to/dataset/images/val -name '*.jpg' | shuf | head -150 | \
    xargs -I{} cp {} /opt/cone_calib/calib_images/
```

**方案 B — 实际路采截图**（推荐，分布最真实）
```bash
# 用 ffmpeg 从双目摄像头抽帧（或从已录制视频抽帧）
ffmpeg -i recorded.mp4 -vf fps=2 -q:v 2 /opt/cone_calib/calib_images/frame_%04d.jpg
```

**方案 C — 用 benchmark/测试图**
任意一批测试推理用的图片目录即可（`--calib-dir` 直接指向它）。

> ⚠️ 校准图分辨率无所谓（脚本会 letterbox 到 `--imgsz`），但内容要覆盖真实分布，否则 INT8 精度会掉。

### 1.3 构建 INT8 engine（Jetson 上执行）

```bash
conda activate yolov5
cd ~/catkin_ws/src/cone_detector

# 步骤 1（若还没有 ONNX）：PT → ONNX
python scripts/export_to_onnx.py \
    --weights models/yolo11n_cone.pt \
    --output models/yolo11n_cone.onnx \
    --imgsz 640 --simplify

# 步骤 2：ONNX → INT8 TensorRT engine（必须在 Jetson 本机构建）
python scripts/convert_to_tensorrt.py \
    --onnx        models/yolo11n_cone.onnx \
    --output      models/yolo11n_cone_int8.engine \
    --int8 \
    --calib-dir   /opt/cone_calib/calib_images \
    --calib-num   150 \
    --imgsz       640

# 可选：同时生成一个 416 输入的 INT8 engine（极限节能模式）
python scripts/convert_to_tensorrt.py \
    --onnx        models/yolo11n_cone.onnx \
    --output      models/yolo11n_cone_int8_416.engine \
    --int8 \
    --calib-dir   /opt/cone_calib/calib_images \
    --calib-num   150 \
    --imgsz       416
```

构建日志应出现：`✓ INT8 模式已启用` 和 `✓ INT8 校准器已绑定`，最后 `✓ 构建完成`。

> 📌 文件名沿用 `yolo26n_cone*` 是历史命名（实际训练权重为 YOLO11n），以 `yolo11n_cone*` 为准更清晰；两种命名均可，只要 `NRT_TRT_ENGINE` 指向正确文件。

### 1.4 让推理侧使用 INT8 engine（无需改代码）

`ConeDetector` 优先加载 `NRT_TRT_ENGINE` 环境变量指向的 engine，INT8/FP16 加载逻辑一致。两种启用方式：

**方式 1 — 环境变量（推荐，不改文件）**
```bash
export NRT_TRT_ENGINE=/home/nvidia/catkin_ws/src/cone_detector/models/yolo11n_cone_int8.engine
python src/camera/yolo_camera.py
# 日志应见：检测后端已加载: tensorrt (.../yolo11n_cone_int8.engine)
```

**方式 2 — 直接覆盖默认路径**
把 INT8 engine 命名为 `yolo26n_cone.engine`（覆盖原 FP16），或改 `yolo_camera.py` 的 `TRT_ENGINE_PATH` 默认值。

### 1.5 验证 INT8 不掉精度（必做！）

精度验证脚本 `scripts/verify_int8.py` 会同时加载 FP16 与 INT8 engine，在相同一批图上推理，对比：
- 检测框一致性（同类别检测数、IOU 重叠度）
- 速度（fps 对比，验证 25→40 目标）
- 若一致性低于阈值（如平均 IOU < 0.8 或漏检率 > 5%），说明校准不充分，需增加校准图或检查分布。

```bash
conda activate yolov5
cd ~/catkin_ws/src/cone_detector
python scripts/verify_int8.py \
    --fp16   models/yolo11n_cone.engine \
    --int8   models/yolo11n_cone_int8.engine \
    --images /opt/cone_calib/calib_images \
    --imgsz  640
```

---

## 2. 动态 batch 双图合并（辅助路线，默认关闭）

双目相机每帧采集左+右两张图，当前 `yolo_camera.py` 是**顺序两次单图推理**（每次 batch=1）。合并为 **batch=2 单次推理** 可省去一次 CUDA 内核启动 / memcpy 开销（双目成对推理层面约 5-15% 提升）。

### 2.1 启用条件（三件都满足才生效）

1. 代码侧：本仓库已新增 `ConeDetector.detect_batch()` + TensorRTEngine 动态 batch buffer 支持
2. 环境变量：`NRT_BATCH_MERGE=1`（默认 0 / 关闭，不影响现有逻辑）
3. engine 必须按 `--batch 2` 构建（opt_shape batch=2）：
   ```bash
   python scripts/convert_to_tensorrt.py \
       --onnx   models/yolo11n_cone.onnx \
       --output models/yolo11n_cone_int8_b2.engine \
       --int8 --calib-dir /opt/cone_calib/calib_images --calib-num 150 \
       --imgsz 640 --batch 2
   export NRT_TRT_ENGINE=models/yolo11n_cone_int8_b2.engine
   export NRT_BATCH_MERGE=1
   python src/camera/yolo_camera.py
   ```

### 2.2 注意事项

- batch=2 engine 也能处理 batch=1（动态 shape min=1），所以单独推理仍正常
- 后端为 onnx/yolov5 时 `NRT_BATCH_MERGE` 自动忽略（仅 TRT 支持 batch 合并）
- 若启用后报错，直接 `NRT_BATCH_MERGE=0` 回退，零风险

---

## 3. 输入分辨率降维（顺手叠加，可选）

FP16 640→416 直接降到 FP16 ≥35fps；INT8 416 更快。无需重训，只需重建 engine（改 `--imgsz`）：
```bash
python scripts/convert_to_tensorrt.py \
    --onnx models/yolo11n_cone.onnx --output models/yolo11n_cone_int8_416.engine \
    --int8 --calib-dir /opt/cone_calib/calib_images --calib-num 150 --imgsz 416
```
运行期通过 `NRT_TRT_ENGINE` 指向新 engine 即可；`TRT_INPUT_SIZE` 会自动从 engine 读取，无需改代码。

> ⚠️ 416 对远处小锥桶（锥桶在图像中仅占几十像素）检测召回可能下降，建议在真实场景复测后再切换。

---

## 4. 性能基准目标（验收标准）

| 配置 | 目标帧率 | 模型大小 | 备注 |
|---|:---:|:---:|---|
| FP16 @640（当前） | ~25 fps | ~3 MB | 生产基准 |
| **INT8 @640（目标）** | **≥40 fps** | **~2 MB** | **本次轻量化主目标** |
| INT8 @416 | ≥50 fps | ~2 MB | 极限节能模式 |
| INT8 @640 + batch=2 | ≥40 fps（双目成对更省） | ~2 MB | 辅助叠加 |

验收方式：`scripts/benchmark_detector.py` 测纯推理时延；`scripts/verify_int8.py` 测 INT8 vs FP16 一致性；实际 ROS 运行看 `/perception/cones` 发布频率与 `current_fps` 显示。

---

## 5. 回滚方案

| 异常 | 处理 |
|---|---|
| INT8 精度明显下降 | 增加校准图数量/覆盖度，或回退 FP16：`NRT_TRT_ENGINE` 重新指向 FP16 engine |
| INT8 构建失败 | `convert_to_tensorrt.py` 不加 `--int8`，保留 FP16 |
| batch=2 报错 | `NRT_BATCH_MERGE=0` 关闭合并，或重建 engine 时去掉 `--batch 2` |
| 一切异常 | 环境变量全部清空，`ConeDetector` 自动回退 ONNX / YOLOv5 PyTorch |

---

## 6. 执行清单（Jetson 上一步到位）

```bash
# ① 环境校验
conda activate yolov5
python src/cone_detector/scripts/check_trt_env.py

# ② 准备校准图（方案 A/B/C 任选）
mkdir -p /opt/cone_calib/calib_images && cp <你的锥桶图>/* /opt/cone_calib/calib_images/

# ③ PT→ONNX（若还没有）
python src/cone_detector/scripts/export_to_onnx.py \
    --weights models/yolo11n_cone.pt --output models/yolo11n_cone.onnx --imgsz 640 --simplify

# ④ ONNX→INT8 engine（必须 Jetson）
python src/cone_detector/scripts/convert_to_tensorrt.py \
    --onnx models/yolo11n_cone.onnx --output models/yolo11n_cone_int8.engine \
    --int8 --calib-dir /opt/cone_calib/calib_images --calib-num 150 --imgsz 640

# ⑤ 精度 + 速度验证
python src/cone_detector/scripts/verify_int8.py \
    --fp16 models/yolo11n_cone.engine --int8 models/yolo11n_cone_int8.engine \
    --images /opt/cone_calib/calib_images --imgsz 640

# ⑥ 启用 INT8 推理
export NRT_TRT_ENGINE=models/yolo11n_cone_int8.engine
python src/camera/yolo_camera.py
```

---

## 8. 代码层时延优化（推理管线，无需重建 engine）

> 与 INT8 / 动态 batch 正交：这些改动**不依赖量化、不改 engine**，纯推理代码路径优化，
> 对 FP16 / INT8 / ONNX 后端同时生效，且本机即可交付（Jetson 运行即生效）。

### 8.1 优化清单

| 优化 | 位置 | 收益 |
|---|---|---|
| 后处理复用 letterbox 缩放参数（ratio/pad），去掉对原图的**重复全分辨率 resize** | `inference/trt_yolo_detector.py` 的 `postprocess` / `_preprocess`；`scripts/cone_detector_node.py` 同名函数 | 每帧省 1 次 1280×720→640 缩放（双目 ×2），CPU 开销显著下降 |
| TensorRT 预分配**锁页（pinned）主机缓冲** + 异步流 H2D→推理→D2H | `TensorRTEngine.__init__` + `infer` | 消除每帧 `np.empty` 分配；锁页内存让 H2D/D2H 异步传输更快（Jetson 上明显） |
| 引擎**预热**（warmup） | `TensorRTEngine._warmup`（加载时跑 3 次 dummy） | 消除首帧 kernel JIT 长时延，首帧反馈不再卡顿 |
| 输入尺寸从 engine 自动推导 | `ConeDetector._sync_input_size` | 416/640 engine 自动适配 `input_size`，无需改 `TRT_INPUT_SIZE` 常量，防配错 |

### 8.2 注意事项

- `infer` 返回值是引擎自有锁页缓冲的**视图**（单次复用），本检测器为单线程顺序调用，安全；若未来改为多线程并发推理，需各自持缓冲或加锁。
- 动态 batch（§2）第一次切到 batch=2 时会自动重分配锁页缓冲，无额外配置。
- 这些是**纯代码改动**，无需 Jetson 重建 engine；部署时直接用新代码即可生效。

---

## 7. 本机（Windows）已交付 / 待 Jetson 执行


| 交付 | 位置 | 说明 |
|---|---|---|
| INT8 构建脚本 | `scripts/convert_to_tensorrt.py` | 已内置校准器，无需改 |
| INT8 验证脚本 | `scripts/verify_int8.py` | 本文档 §1.5，Jetson 验证用 |
| 本指南 | `docs/LIGHTWEIGHT_GUIDE.md` | 全流程 |
| 动态 batch 代码 | `inference/trt_yolo_detector.py` + `camera/yolo_camera.py` | `NRT_BATCH_MERGE=1` 启用，默认关 |
| **engine 构建 / 校准 / 验证** | **须在 Jetson 执行** | 本机无 CUDA/TensorRT，无法构建 |
