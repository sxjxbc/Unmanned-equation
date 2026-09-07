# NRT_WS × cone_detector 集成验证清单

> 目标：验证「检测器后端替换（YOLOv5 PyTorch → TensorRT YOLO11n）」后，
> `/perception/cones` 三维输出 **schema 与坐标语义完全不变**、下游规划器**代码零改动**，同时获得推理提速。

## 1. 集成架构

```
双目摄像头 (ffmpeg /dev/video0,1)
      │
      ▼
camera/yolo_camera.py                     [新增] cone_detector/inference/trt_yolo_detector.py
  ├─ load_model()  ───────────────────────────►  ConeDetector（统一后端）
  │                                                ├─ TensorRT .engine  ← 默认（最快）
  │                                                ├─ ONNX Runtime .onnx ← 备选
  │                                                └─ YOLOv5 PyTorch .pt ← 回退
  ├─ detect_objects(model, img)  =  model.detect(img)   # 输出 schema 不变
  ├─ calculate_coordinates()  ◄── 视差测距（保留，未改动）
  ├─ matched_pairs()          ◄── 左右同色+Y 相近匹配（保留，未改动）
  ├─ /perception/cones 发布  ◄── msgs/ConeDetection（x/y/z 米，schema 不变）
  └─ CAN UDP (192.168.0.7:20001)             （保留，未改动）
```

- 消息冲突处理：`msgs/ConeDetection`（3D：color/x/y/z 米）与 `cone_detector/ConeDetection`
  （2D：color/decision/bbox）在 ROS 中按包名空间隔离，**无冲突**。相机节点继续使用
  `msgs/ConeDetection`，cone_detector 独立节点（可选）使用自己的消息。

### 下游消费拓扑（2026-09-03 矫正后，与根 README §7 一致）

| 规划器 | 位置来源 | 视觉来源 | 依赖颜色 |
|---|---|---|---|
| `high_speed_path_slam_node`（高速） | `/clustered_points`（激光 DBSCAN） | `/perception/cones` **直连**（20 帧缓存） | ✅ 只取颜色 |
| `figure_eight_planner_node`（八字，包名 **bazi**） | `/clustered_points` | `/perception/cones` → `cone_detection_to_array_node`（0.5 s 累积窗）→ `/visual_cone_array` | ✅ 颜色置信度融合 |
| `acceleration_event_node`（加速） | `/clustered_points` | —（无视觉订阅） | ❌ |

- 后端替换的正确性承诺 = `/perception/cones` 的 **schema（color/x/y/z 米）与坐标语义（x=前方深度、
  y=水平偏移左正、相对激光坐标）不变** → 三个规划器**代码零改动**。
- ⚠️ 注意：`figure_eight` 不经 `/visual_cone_array` 桥接就收不到视觉；桥接节点
  `cone_detection_to_array_node` 已注册进 `figure_eight_planner.launch`
  （`roslaunch bazi figure_eight_planner.launch`），现场勿单独漏起该节点。

## 2. 环境准备（Jetson，一次性的）

```bash
conda activate yolov5

# ① 环境校验
python src/cone_detector/scripts/check_trt_env.py
#   期望：tensorrt ✓ / pycuda ✓ / onnxruntime ✓ / GPU ✓ / engine 文件 ✓

# ② 缺依赖时安装
sudo apt install tensorrt python3-pycuda     # Jetson 通常已预装
pip install onnxruntime

# ③ 放置模型（engine 必须在 Jetson 本机构建，不可跨平台复制）
#   见 src/cone_detector/models/README.md
python src/cone_detector/scripts/export_to_onnx.py --weights best.pt \
    --output src/cone_detector/models/yolo26n_cone.onnx --imgsz 640
python src/cone_detector/scripts/convert_to_tensorrt.py \
    --onnx src/cone_detector/models/yolo26n_cone.onnx \
    --output src/cone_detector/models/yolo26n_cone.engine --fp16 --imgsz 640

# ④ （如做 catkin 编译）工作空间编译，确认 cone_detector 消息生成
cd ~/catkin_ws && catkin_make && source devel/setup.bash
```

## 3. 功能验证清单

### 3.1 消息 schema 不变（P0）

```bash
rostopic echo -n1 /perception/cones
# 期望字段：header{stamp,frame_id="camera"}, color(string), x, y, z(float64 米)
# 与集成前 msgs/ConeDetection.msg 完全一致
rostopic type /perception/cones        # → msgs/ConeDetection
```

### 3.2 三维坐标可用（P0）

- 摆放单个红色锥桶于正前方约 3m：`rostopic echo /perception/cones`
- 期望 `x ≈ 3.0`（米），`|y| < 0.3`，`z = 0.0`；左右移动锥桶，x/y 随之正确变化
- 与集成前 PyTorch 后端的读数对比，偏差应在标定误差范围内（同帧画面 A/B 对照）

### 3.3 颜色语义一致（P0）

| 场景 | 期望 color | 期望下游行为 |
|---|---|---|
| 红锥桶 | `red` | 八字/高速按 TURN_LEFT 语义（左转绕行） |
| 蓝锥桶 | `blue` | 八字/高速按 TURN_RIGHT 语义（右转绕行） |
| 黄锥桶 | `yellow` | STOP（停车避让） |

- 注意：`camera` 节点**不发布 decision 字段**；颜色 → 行为（红→TURN_LEFT、蓝→TURN_RIGHT、黄→STOP）
  由**消费颜色的规划器**自行解析（`figure_eight` / `high_speed`），`acceleration_event` 为纯激光
  路径规划、无颜色依赖。语义一致性 = color 字段映射正确（与 cone_detector 的
  COLOR_MAP：0/1/2=red/blue/yellow 对齐）。

### 3.4 后端回退验证（P1）

```bash
NRT_DETECTOR=trt    python src/camera/yolo_camera.py   # 日志应见: 后端：tensorrt
NRT_DETECTOR=onnx   python src/camera/yolo_camera.py   # 日志应见: 后端：onnx
NRT_DETECTOR=yolov5 python src/camera/yolo_camera.py   # 日志应见: 后端：yolov5
# 三种模式均能正常发布 /perception/cones，且颜色/坐标结果一致
```

### 3.5 FPS 提速对比（P1）

在 yolo_camera.py 窗口左上角已有实时 FPS 显示。对比：

| 后端 | 预期 FPS（参考） | 说明 |
|---|---|---|
| YOLOv5 PyTorch（集成前） | 基线（记录数值） | FP32，无 TensorRT |
| TensorRT FP16 | ≥ 2× 基线 | YOLO11n + FP16，Jetson Volta 以上 |
| ONNX Runtime | 1.2~2× 基线 | CUDA EP |

> 评估指标：同场景、同分辨率（1280×720）、同阈值（conf=0.45, iou=0.45），
> 各跑 ≥100 帧取平均。

## 4. 回滚方案（任何时候可恢复）

### 方案 A：环境变量（推荐，无需改代码）

```bash
NRT_DETECTOR=yolov5 python src/camera/yolo_camera.py
```

### 方案 B：git 还原（代码级）

```bash
git checkout -- src/camera/yolo_camera.py        # 恢复原 PyTorch 后端
# 或整目录还原：git checkout -- src/camera
# cone_detector 子包删除不影响原系统：
rm -rf src/cone_detector                          # 仅删除新增子包（请先确认已备份）
```

### 方案 C：后端异常自动回退（运行时）

- `ConeDetector` 加载时按 trt → onnx → yolov5 顺序尝试，某后端加载失败自动降级
- 启动日志会打印实际生效的后端（`[加载] 模型加载成功（后端：xxx）`）

## 5. 变更文件清单

| 文件 | 操作 | 说明 |
|---|---|---|
| `src/cone_detector/` | 新增（整体） | cone_detector 子包（排除训练数据集/缓存） |
| `src/cone_detector/inference/trt_yolo_detector.py` | 新增 | 统一三后端检测器（核心） |
| `src/cone_detector/inference/__init__.py` | 新增 | 模块导出 |
| `src/cone_detector/scripts/check_trt_env.py` | 新增 | Jetson 环境校验脚本 |
| `src/cone_detector/models/README.md` | 新增 | 权重放置/生成说明 |
| `src/cone_detector/docs/NRT_TRT_INTEGRATION.md` | 新增 | 本文档 |
| `src/cone_detector/CMakeLists.txt` | 修改 | 注册 check 脚本 + 安装 inference 模块 |
| `src/camera/yolo_camera.py` | 修改 | ①后端配置常量+sys.path 注入 ②load_model 三后端回退 ③detect_objects 统一契约 |

**未改动**（TRT 后端替换范围内）：`msgs/`（消息定义）、三个规划器代码、`calculate_coordinates`/`matched_pairs`/ROS 发布/CAN UDP/摄像头采集。

> 注：三个规划器在 2026-09-03 有过一次**独立的融合链路矫正**（订阅话题统一为 `/clustered_points`、
> 新增 `/visual_cone_array` 桥接节点），与本 TRT 替换正交，不影响本清单结论。
> 融合链路完整说明见根 `README.md §7` 与 `figure_eight_planner/launch/figure_eight_planner.launch`。
