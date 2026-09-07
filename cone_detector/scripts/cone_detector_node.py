#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cone_detector_node.py
=====================
ROS1 锥桶检测节点

订阅话题:
  /camera/image_raw          — 摄像头原始图像
  /camera/compressed        — 压缩图像（备选）

发布话题:
  /cone_detector/detections  — ConeDetectionArray 检测结果
  /cone_detector/image      — 可视化图像（含检测框）
  /cone_detector/decision   — std_msgs/String 当前决策信号

参数（launch 或 rosparam 配置）:
  ~model_path         模型路径（.engine 或 .onnx）
  ~confidence_thresh  置信度阈值（默认 0.25）
  ~nms_iou_thresh    NMS IOU 阈值（默认 0.45）
  ~input_size        模型输入尺寸（默认 640）
  ~enable_vis        启用可视化（默认 True）
  ~device            推理设备: cuda / cpu（默认 cuda）

锥桶类别映射:
  class 0 → red_cone   → TURN_LEFT （左转）
  class 1 → blue_cone  → TURN_RIGHT（右转）
  class 2 → yellow_cone→ STOP      （停止）
"""

import os
import sys
import time
import traceback
from pathlib import Path

import cv2
import numpy as np
import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import String

# 动态导入 message（依赖 message_generation）
try:
    from cone_detector.msg import ConeDetection, ConeDetectionArray
except ImportError:
    rospy.logerr("cone_detector 消息未生成，请先 catkin_make")
    sys.exit(1)


# ──────────────────────────────────────────────
#  工具函数
# ──────────────────────────────────────────────

def letterbox(img: np.ndarray, new_shape: int = 640,
              color: tuple = (114, 114, 114)) -> tuple:
    """
    YOLO 标准预处理：等比缩放 + 灰边填充

    Returns:
        resized: 缩放后图像（new_shape × new_shape）
        ratio:   (w_ratio, h_ratio)
        pad:     (dx, dy) 左上角填充量
    """
    shape = img.shape[:2]  # (h, w)
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])

    new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    dw /= 2
    dh /= 2

    if shape[::-1] != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)

    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right,
                              cv2.BORDER_CONSTANT, value=color)

    return img, (r, r), (dw, dh)


def xywh2xyxy(x: np.ndarray) -> np.ndarray:
    """归一化 xywh → 归一化 xyxy"""
    y = np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2  # x_min
    y[..., 1] = x[..., 1] - x[..., 3] / 2  # y_min
    y[..., 2] = x[..., 0] + x[..., 2] / 2  # x_max
    y[..., 3] = x[..., 1] + x[..., 3] / 2  # y_max
    return y


def non_max_suppression(
    pred: np.ndarray,
    conf_thresh: float = 0.25,
    iou_thresh: float = 0.45,
    max_det: int = 300
) -> list:
    """
    YOLO NMS 后处理

    Args:
        pred:   模型输出 (batch, num_boxes, 5+nc)
                每行: [x,y,w,h,conf, cls1_score, cls2_score, ...]
        conf_thresh: 置信度过滤阈值
        iou_thresh:  NMS IOU 阈值

    Returns:
        list of detections per image, each: [x1,y1,x2,y2,conf,cls]
    """
    output = []
    for p in pred:
        # 过滤低置信度
        scores = p[:, 4:].max(axis=1)
        mask = scores > conf_thresh
        p = p[mask]
        scores = scores[mask]
        if not p.shape[0]:
            output.append(np.empty((0, 6)))
            continue

        # 类别
        class_ids = p[:, 4:].argmax(axis=1)
        class_scores = p[np.arange(len(scores)), 4 + class_ids]

        # 合并 [box, conf, cls]
        boxes = xywh2xyxy(p[:, :4])
        dets = np.concatenate([boxes, scores[:, None], class_ids[:, None]], axis=1)

        # NMS
        keep = cv2.dnn.NMSBoxes(
            bboxes=[tuple(map(int, b)) for b in boxes],
            scores=scores.tolist(),
            score_threshold=conf_thresh,
            nms_threshold=iou_thresh,
            eta=1.0,
            top_k=max_det,
        )
        keep = keep.flatten() if len(keep) else []
        output.append(dets[keep])

    return output


def scale_boxes(boxes: np.ndarray, orig_shape: tuple,
                resized_shape: tuple, ratio: tuple, pad: tuple) -> np.ndarray:
    """将检测框坐标从模型输入空间还原到原图像素坐标"""
    r = ratio[0]
    dw, dh = pad
    boxes = boxes.copy()
    boxes[:, [0, 2]] = (boxes[:, [0, 2]] - dw) / r  # x
    boxes[:, [1, 3]] = (boxes[:, [1, 3]] - dh) / r  # y
    # 裁剪到图像边界
    h, w = orig_shape
    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, w)
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, h)
    return boxes


# ──────────────────────────────────────────────
#  TensorRT 推理引擎
# ──────────────────────────────────────────────

class TensorRTEngine:
    """TensorRT .engine 文件推理封装"""

    def __init__(self, engine_path: str, device_id: int = 0):
        import tensorrt as trt
        import pycuda.driver as cuda

        self.trt = trt
        self.cuda = cuda
        cuda.init()
        self.ctx = cuda.Device(device_id).make_context()

        logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, "rb") as f:
            self.engine = trt.Runtime(logger).deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()

        # 获取 I/O 信息
        self.input_idx = 0
        self.output_idx = 1
        self.input_name  = self.engine.get_binding_name(self.input_idx)
        self.output_name = self.engine.get_binding_name(self.output_idx)

        # 分配 GPU 显存
        self.input_shape  = self.engine.get_binding_shape(self.input_idx)
        self.output_shape = self.engine.get_binding_shape(self.output_idx)
        self.input_size  = np.prod(self.input_shape)  * 4  # float32
        self.output_size = np.prod(self.output_shape) * 4

        self.d_input  = cuda.mem_alloc(self.input_size)
        self.d_output = cuda.mem_alloc(self.output_size)
        self.bindings = [int(self.d_input), int(self.d_output)]
        self.stream   = cuda.Stream()

        rospy.loginfo(f"[trt] TensorRT Engine 加载: {engine_path}")
        rospy.loginfo(f"[trt]   输入 shape: {self.input_shape}  dtype: float32")
        rospy.loginfo(f"[trt]   输出 shape: {self.output_shape}")

    def infer(self, blob: np.ndarray) -> np.ndarray:
        """执行推理"""
        self.ctx.push()

        # CPU→GPU
        self.cuda.memcpy_htod_async(self.d_input, blob, self.stream)
        # 执行
        self.context.execute_async_v2(
            bindings=self.bindings,
            stream_handle=self.stream.handle,
        )
        # GPU→CPU
        output = np.empty(self.output_shape, dtype=np.float32)
        self.cuda.memcpy_dtoh_async(output, self.d_output, self.stream)
        self.stream.synchronize()

        self.ctx.pop()
        return output

    def __del__(self):
        try:
            self.ctx.pop()
            del self.d_input
            del self.d_output
            del self.stream
        except Exception:
            pass


class ONNXEngine:
    """ONNX Runtime 推理封装（TensorRT 不可用时的备选）"""

    def __init__(self, onnx_path: str, device_id: int = 0):
        import onnxruntime as ort

        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        try:
            self.session = ort.InferenceSession(
                onnx_path,
                providers=providers,
            )
            rospy.loginfo("[onnx] ONNX Runtime 加载，使用 CUDA")
        except Exception:
            self.session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
            rospy.logwarn("[onnx] ONNX Runtime 加载，使用 CPU")

        self.io_names = {i.name: i.name for i in self.session.get_inputs()}
        rospy.loginfo(f"[onnx] ONNX 模型加载: {onnx_path}")

    def infer(self, blob: np.ndarray) -> np.ndarray:
        return self.session.run(None, {self.io_names[self.io_names.popitem()[0]]: blob})[0]


# ──────────────────────────────────────────────
#  推理后处理
# ──────────────────────────────────────────────

COLOR_MAP = {
    # class_id: (color_name,  decision, BGR_color)
    # 红色锥桶 → 左转，蓝色锥桶 → 右转，黄色锥桶 → 停止
    0: ("red",    "TURN_LEFT",  (0,   0,   255)),   # 红色
    1: ("blue",   "TURN_RIGHT", (255, 0,   0  )),   # 蓝色
    2: ("yellow", "STOP",       (0,   255, 255)),   # 黄色
}


def letterbox_params(orig_shape: tuple, new_shape=640) -> tuple:
    """纯数学计算 letterbox 的缩放比与填充量（不做 resize），供后处理复用"""
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    r = min(new_shape[0] / orig_shape[0], new_shape[1] / orig_shape[1])
    new_unpad = (int(round(orig_shape[1] * r)), int(round(orig_shape[0] * r)))
    dw = new_shape[1] - new_unpad[0]
    dh = new_shape[0] - new_unpad[1]
    dw /= 2
    dh /= 2
    return (r, r), (dw, dh)


def postprocess(
    output: np.ndarray,
    orig_img: np.ndarray,
    input_size: int,
    conf_thresh: float,
    iou_thresh: float,
) -> tuple:
    """
    YOLO 输出后处理

    Args:
        output:     模型原始输出 (1, num_detections, 5+nc)
        orig_img:   原始 BGR 图像
        input_size: 模型输入尺寸
        conf_thresh: 置信度阈值
        iou_thresh:   NMS IOU 阈值

    Returns:
        dets:      检测结果列表 [(x1,y1,x2,y2,conf,cls_id), ...]
        orig_copy: 原图副本（用于可视化）
        ratio:     缩放比例
        pad:       填充量
    """
    orig_h, orig_w = orig_img.shape[:2]

    # YOLO 输出通常是 (1, num_anchors, 5+nc)
    if output.ndim == 3:
        output = output[0]

    # NMS
    dets_list = non_max_suppression(
        output[None], conf_thresh, iou_thresh
    )
    dets = dets_list[0]

    if len(dets) == 0:
        return np.array([]), orig_img.copy(), (1, 1), (0, 0)

    # 缩放到原图坐标（复用纯数学 letterbox_params，避免对原图重复 resize）
    ratio, pad = letterbox_params((orig_h, orig_w), input_size)
    dets[:, :4] = scale_boxes(dets[:, :4], (orig_h, orig_w),
                               (input_size, input_size), ratio, pad)

    return dets, orig_img.copy(), ratio, pad


def draw_detections(img: np.ndarray, dets: np.ndarray) -> np.ndarray:
    """在图像上绘制检测框和标签"""
    for det in dets:
        x1, y1, x2, y2, conf, cls_id = det
        cls_id = int(cls_id)
        color_name, decision, rgb = COLOR_MAP.get(cls_id, ("unknown", "?", (200, 200, 200)))

        label = f"{color_name} {conf:.2f} → {decision}"
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), rgb, 2)

        (lw, lh), bl = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img, (int(x1), int(y1) - lh - 8),
                       (int(x1) + lw + 4, int(y1)), rgb, -1)
        cv2.putText(img, label, (int(x1) + 2, int(y1) - 4),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return img


# ──────────────────────────────────────────────
#  ROS 节点
# ──────────────────────────────────────────────

class ConeDetectorNode:
    def __init__(self):
        rospy.init_node("cone_detector", anonymous=True)

        # 参数加载
        self.model_path       = rospy.get_param("~model_path",       "")
        self.confidence_thresh = rospy.get_param("~confidence_thresh", 0.25)
        self.nms_iou_thresh  = rospy.get_param("~nms_iou_thresh",   0.45)
        self.input_size      = rospy.get_param("~input_size",       640)
        self.enable_vis      = rospy.get_param("~enable_vis",       True)
        self.device          = rospy.get_param("~device",           "cuda")
        self.image_topic     = rospy.get_param("~image_topic",      "/camera/image_raw")

        # 参数校验
        if not self.model_path or not Path(self.model_path).exists():
            rospy.logerr(f"[cone_detector] 模型文件不存在: {self.model_path}")
            rospy.logerr("[cone_detector] 请通过 launch 文件或 rosparam 设置 ~model_path")
            sys.exit(1)

        # 加载推理引擎
        self.engine = None
        self._load_engine()

        # ROS 发布/订阅
        self.pub_det   = rospy.Publisher(
            "/cone_detector/detections", ConeDetectionArray, queue_size=1
        )
        self.pub_dec   = rospy.Publisher(
            "/cone_detector/decision", String, queue_size=1
        )
        self.pub_vis   = rospy.Publisher(
            "/cone_detector/image", Image, queue_size=1
        )

        self.sub = rospy.Subscriber(
            self.image_topic, Image, self._on_image, queue_size=1, buff_size=2**20
        )

        # 统计
        self.frame_count  = 0
        self.total_infer_time = 0.0
        rospy.loginfo(f"[cone_detector] 启动！订阅: {self.image_topic}")
        rospy.loginfo(f"[cone_detector] 置信度阈值: {self.confidence_thresh}  "
                      f"NMS IOU: {self.nms_iou_thresh}  输入尺寸: {self.input_size}")
        rospy.loginfo(f"[cone_detector] 模型: {self.model_path}")

    def _load_engine(self):
        """加载 TensorRT 或 ONNX 推理引擎"""
        ext = Path(self.model_path).suffix.lower()
        if ext in [".engine", ".trt"]:
            self.engine = TensorRTEngine(self.model_path)
            self.engine_type = "tensorrt"
        elif ext == ".onnx":
            self.engine = ONNXEngine(self.model_path)
            self.engine_type = "onnx"
        else:
            rospy.logerr(f"[cone_detector] 不支持的模型格式: {ext}（支持 .engine / .onnx）")
            sys.exit(1)

    def _preprocess(self, img: np.ndarray) -> np.ndarray:
        """图像预处理：BGR→RGB + Letterbox + 归一化 + CHW"""
        blob, _, _ = letterbox(img, new_shape=self.input_size)
        blob = blob[:, :, ::-1].transpose(2, 0, 1)  # HWC→CHW, BGR→RGB
        blob = blob.astype(np.float32) / 255.0
        blob = np.expand_dims(blob, axis=0)          # → (1,3,H,W)
        return np.ascontiguousarray(blob)

    def _on_image(self, msg: rospy.AnyMsg):
        """图像话题回调"""
        t_start = rospy.Time.now()

        try:
            # 解码图像
            img = self._msg_to_image(msg)
            if img is None:
                return
        except Exception as e:
            rospy.logerr(f"[cone_detector] 图像解码失败: {e}")
            return

        # 预处理 + 推理
        try:
            blob = self._preprocess(img)
            output = self.engine.infer(blob)
            t_infer = (rospy.Time.now() - t_start).to_sec()
        except Exception as e:
            rospy.logerr(f"[cone_detector] 推理失败: {e}\n{traceback.format_exc()}")
            return

        # 后处理
        dets, vis_img, _, _ = postprocess(
            output, img,
            self.input_size,
            self.confidence_thresh,
            self.nms_iou_thresh,
        )

        # 发布检测结果
        self._publish_detections(dets, img, t_infer, t_start)

        # 发布决策信号
        self._publish_decision(dets)

        # 发布可视化
        if self.enable_vis:
            self._publish_vis(draw_detections(vis_img, dets))

        self.frame_count += 1
        self.total_infer_time += t_infer

        if self.frame_count % 100 == 0:
            avg_fps = self.frame_count / self.total_infer_time
            rospy.loginfo(f"[cone_detector] 已处理 {self.frame_count} 帧  "
                          f"平均推理耗时: {self.total_infer_time/self.frame_count*1000:.1f}ms  "
                          f"≈ {avg_fps:.1f} fps")

    def _msg_to_image(self, msg) -> np.ndarray:
        """将 ROS Image 消息转换为 OpenCV BGR 图像"""
        import cv_bridge
        bridge = cv_bridge.CvBridge()
        try:
            img = bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except cv_bridge.CvBridgeError:
            img = bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        return img

    def _publish_detections(self, dets: np.ndarray, orig_img: np.ndarray,
                             infer_time: float, stamp: rospy.Time):
        """发布 ConeDetectionArray"""
        h, w = orig_img.shape[:2]
        arr = ConeDetectionArray()
        arr.header.stamp = stamp
        arr.header.frame_id = "camera_link"
        arr.inference_time = infer_time
        arr.num_detections = len(dets)
        arr.has_cones = len(dets) > 0

        dominant_conf = 0.0
        dominant_dec = "default"

        for det in dets:
            x1, y1, x2, y2, conf, cls_id = det
            cls_id = int(cls_id)
            color_name, decision, _ = COLOR_MAP.get(cls_id, ("unknown", "?", (200, 200, 200)))

            cone = ConeDetection()
            cone.color = color_name
            cone.decision = decision
            cone.confidence = float(conf)
            cone.x_min, cone.y_min = int(x1), int(y1)
            cone.x_max, cone.y_max = int(x2), int(y2)
            cone.center_x = float((x1 + x2) / 2) / w
            cone.center_y = float((y1 + y2) / 2) / h
            arr.detections.append(cone)

            if conf > dominant_conf:
                dominant_conf = conf
                dominant_dec = decision

        arr.dominant_decision = dominant_dec if arr.has_cones else "default"
        self.pub_det.publish(arr)

    def _publish_decision(self, dets: np.ndarray):
        """发布当前帧主导决策"""
        if len(dets) == 0:
            msg = String(data="default")
        else:
            # 置信度最高的检测框决定决策
            best = dets[dets[:, 4].argmax()]
            cls_id = int(best[5])
            _, decision, _ = COLOR_MAP.get(cls_id, ("unknown", "?", (200, 200, 200)))
            msg = String(data=decision)
        self.pub_dec.publish(msg)

    def _publish_vis(self, img: np.ndarray):
        """发布可视化图像"""
        import cv_bridge
        bridge = cv_bridge.CvBridge()
        msg = bridge.cv2_to_imgmsg(img, encoding="bgr8")
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = "camera_link"
        self.pub_vis.publish(msg)

    def run(self):
        rospy.spin()


# ──────────────────────────────────────────────
#  入口
# ──────────────────────────────────────────────

if __name__ == "__main__":
    try:
        node = ConeDetectorNode()
        node.run()
    except rospy.ROSInterruptException:
        pass
    except Exception as e:
        rospy.logerr(f"[cone_detector] 未捕获异常: {e}")
        traceback.print_exc()
