# -*- coding: utf-8 -*-
"""
cone_detector.inference
=======================
NRT_WS 集成用共享推理模块。

提供统一的锥桶检测器封装（TensorRT / ONNX Runtime / YOLOv5 PyTorch 三后端
自动回退），输出与 NRT_WS camera/yolo_camera.py 原 detect_objects() 完全
一致的检测字典列表，供双目立体测距与 ROS/CAN 发布链路直接消费。
"""

from .trt_yolo_detector import (
    ConeDetector,
    create_detector,
    letterbox,
    non_max_suppression,
    scale_boxes,
    postprocess,
    COLOR_MAP,
)

__all__ = [
    "ConeDetector",
    "create_detector",
    "letterbox",
    "non_max_suppression",
    "scale_boxes",
    "postprocess",
    "COLOR_MAP",
]
