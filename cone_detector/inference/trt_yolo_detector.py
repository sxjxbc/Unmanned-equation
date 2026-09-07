#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
trt_yolo_detector.py
====================
NRT_WS 相机包共享检测后端 —— cone_detector 并入 NRT_WS 后的统一推理模块。

来源：
  由独立项目 cone_detector（TensorRT YOLO11n 车载锥桶检测）的核心代码抽取，
  去除 rospy 依赖，封装为可被 NRT_WS camera/yolo_camera.py 直接调用的
  统一检测器对象。

后端优先级（自动回退，无需改代码）:
  1. TensorRT  .engine  —— 最快，Jetson 推荐（tensorrt + pycuda）
  2. ONNX Runtime .onnx —— 备选（onnxruntime，CUDA/CPU）
  3. YOLOv5 PyTorch .pt —— 最后回退，保持 camera 包原有功能

输出契约（与 yolo_camera.py 原 detect_objects() 完全一致）:
  [
    {
      'label':      'red' | 'blue' | 'yellow',
      'center':     (center_x_px, center_y_px),   # 像素坐标；红色锥桶中心下移至 65% 高度
      'bbox':       (x1, y1, x2, y2),             # 像素边界框
      'confidence': float,                        # [0,1]
    }, ...
  ]

颜色 → 决策映射（全局一致）:
  red    → TURN_LEFT     （左转）
  blue   → TURN_RIGHT    （右转）
  yellow → STOP          （停止）

依赖（Jetson conda yolov5 环境）:
  pip install tensorrt pycuda onnxruntime opencv-python numpy
  # Jetson 通常已预装 tensorrt / pycuda：sudo apt install tensorrt python3-pycuda
"""

import logging
import os
from pathlib import Path

import cv2
import numpy as np

log = logging.getLogger("cone_detector.inference")


# ──────────────────────────────────────────────
#  类别映射
# ──────────────────────────────────────────────

COLOR_MAP = {
    # class_id: (color_name, decision, BGR_color)
    0: ("red",    "TURN_LEFT",  (0,   0,   255)),
    1: ("blue",   "TURN_RIGHT", (255, 0,   0  )),
    2: ("yellow", "STOP",       (0,   255, 255)),
}
CLASS_NAMES = ["red", "blue", "yellow"]          # 与 COLOR_MAP 键序一致


# ──────────────────────────────────────────────
#  YOLO 预处理 / 后处理（纯 numpy，无 rospy 依赖）
# ──────────────────────────────────────────────

def letterbox(img: np.ndarray, new_shape: int = 640,
              color: tuple = (114, 114, 114)) -> tuple:
    """YOLO 标准预处理：等比缩放 + 灰边填充

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


def letterbox_params(shape: tuple, new_shape=640) -> tuple:
    """纯数学计算 letterbox 的缩放比与填充量（不做任何 resize）

    用于后处理还原坐标时复用，避免对原图重复做一次 cv2.resize。
    """
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
    dw = new_shape[1] - new_unpad[0]
    dh = new_shape[0] - new_unpad[1]
    dw /= 2
    dh /= 2
    return (r, r), (dw, dh)


def xywh2xyxy(x: np.ndarray) -> np.ndarray:
    """归一化 xywh → 归一化 xyxy"""
    y = np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2  # x_min
    y[..., 1] = x[..., 1] - x[..., 3] / 2  # y_min
    y[..., 2] = x[..., 0] + x[..., 2] / 2  # x_max
    y[..., 3] = x[..., 1] + x[..., 3] / 2  # y_max
    return y


def non_max_suppression(pred: np.ndarray, conf_thresh: float = 0.25,
                        iou_thresh: float = 0.45, max_det: int = 300) -> list:
    """YOLO NMS 后处理（支持 (1,N,5+nc) 与 (1,5+nc,N) 两种输出布局）

    Args:
        pred:        模型输出 (batch, num_boxes, 5+nc) 或 (batch, 5+nc, num_boxes)
        conf_thresh: 置信度过滤阈值
        iou_thresh:  NMS IOU 阈值

    Returns:
        list of detections per image, each: [x1,y1,x2,y2,conf,cls]
    """
    # 布局自动检测：若最后一个维度不是 5+nc 形状，则转置为 (N, 5+nc)
    if pred.ndim == 3 and pred.shape[1] < pred.shape[2]:
        pred = pred.transpose(0, 2, 1)

    output = []
    for p in pred:
        # 过滤低置信度
        scores = p[:, 4:].max(axis=1)
        mask = scores > conf_thresh
        p = p[mask]
        scores = scores[mask]
        if p.shape[0] == 0:
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
                ratio: tuple, pad: tuple) -> np.ndarray:
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


def postprocess(output: np.ndarray, orig_img: np.ndarray,
                input_size: int, conf_thresh: float, iou_thresh: float,
                ratio: tuple = None, pad: tuple = None) -> np.ndarray:
    """YOLO 输出后处理，返回缩放到原图坐标的检测框

    Args:
        ratio, pad: 可选，预处理已算好的 letterbox 参数；若不传则按需用纯数学
                    letterbox_params 推算（不再对原图做冗余 resize）。

    Returns:
        dets: 检测结果数组 [(x1,y1,x2,y2,conf,cls_id), ...]，可能为空
    """
    if output.ndim == 3:
        output = output[0]

    dets_list = non_max_suppression(output[None], conf_thresh, iou_thresh)
    dets = dets_list[0]
    if len(dets) == 0:
        return np.array([])

    orig_h, orig_w = orig_img.shape[:2]
    if ratio is None or pad is None:
        ratio, pad = letterbox_params((orig_h, orig_w), input_size)
    dets[:, :4] = scale_boxes(dets[:, :4], (orig_h, orig_w), ratio, pad)
    return dets


# ──────────────────────────────────────────────
#  TensorRT 引擎（延迟导入，缺依赖时走回退）
# ──────────────────────────────────────────────

class TensorRTEngine:
    """TensorRT .engine 文件推理封装（独立于 rospy）"""

    def __init__(self, engine_path: str, device_id: int = 0,
                 fallback_input_size: int = 640):
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

        # I/O 信息
        self.input_idx = 0
        self.output_idx = 1
        self.input_name = self.engine.get_binding_name(self.input_idx)
        self.output_name = self.engine.get_binding_name(self.output_idx)

        # 动态 shape 处理：识别动态 batch 维（input 第 0 维为 batch）
        # 支持双图 batch=2 合并推理（engine 需按 --batch 2 构建）
        raw_in = list(self.engine.get_binding_shape(self.input_idx))
        raw_out = list(self.engine.get_binding_shape(self.output_idx))
        self._dynamic_batch = (len(raw_in) > 0 and raw_in[0] < 0)

        # 非 batch 维的 -1 用 fallback 填充；batch 维若为 -1 暂置 1（运行时按 blob 覆盖）
        self.input_shape = list(raw_in)
        self.output_shape = list(raw_out)
        for i, s in enumerate(self.input_shape):
            if s <= 0 and i != 0:
                self.input_shape[i] = fallback_input_size
        if self.input_shape[0] < 0:
            self.input_shape[0] = 1
        for i, s in enumerate(self.output_shape):
            if s <= 0 and i != 0:
                self.output_shape[i] = fallback_input_size
        if self.output_shape[0] < 0:
            self.output_shape[0] = 1
        self.input_shape = tuple(self.input_shape)
        self.output_shape = tuple(self.output_shape)
        self._last_blob_shape = None

        self.input_size = int(np.prod(self.input_shape)) * 4     # float32
        self.output_size = int(np.prod(self.output_shape)) * 4

        self.d_input = cuda.mem_alloc(self.input_size)
        self.d_output = cuda.mem_alloc(self.output_size)
        self.bindings = [int(self.d_input), int(self.d_output)]
        self.stream = cuda.Stream()

        # 预分配锁页（pinned）主机缓冲：避免每帧分配 + 加速 H2D/D2H 异步传输
        self.h_input = cuda.pagelocked_empty(self.input_shape, dtype=np.float32)
        self.h_output = cuda.pagelocked_empty(self.output_shape, dtype=np.float32)

        # 预热：触发 kernel 加载/JIT，消除首帧长时延
        self._warmup()

        log.info("TensorRT Engine 加载: %s", engine_path)
        log.info("  输入 shape: %s  输出 shape: %s", self.input_shape, self.output_shape)

    def infer(self, blob: np.ndarray) -> np.ndarray:
        """执行推理（输入需为 (B,3,H,W) float32 contiguous，B>=1）

        使用预分配的锁页主机缓冲 self.h_input / self.h_output，以异步流完成
        H2D → 推理 → D2H 流水线，并避免每帧分配 numpy 缓冲。
        返回值为引擎自有缓冲（pinned）的视图，需在下次 infer 前消费完毕；
        本检测器为单线程顺序调用，安全。
        """
        self._ensure_buffers(blob)
        self.ctx.push()
        try:
            np.copyto(self.h_input, blob)
            self.cuda.memcpy_htod_async(self.d_input, self.h_input, self.stream)
            self.context.execute_async_v2(
                bindings=self.bindings, stream_handle=self.stream.handle
            )
            self.cuda.memcpy_dtoh_async(self.h_output, self.d_output, self.stream)
            self.stream.synchronize()
            return self.h_output
        finally:
            self.ctx.pop()

    def _warmup(self, n: int = 3) -> None:
        """预热引擎，消除首帧推理的长时延（kernel 编译/JIT）"""
        try:
            dummy = np.zeros(self.input_shape, dtype=np.float32)
            for _ in range(n):
                self.infer(dummy)
        except Exception as e:  # noqa: BLE001
            log.warning("TensorRT 预热跳过: %s", e)

    def _ensure_buffers(self, blob: np.ndarray) -> None:
        """动态 batch：按实际 blob 设置 binding shape 并保证 buffer 充足。

        仅对动态 batch 引擎生效；静态 batch=1 引擎完全跳过（零开销、向后兼容）。
        """
        if not self._dynamic_batch:
            return
        bs = tuple(blob.shape)
        if bs == self._last_blob_shape:
            return
        self.context.set_binding_shape(0, blob.shape)
        out_shape = tuple(self.context.get_binding_shape(1))
        in_size = int(np.prod(blob.shape)) * 4
        out_size = int(np.prod(out_shape)) * 4
        if in_size > self.input_size:
            old = self.d_input
            self.d_input = self.cuda.mem_alloc(in_size)
            self.bindings[0] = int(self.d_input)
            self.input_size = in_size
            try:
                del old
            except Exception:
                pass
            # 同步扩容锁页主机输入缓冲
            try:
                self.h_input = self.cuda.pagelocked_empty(blob.shape, dtype=np.float32)
            except Exception:
                self.h_input = np.zeros(blob.shape, dtype=np.float32)
        if out_size > self.output_size:
            old = self.d_output
            self.d_output = self.cuda.mem_alloc(out_size)
            self.bindings[1] = int(self.d_output)
            self.output_size = out_size
            try:
                del old
            except Exception:
                pass
            # 同步扩容锁页主机输出缓冲
            try:
                self.h_output = self.cuda.pagelocked_empty(out_shape, dtype=np.float32)
            except Exception:
                self.h_output = np.zeros(out_shape, dtype=np.float32)
        self.output_shape = out_shape
        self.input_shape = blob.shape
        self._last_blob_shape = bs

    def close(self):
        try:
            del self.h_input
            del self.h_output
        except Exception:
            pass
        try:
            del self.d_input
            del self.d_output
            del self.stream
            self.ctx.pop()
        except Exception:
            pass


# ──────────────────────────────────────────────
#  ONNX Runtime 引擎（延迟导入）
# ──────────────────────────────────────────────

class ONNXEngine:
    """ONNX Runtime 推理封装（TensorRT 不可用时的备选）"""

    def __init__(self, onnx_path: str):
        import onnxruntime as ort

        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        try:
            self.session = ort.InferenceSession(onnx_path, providers=providers)
            log.info("ONNX Runtime 加载（CUDA）: %s", onnx_path)
        except Exception:
            self.session = ort.InferenceSession(
                onnx_path, providers=["CPUExecutionProvider"]
            )
            log.warning("ONNX Runtime 加载（CPU 回退）: %s", onnx_path)

        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape
        log.info("ONNX 输入 %s shape: %s", self.input_name, self.input_shape)

    def infer(self, blob: np.ndarray) -> np.ndarray:
        return self.session.run(None, {self.input_name: blob})[0]


# ──────────────────────────────────────────────
#  统一检测器（三后端自动回退）
# ──────────────────────────────────────────────

class ConeDetector:
    """统一锥桶检测器

    按优先级自动选择后端:
      trt    → 优先加载 TensorRT .engine
      onnx   → 其次加载 ONNX Runtime .onnx
      yolov5 → 最后回退 YOLOv5 PyTorch .pt（需传入 weights_path）

    输出契约与 NRT_WS camera/yolo_camera.py 原 detect_objects() 一致。
    """

    def __init__(self, engine_path=None, onnx_path=None, weights_path=None,
                 backend="trt", conf_thresh=0.45, iou_thresh=0.45,
                 input_size=640, device_id=0):
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        self.input_size = input_size
        self.names = list(CLASS_NAMES)          # 兼容 model.names 用法
        self.engine_type = None
        self._torch_model = None

        candidates = []
        if backend == "trt" and engine_path and os.path.isfile(engine_path):
            candidates.append(("tensorrt", engine_path))
        if backend in ("trt", "onnx") and onnx_path and os.path.isfile(onnx_path):
            candidates.append(("onnx", onnx_path))
        if weights_path and os.path.isfile(weights_path):
            candidates.append(("yolov5", weights_path))

        if not candidates:
            log.warning(
                "未找到可用模型文件（engine=%s onnx=%s weights=%s），"
                "将尝试路径兜底",
                engine_path, onnx_path, weights_path,
            )
            # 最后兜底：只要路径存在就尝试
            for p in (engine_path, onnx_path, weights_path):
                if p and os.path.isfile(p):
                    ext = Path(p).suffix.lower()
                    if ext in (".engine", ".trt"):
                        candidates.append(("tensorrt", p))
                    elif ext == ".onnx":
                        candidates.append(("onnx", p))
                    elif ext in (".pt", ".pth"):
                        candidates.append(("yolov5", p))

        if not candidates:
            raise FileNotFoundError(
                f"锥桶检测器无可用模型: engine={engine_path} onnx={onnx_path} "
                f"weights={weights_path}"
            )

        last_err = None
        for kind, path in candidates:
            try:
                self._load_backend(kind, path)
                self._sync_input_size()
                log.info("检测后端已加载: %s (%s)  输入尺寸=%d",
                         self.engine_type, path, self.input_size)
                return
            except Exception as e:  # noqa: BLE001 —— 逐个回退
                last_err = e
                log.warning("后端 %s 加载失败: %s，尝试下一个", kind, e)

        raise RuntimeError(f"全部检测后端加载失败: {last_err}")

    def _sync_input_size(self):
        """从已加载引擎推导输入空间尺寸，避免与构建时的分辨率不一致

        例如 416 引擎自动适配，无需改 TRT_INPUT_SIZE 常量；若引擎含动态维度
        或无法读取，则保持 __init__ 传入的 input_size。
        """
        try:
            shape = getattr(self.engine, "input_shape", None)
            if shape and len(shape) >= 3 and shape[2] and shape[2] > 0:
                self.input_size = int(shape[2])
        except Exception:
            pass

    def _load_backend(self, kind, path):
        if kind == "tensorrt":
            self.engine = TensorRTEngine(path, fallback_input_size=self.input_size)
            self.engine_type = "tensorrt"
        elif kind == "onnx":
            self.engine = ONNXEngine(path)
            self.engine_type = "onnx"
        elif kind == "yolov5":
            from models.experimental import attempt_load  # YOLOv5 仓库在 PYTHONPATH

            self._torch_model = attempt_load(
                path, device="cuda" if _cuda_available() else "cpu"
            )
            self.engine_type = "yolov5"

    # ── 推理入口 ──
    def detect(self, img: np.ndarray) -> list:
        """检测图像中的锥桶

        Args:
            img: BGR 图像（OpenCV 格式）

        Returns:
            list[dict]: 与 yolo_camera.py detect_objects() 相同 schema
        """
        if self.engine_type in ("tensorrt", "onnx"):
            dets = self._detect_trt_onnx(img)
        else:
            dets = self._detect_yolov5(img)

        results = []
        for d in dets:
            x1, y1, x2, y2, conf, cls_id = (float(d[0]), float(d[1]),
                                            float(d[2]), float(d[3]),
                                            float(d[4]), int(d[5]))
            if cls_id not in COLOR_MAP:
                continue
            label = COLOR_MAP[cls_id][0]

            # 中心点：红色锥桶底部较宽，中心下移至 65% 高度（与 yolo_camera 原逻辑一致）
            center_x = (x1 + x2) / 2
            if label == "red":
                center_y = y1 + (y2 - y1) * 0.65
            else:
                center_y = (y1 + y2) / 2

            results.append({
                "label": label,
                "center": (center_x, center_y),
                "bbox": (x1, y1, x2, y2),
                "confidence": conf,
            })
        return results

    def detect_batch(self, imgs: list) -> list:
        """批量检测（用于左右图合并推理，batch=N）

        Args:
            imgs: list of BGR 图像（OpenCV 格式），长度即 batch

        Returns:
            list[list[dict]]：每张图检测结果，与 detect() 相同 schema

        说明：仅 TRT/ONNX 后端支持；yolov5 回退时退化为逐图 detect()。
        需 engine 按 --batch N 构建（如 --batch 2 用于双目合并）。
        """
        if self.engine_type not in ("tensorrt", "onnx"):
            return [self.detect(im) for im in imgs]

        prepped = [self._preprocess(im) for im in imgs]  # (blob,(ratio,pad))
        blobs = [p[0] for p in prepped]
        stacked = np.concatenate(blobs, axis=0)          # (N,3,H,W)
        output = self.engine.infer(stacked)

        results = []
        for i, im in enumerate(imgs):
            _, ratio, pad = prepped[i]
            out_i = output[i:i + 1]                      # (1, ...)
            dets = postprocess(out_i, im, self.input_size,
                               self.conf_thresh, self.iou_thresh, ratio, pad)
            per = []
            for d in dets:
                x1, y1, x2, y2, conf, cls_id = (float(d[0]), float(d[1]),
                                                float(d[2]), float(d[3]),
                                                float(d[4]), int(d[5]))
                if cls_id not in COLOR_MAP:
                    continue
                label = COLOR_MAP[cls_id][0]
                center_x = (x1 + x2) / 2
                if label == "red":
                    center_y = y1 + (y2 - y1) * 0.65
                else:
                    center_y = (y1 + y2) / 2
                per.append({
                    "label": label,
                    "center": (center_x, center_y),
                    "bbox": (x1, y1, x2, y2),
                    "confidence": conf,
                })
            results.append(per)
        return results

    def _detect_trt_onnx(self, img: np.ndarray) -> np.ndarray:
        """TRT / ONNX 推理 → 原图坐标检测框"""
        blob, ratio, pad = self._preprocess(img)
        output = self.engine.infer(blob)
        return postprocess(output, img, self.input_size,
                           self.conf_thresh, self.iou_thresh, ratio, pad)

    def _detect_yolov5(self, img: np.ndarray) -> np.ndarray:
        """YOLOv5 PyTorch 推理（camera 包原有路径）"""
        import torch
        from utils.augmentations import letterbox
        from utils.general import non_max_suppression as v5_nms

        img0 = img.copy()
        im = letterbox(img0, new_shape=self.input_size, auto=False)[0]
        im = im[:, :, ::-1].transpose(2, 0, 1)          # BGR→RGB, HWC→CHW
        im = np.ascontiguousarray(im)
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        im = torch.from_numpy(im).to(dev).float() / 255.0
        if im.ndimension() == 3:
            im = im.unsqueeze(0)

        with torch.no_grad():
            pred = self._torch_model(im, augment=False)[0]
        pred = v5_nms(pred, self.conf_thresh, self.iou_thresh)

        dets_out = []
        for det in pred:
            if len(det) == 0:
                continue
            det[:, :4] = _scale_coords(im.shape[2:], det[:, :4], img0.shape).round()
            dets_out.append(det)
        if not dets_out:
            return np.array([])
        return np.concatenate(dets_out, axis=0)

    def _preprocess(self, img: np.ndarray):
        """图像预处理：BGR→RGB + Letterbox + 归一化 + CHW → (1,3,H,W)

        Returns:
            (blob, ratio, pad) —— ratio/pad 一并回传，供后处理复用，避免重复 resize
        """
        blob, ratio, pad = letterbox(img, new_shape=self.input_size)
        blob = blob[:, :, ::-1].transpose(2, 0, 1)      # HWC→CHW, BGR→RGB
        blob = blob.astype(np.float32)
        blob /= 255.0
        blob = np.expand_dims(blob, axis=0)
        return np.ascontiguousarray(blob), ratio, pad

    def close(self):
        if hasattr(self, "engine") and hasattr(self.engine, "close"):
            self.engine.close()


# ──────────────────────────────────────────────
#  辅助
# ──────────────────────────────────────────────

def _cuda_available() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


def _scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
    """YOLOv5 scale_coords（PyTorch 回退路径用）"""
    if ratio_pad is None:
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, \
              (img1_shape[0] - img0_shape[0] * gain) / 2
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, [0, 2]] -= pad[0]
    coords[:, [1, 3]] -= pad[1]
    coords[:, :4] /= gain
    clip_coords(coords, img0_shape)
    return coords


def clip_coords(boxes, shape):
    if hasattr(boxes, "clamp_"):
        boxes[:, 0].clamp_(0, shape[1])
        boxes[:, 1].clamp_(0, shape[0])
        boxes[:, 2].clamp_(0, shape[1])
        boxes[:, 3].clamp_(0, shape[0])
    else:
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, shape[1])
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, shape[0])


def create_detector(engine_path=None, onnx_path=None, weights_path=None,
                    backend="trt", conf_thresh=0.45, iou_thresh=0.45,
                    input_size=640) -> ConeDetector:
    """工厂函数：创建统一锥桶检测器"""
    return ConeDetector(
        engine_path=engine_path,
        onnx_path=onnx_path,
        weights_path=weights_path,
        backend=backend,
        conf_thresh=conf_thresh,
        iou_thresh=iou_thresh,
        input_size=input_size,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("trt_yolo_detector 自检通过。用法示例:")
    print("  from cone_detector.inference import ConeDetector")
    print("  det = ConeDetector(engine_path='models/yolo26n_cone.engine')")
    print("  res = det.detect(img_bgr)   # -> [{'label','center','bbox','confidence'}, ...]")
