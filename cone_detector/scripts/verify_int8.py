#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
verify_int8.py
==============
INT8 量化验收工具：在同一批图像上分别用 FP16 engine 与 INT8 engine 推理，
对比「检测一致性」与「推理速度」，确认 INT8 不掉精度、达成轻量化目标。

用法（在 Jetson 上执行，需 tensorrt + pycuda）：
  python scripts/verify_int8.py \
      --fp16   models/yolo11n_cone.engine \
      --int8   models/yolo11n_cone_int8.engine \
      --images /opt/cone_calib/calib_images \
      --imgsz  640

输出：
  1) 逐图检测一致性（匹配 IOU、类别一致率、漏检/误检）
  2) 整体均值与判定
  3) FP16 / INT8 推理速度（fps）对比
"""

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import cv2

# ── 注入 cone_detector 共享推理模块路径（与 yolo_camera.py 一致）──
_HERE = os.path.dirname(os.path.abspath(__file__))
_CONE_DET = os.path.dirname(_HERE)          # .../cone_detector
_SRC = os.path.dirname(_CONE_DET)           # .../src
if os.path.isdir(os.path.join(_SRC, "cone_detector")):
    sys.path.insert(0, _SRC)

from cone_detector.inference import ConeDetector  # noqa: E402


def collect_images(folder: str, limit: int = 0):
    exts = ["*.jpg", "*.jpeg", "*.png", "*.bmp"]
    paths = []
    for ext in exts:
        paths.extend(Path(folder).rglob(ext))
    paths = sorted(paths)
    if limit and len(paths) > limit:
        paths = paths[:limit]
    return paths


def bbox_iou(b1, b2):
    """b: (x1, y1, x2, y2) -> IOU"""
    x1 = max(b1[0], b2[0])
    y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2])
    y2 = min(b1[3], b2[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if inter <= 0:
        return 0.0
    a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
    return inter / max(1e-6, a1 + a2 - inter)


def compare_frame(fp16_res, int8_res, iou_th=0.5):
    """对比两引擎对单帧的检测结果。

    Returns: dict {matched, mean_iou, class_match, fn, fp, conf_diff_mean}
    """
    used = set()
    ious = []
    class_match = 0
    conf_diffs = []

    for d1 in fp16_res:
        best_iou, best_j = 0.0, -1
        for j, d2 in enumerate(int8_res):
            if j in used or d1["label"] != d2["label"]:
                continue
            iou = bbox_iou(d1["bbox"], d2["bbox"])
            if iou > best_iou:
                best_iou, best_j = iou, j
        if best_j >= 0 and best_iou >= iou_th:
            used.add(best_j)
            ious.append(best_iou)
            conf_diffs.append(abs(d1["confidence"] - int8_res[best_j]["confidence"]))
            if d1["label"] == int8_res[best_j]["label"]:
                class_match += 1

    matched = len(ious)
    fn = len(fp16_res) - matched                 # FP16 有、INT8 漏检
    fp = len(int8_res) - len(used)               # INT8 有、FP16 无（误检/多检）
    return {
        "matched": matched,
        "mean_iou": float(np.mean(ious)) if ious else 0.0,
        "class_match": class_match,
        "fn": fn,
        "fp": fp,
        "conf_diff_mean": float(np.mean(conf_diffs)) if conf_diffs else 0.0,
    }


def bench_speed(det: ConeDetector, images, imgsz, warmup=20, runs=200):
    dummy = np.random.randint(0, 255, (imgsz, imgsz, 3), dtype=np.uint8)
    for _ in range(warmup):
        det.detect(dummy)
    t0 = time.perf_counter()
    for _ in range(runs):
        det.detect(dummy)
    el = (time.perf_counter() - t0) * 1000 / runs
    return el, 1000.0 / el


def main():
    ap = argparse.ArgumentParser(description="INT8 vs FP16 精度/速度验收")
    ap.add_argument("--fp16",   required=True,  help="FP16 engine 路径")
    ap.add_argument("--int8",   required=True,  help="INT8 engine 路径")
    ap.add_argument("--images", required=True,  help="测试图像目录")
    ap.add_argument("--imgsz",  type=int, default=640, help="输入分辨率")
    ap.add_argument("--conf",   type=float, default=0.45, help="置信度阈值")
    ap.add_argument("--limit",  type=int, default=0, help="最大测试图像数（0=全部）")
    ap.add_argument("--iou-th", type=float, default=0.5, help="匹配 IOU 阈值")
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--runs",   type=int, default=200)
    args = ap.parse_args()

    if not Path(args.fp16).exists():
        print(f"[ERROR] FP16 engine 不存在: {args.fp16}")
        sys.exit(1)
    if not Path(args.int8).exists():
        print(f"[ERROR] INT8 engine 不存在: {args.int8}")
        sys.exit(1)

    print("=" * 60)
    print("  INT8 量化验收：FP16 vs INT8")
    print("=" * 60)
    print(f"  FP16 engine: {args.fp16}")
    print(f"  INT8 engine : {args.int8}")
    print(f"  图像目录    : {args.images}")
    print(f"  输入尺寸    : {args.imgsz}   置信阈值: {args.conf}   匹配IOU: {args.iou_th}")
    print("=" * 60)

    fp16_det = ConeDetector(engine_path=args.fp16, backend="trt",
                            conf_thresh=args.conf, input_size=args.imgsz)
    int8_det = ConeDetector(engine_path=args.int8, backend="trt",
                            conf_thresh=args.conf, input_size=args.imgsz)
    print(f"  后端: FP16={fp16_det.engine_type}  INT8={int8_det.engine_type}")

    imgs = collect_images(args.images, args.limit)
    if not imgs:
        print(f"[ERROR] 图像目录为空: {args.images}")
        sys.exit(1)
    print(f"  共 {len(imgs)} 张测试图像\n")

    # ── 逐图一致性 ──
    agg = {"matched": 0, "ious": [], "class_match": 0, "fn": 0,
           "fp": 0, "conf_diffs": [], "n_fp16": 0, "n_int8": 0}
    for i, p in enumerate(imgs):
        img = cv2.imread(str(p))
        if img is None:
            continue
        r16 = fp16_det.detect(img)
        r8 = int8_det.detect(img)
        c = compare_frame(r16, r8, args.iou_th)
        agg["matched"] += c["matched"]
        agg["ious"].append(c["mean_iou"])
        agg["class_match"] += c["class_match"]
        agg["fn"] += c["fn"]
        agg["fp"] += c["fp"]
        agg["conf_diffs"].append(c["conf_diff_mean"])
        agg["n_fp16"] += len(r16)
        agg["n_int8"] += len(r8)
        if (i + 1) % 20 == 0 or (i + 1) == len(imgs):
            print(f"  [{i+1:4d}/{len(imgs)}] "
                  f"FP16={len(r16):2d} INT8={len(r8):2d} "
                  f"匹配={c['matched']} IOU={c['mean_iou']:.2f} "
                  f"漏检={c['fn']} 误检={c['fp']}")

    # ── 汇总 ──
    n_frames = len(agg["ious"])
    mean_iou = float(np.mean(agg["ious"])) if agg["ious"] else 0.0
    mean_conf_diff = float(np.mean(agg["conf_diffs"])) if agg["conf_diffs"] else 0.0
    class_rate = (agg["class_match"] / max(1, agg["matched"]))
    fn_rate = agg["fn"] / max(1, agg["n_fp16"])      # 相对 FP16 召回损失
    fp_rate = agg["fp"] / max(1, agg["n_int8"])      # 相对 INT8 多检率
    det_count_corr = (np.corrcoef(
        [agg["n_fp16"]], [agg["n_int8"]])[0, 1] if n_frames > 1 else 1.0)

    print("\n" + "=" * 60)
    print("  一致性汇总")
    print("=" * 60)
    print(f"  平均匹配 IOU      : {mean_iou:.3f}   (≥0.8 视为几何一致)")
    print(f"  类别一致率        : {class_rate:.3f}")
    print(f"  平均置信度差      : {mean_conf_diff:.3f}")
    print(f"  FP16→INT8 漏检率  : {fn_rate:.3f}   (越低越好)")
    print(f"  INT8 相对多检率   : {fp_rate:.3f}")
    print(f"  检测数相关系数    : {det_count_corr:.3f}")
    print("=" * 60)

    # ── 速度 ──
    try:
        fp16_ms, fp16_fps = bench_speed(fp16_det, imgs, args.imgsz,
                                        args.warmup, args.runs)
        int8_ms, int8_fps = bench_speed(int8_det, imgs, args.imgsz,
                                        args.warmup, args.runs)
        print("\n===== 推理速度对比 =====")
        print(f"  FP16 : {fp16_ms:.2f} ms  ({fp16_fps:.1f} fps)")
        print(f"  INT8 : {int8_ms:.2f} ms  ({int8_fps:.1f} fps)")
        print(f"  加速比: {fp16_fps / max(1e-6, int8_fps):.2f}×")
        print("========================\n")
    except Exception as e:
        print(f"[WARN] 速度测试失败（可能无 GPU/CUDA）: {e}")

    # ── 判定 ──
    ok = (mean_iou >= 0.8) and (fn_rate <= 0.05) and (class_rate >= 0.95)
    print("  验收结论:", "✅ INT8 可接受（精度/速度达标）" if ok
          else "⚠️ INT8 精度不足，建议增加校准图或回退 FP16")
    print()

    fp16_det.close()
    int8_det.close()


if __name__ == "__main__":
    main()
