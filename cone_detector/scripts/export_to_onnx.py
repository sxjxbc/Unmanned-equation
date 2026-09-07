#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
export_to_onnx.py
=================
将 PyTorch YOLO 权重（.pt）导出为 ONNX 格式，支持：

  1. Ultralytics YOLO  (yolo8n ~ yolo11n / yolo26n)
  2. YOLOv6            (meituan/YOLOv6)
  3. 通用 torch.onnx.export

导出的 ONNX 会经过 onnxsim 简化，再进入 TensorRT 转换流程。

用法:
  # Ultralytics YOLO26n → ONNX（推荐）
  python export_to_onnx.py \
      --weights runs/train/cone_exp/weights/best.pt \
      --output models/yolo26n_cone.onnx \
      --imgsz 640 --batch 1

  # YOLOv6n → ONNX
  python export_to_onnx.py \
      --weights yolov6n_cone.pt \
      --output models/yolov6n_cone.onnx \
      --source yolov6 --simplify

依赖:
  pip install torch torchvision onnx onnxsim ultralytics
  # YOLOv6: pip install yolov6
"""

import argparse
import os
import shutil
import sys
from pathlib import Path

import torch
import onnx


# ──────────────────────────────────────────────
#  工具函数
# ──────────────────────────────────────────────

def check_input(weights: str) -> Path:
    p = Path(weights)
    if not p.exists():
        print(f"[ERROR] 权重文件不存在: {p}")
        sys.exit(1)
    print(f"[export] 输入权重: {p}  大小: {p.stat().st_size / 1024 / 1024:.1f} MB")
    return p


def verify_onnx(onnx_path: str) -> bool:
    """验证 ONNX 模型可读性"""
    try:
        model = onnx.load(onnx_path)
        onnx.checker.check_model(model)
        print(f"[export] ✓ ONNX 验证通过: {onnx_path}")
        return True
    except Exception as e:
        print(f"[export] ✗ ONNX 验证失败: {e}")
        return False


def simplify_onnx(onnx_path: str, force: bool = True) -> str:
    """使用 onnx-simplifier 简化计算图"""
    try:
        import onnxsim
    except ImportError:
        print("[export] onnxsim 未安装，跳过简化（pip install onnxsim）")
        return onnx_path

    try:
        print("[export] 正在简化 ONNX 图...")
        model = onnx.load(onnx_path)
        model_sim, ok = onnxsim.simplify(
            model,
            input_shapes={"images": [1, 3, 640, 640]},
            dynamic_input_shape=True,
        )
        if ok:
            sim_path = onnx_path.replace(".onnx", "_sim.onnx")
            onnx.save(model_sim, sim_path)
            orig_size = Path(onnx_path).stat().st_size
            sim_size = Path(sim_path).stat().st_size
            ratio = sim_size / orig_size * 100
            print(f"[export] ✓ ONNX 简化成功: {sim_path}")
            print(f"[export]   原始: {orig_size/1024/1024:.1f} MB  "
                  f"简化后: {sim_size/1024/1024:.1f} MB  ({ratio:.0f}%)")
            if force:
                shutil.move(sim_path, onnx_path)
                print(f"[export]   已覆盖原文件: {onnx_path}")
                return onnx_path
            return sim_path
        else:
            print("[export] ✗ ONNX 简化失败，使用原始文件")
            return onnx_path
    except Exception as e:
        print(f"[export] onnxsim 出错: {e}，跳过简化")
        return onnx_path


def print_model_info(onnx_path: str):
    """打印 ONNX 模型基本信息"""
    try:
        model = onnx.load(onnx_path)
        graph = model.graph
        print("[export] === ONNX 模型信息 ===")
        print(f"  IR 版本:      {model.ir_version}")
        print(f"  Opset 版本:   {model.opset_import[0].version}")
        print(f"  输入节点:     {[i.name for i in graph.input]}")
        print(f"  输出节点:     {[o.name for o in graph.output]}")
        print(f"  节点总数:     {len(graph.node)}")
        print("=" * 26)
    except Exception:
        pass


# ──────────────────────────────────────────────
#  导出路径
# ──────────────────────────────────────────────

def export_ultralytics(weights: str, output: str, imgsz: int,
                       simplify: bool, dynamic: bool) -> str:
    """
    Ultralytics YOLO（YOLO8 / YOLO11 / YOLO26）导出路径
    使用 model.export(format='onnx') 自动处理 NMS 后处理图
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        print("[export] ultralytics 未安装，无法使用 Ultralytics 导出路径")
        sys.exit(1)

    print(f"[export] 使用 Ultralytics 导出路径 (支持 yolo8/11/26)")
    model = YOLO(weights)

    # Ultralytics 导出的 ONNX 默认位置
    default_out = str(Path(weights).with_suffix(".onnx"))

    # 执行导出
    exported = model.export(
        format="onnx",
        imgsz=imgsz,
        simplify=simplify,
        dynamic=dynamic,
        opset=12,
    )

    # 将导出文件移动到目标路径
    exported = str(exported)
    if exported != output:
        os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
        shutil.move(exported, output)
        print(f"[export] 已移动到: {output}")
    else:
        print(f"[export] 导出路径: {output}")

    return output


def export_yolov6(weights: str, output: str, imgsz: int,
                   batch: int, dynamic: bool) -> str:
    """YOLOv6 官方导出路径"""
    try:
        # 尝试 yolov6 包
        sys.path.insert(0, str(Path(weights).parent.parent))
        from yolov6.models.yolo import Model as YOLOv6Model
        print("[export] 使用 YOLOv6 官方模型加载")

        ckpt = torch.load(weights, map_location="cpu")
        model = ckpt.get("model", ckpt)

        if hasattr(model, "float"):
            model = model.float()
        model.eval()

        dummy = torch.zeros(1 if not dynamic else batch, 3, imgsz, imgsz)
        dynamic_axes = {
            "images": {0: "batch"} if dynamic else {},
            "output": {0: "batch"} if dynamic else {},
        } if dynamic else None

        torch.onnx.export(
            model, dummy, output,
            opset_version=12,
            input_names=["images"],
            output_names=["output"],
            dynamic_axes=dynamic_axes,
        )
        print(f"[export] YOLOv6 ONNX 导出完成: {output}")
        return output
    except ImportError:
        print("[export] yolov6 包未找到，尝试通用路径")
        return export_generic(weights, output, imgsz, batch, dynamic)


def export_generic(weights: str, output: str, imgsz: int,
                   batch: int, dynamic: bool) -> str:
    """通用 torch.onnx.export（兼容任意 PyTorch 模型）"""
    print("[export] 使用通用 torch.onnx 导出路径")

    ckpt = torch.load(weights, map_location="cpu")
    model = ckpt.get("model", ckpt)

    if hasattr(model, "float"):
        model = model.float()
    if hasattr(model, "eval"):
        model.eval()

    dummy = torch.zeros(batch if dynamic else 1, 3, imgsz, imgsz)

    dynamic_axes = {
        "images": {0: "batch"},
        "output": {0: "batch"},
    } if dynamic else None

    torch.onnx.export(
        model, dummy, output,
        opset_version=12,
        input_names=["images"],
        output_names=["output"],
        dynamic_axes=dynamic_axes,
    )
    print(f"[export] 通用 ONNX 导出完成: {output}")
    return output


def auto_detect_source(weights: str) -> str:
    """根据权重文件名自动推断模型来源"""
    w = Path(weights).stem.lower()
    if w.startswith("yolo8") or w.startswith("yolov8"):
        return "ultralytics"
    if w.startswith("yolo11") or w.startswith("yolov11"):
        return "ultralytics"
    if w.startswith("yolo26") or w.startswith("yolov26"):
        return "ultralytics"
    if w.startswith("yolov6"):
        return "yolov6"
    # 尝试 ultralytics（最通用）
    return "ultralytics"


# ──────────────────────────────────────────────
#  主函数
# ──────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="PyTorch YOLO → ONNX 导出工具（支持 yolo8/11/26）"
    )
    parser.add_argument(
        "--weights", required=True,
        help="PyTorch 权重路径 (.pt)"
    )
    parser.add_argument(
        "--output", required=True,
        help="ONNX 输出路径 (.onnx)"
    )
    parser.add_argument(
        "--imgsz", type=int, default=640,
        help="模型输入尺寸（默认 640）"
    )
    parser.add_argument(
        "--batch", type=int, default=1,
        help="批大小（默认 1，动态 batch 建议 0）"
    )
    parser.add_argument(
        "--source", default="auto",
        choices=["auto", "ultralytics", "yolov6", "generic"],
        help="模型来源（auto=自动检测）"
    )
    parser.add_argument(
        "--simplify", action="store_true", default=True,
        help="导出后用 onnxsim 简化（默认开启）"
    )
    parser.add_argument(
        "--no-simplify", dest="simplify", action="store_false",
        help="禁用 onnxsim 简化"
    )
    parser.add_argument(
        "--dynamic", action="store_true", default=True,
        help="启用动态 batch/分辨率（默认开启，方便 TensorRT 转换）"
    )
    parser.add_argument(
        "--no-dynamic", dest="dynamic", action="store_false",
        help="禁用动态 shape"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # 前置检查
    check_input(args.weights)

    # 输出目录
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    print("=" * 55)
    print("  YOLO → ONNX 导出工具")
    print("=" * 55)
    source = args.source if args.source != "auto" else auto_detect_source(args.weights)
    print(f"  权重文件:     {args.weights}")
    print(f"  输出路径:     {args.output}")
    print(f"  输入尺寸:     {args.imgsz}×{args.imgsz}")
    print(f"  批大小:       {args.batch if not args.dynamic else 'dynamic'}")
    print(f"  模型来源:     {source}")
    print(f"  简化 ONNX:   {'是' if args.simplify else '否'}")
    print(f"  动态 shape:  {'是' if args.dynamic else '否'}")
    print("=" * 55)

    # 导出
    if source == "ultralytics":
        out_onnx = export_ultralytics(
            args.weights, args.output,
            args.imgsz, args.simplify, args.dynamic
        )
    elif source == "yolov6":
        out_onnx = export_yolov6(
            args.weights, args.output,
            args.imgsz, args.batch, args.dynamic
        )
    else:
        out_onnx = export_generic(
            args.weights, args.output,
            args.imgsz, args.batch, args.dynamic
        )

    # 验证
    if not verify_onnx(out_onnx):
        sys.exit(1)

    # 打印模型信息
    print_model_info(out_onnx)

    # 手动简化（仅 Ultralytics 路径，且 simplify=True 但导出时未简化）
    if args.simplify and source != "ultralytics":
        simplify_onnx(out_onnx)

    print("[export] ✓ 导出完成！")
    print(f"[export] 路径: {out_onnx}")
    print()
    print("  下一步 → TensorRT 转换:")
    print(f"  python scripts/convert_to_tensorrt.py \\")
    print(f"      --onnx {out_onnx} \\")
    print(f"      --output models/yolo26n_cone.engine \\")
    print(f"      --fp16 --imgsz {args.imgsz}")
    print()


if __name__ == "__main__":
    main()
