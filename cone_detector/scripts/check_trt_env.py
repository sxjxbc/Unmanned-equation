#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
check_trt_env.py
================
NRT_WS 相机包 TensorRT 集成环境校验脚本（Jetson 部署侧）。

用途：
  在部署目标（NVIDIA Jetson + conda yolov5 环境）上校验
  cone_detector TensorRT 后端是否可用，避免运行时才发现缺依赖。

校验项：
  1. Python / 关键库版本（tensorrt / pycuda / onnxruntime / opencv / numpy / torch）
  2. GPU 可用性（pycuda 或 nvidia-smi）
  3. 推理模型文件是否存在（.engine / .onnx / .pt）
  4. 环境变量覆盖提示（NRT_DETECTOR / NRT_TRT_ENGINE / NRT_ONNX_MODEL）

用法（在 Jetson 的 conda yolov5 环境）:
  conda activate yolov5
  python scripts/check_trt_env.py
  python scripts/check_trt_env.py --engine /path/to/yolo26n_cone.engine
"""

import argparse
import os
import shutil
import sys
from pathlib import Path

PASS, WARN, FAIL = "✓", "⚠", "✗"


def check_import(name: str, ver_attr: str = "__version__") -> str:
    """尝试导入模块，返回 (状态, 版本)"""
    try:
        mod = __import__(name)
        ver = getattr(mod, ver_attr, "未知")
        return PASS, str(ver)
    except ImportError:
        return FAIL, "未安装"
    except Exception as e:  # noqa: BLE001
        return WARN, f"导入异常: {e}"


def check_gpu() -> str:
    """检测 GPU 可用性"""
    # 1) pycuda
    try:
        import pycuda.driver as cuda
        cuda.init()
        n = cuda.Device.count()
        if n == 0:
            return FAIL, "pycuda 可用但无 CUDA 设备"
        names = [cuda.Device(i).name() for i in range(n)]
        return PASS, f"{n} 个设备: {', '.join(names)}"
    except ImportError:
        pass
    except Exception as e:  # noqa: BLE001
        return WARN, f"pycuda 报错: {e}"

    # 2) nvidia-smi
    smi = shutil.which("nvidia-smi")
    if smi:
        return WARN, "pycuda 未装，但 nvidia-smi 存在（TensorRT 推理仍需 pycuda）"
    return FAIL, "pycuda 与 nvidia-smi 均不可用"


def check_files(engine_path, onnx_path, weights_path) -> list:
    """检查模型文件是否存在"""
    rows = []
    for label, p in [("TensorRT .engine", engine_path),
                     ("ONNX .onnx", onnx_path),
                     ("YOLOv5 .pt", weights_path)]:
        if not p:
            rows.append((label, WARN, "未指定"))
            continue
        p = Path(p)
        if p.is_file():
            rows.append((label, PASS, f"{p} ({p.stat().st_size/1024/1024:.1f} MB)"))
        else:
            rows.append((label, FAIL, f"不存在: {p}"))
    return rows


def main():
    ap = argparse.ArgumentParser(description="NRT_WS TensorRT 集成环境校验")
    ap.add_argument("--engine", default=None, help="TensorRT .engine 路径")
    ap.add_argument("--onnx", default=None, help="ONNX .onnx 路径")
    ap.add_argument("--weights", default=None,
                    help="YOLOv5 .pt 路径（PyTorch 回退）")
    args = ap.parse_args()

    # 默认路径推导（与 yolo_camera.py 同规则）
    here = Path(__file__).resolve().parent          # scripts/
    pkg_dir = here.parent                            # cone_detector/
    src_dir = pkg_dir.parent                         # src/
    cam_dir = src_dir / "camera"

    engine = args.engine or os.environ.get(
        "NRT_TRT_ENGINE", str(pkg_dir / "models" / "yolo26n_cone.engine")
    )
    onnx = args.onnx or os.environ.get(
        "NRT_ONNX_MODEL", str(pkg_dir / "models" / "yolo26n_cone.onnx")
    )
    weights = args.weights or str(cam_dir / "weights1" / "best.pt")

    print("=" * 60)
    print("  NRT_WS 相机包 TensorRT 集成环境校验")
    print("=" * 60)
    print(f"  Python:        {sys.version.split()[0]}  ({sys.executable})")
    print()

    # 1) 关键库
    print("── 推理依赖 ──────────────────────────────")
    checks = [
        ("tensorrt", "__version__"),
        ("pycuda", "__version__"),
        ("onnxruntime", "__version__"),
        ("cv2", "__version__"),
        ("numpy", "__version__"),
        ("torch", "__version__"),
    ]
    lib_rows = [(name,) + check_import(name, attr) for name, attr in checks]
    for name, st, ver in lib_rows:
        print(f"  {st} {name:<14} {ver}")
    trt_ok = lib_rows[0][1] == PASS
    pycuda_ok = lib_rows[1][1] == PASS
    onnx_ok = lib_rows[2][1] == PASS
    print()

    # 2) GPU
    print("── GPU ───────────────────────────────────")
    st, info = check_gpu()
    print(f"  {st} {info}")
    print()

    # 3) 模型文件
    print("── 模型文件 ──────────────────────────────")
    for label, st, info in check_files(engine, onnx, weights):
        print(f"  {st} {label:<16} {info}")
    print()

    # 4) 结论
    print("── 结论 ──────────────────────────────────")
    n_fail = 0
    if trt_ok and pycuda_ok:
        print("  [TRT]  TensorRT + pycuda 齐备，可用 .engine 后端（推荐）")
    elif onnx_ok:
        print("  [ONNX] TensorRT/pycuda 缺失，将回退 ONNX Runtime 后端")
        print("        建议: conda install -c conda-forge tensorrt pycuda")
        n_fail += 1
    else:
        print("  [FAIL] 推理后端不可用，将回退 YOLOv5 PyTorch（无加速）")
        print("        建议: conda install -c conda-forge tensorrt pycuda")
        print("              pip install onnxruntime")
        n_fail += 1

    if not Path(engine).is_file() and not Path(onnx).is_file():
        print("  [WARN] 无 .engine/.onnx 模型，运行将回退 YOLOv5 PyTorch")
        print("        权重导出与转换（在 PC 或 Jetson 上）:")
        print("          python export_to_onnx.py --weights best.pt \\")
        print("              --output models/yolo26n_cone.onnx --imgsz 640")
        print("          python convert_to_tensorrt.py --onnx models/yolo26n_cone.onnx \\")
        print("              --output models/yolo26n_cone.engine --fp16 --imgsz 640")

    print()
    print("  环境变量覆盖:")
    print("    NRT_DETECTOR  = trt | onnx | yolov5   （强制后端）")
    print("    NRT_TRT_ENGINE = TensorRT 引擎路径")
    print("    NRT_ONNX_MODEL = ONNX 模型路径")
    print()
    if n_fail:
        print("  结果: 存在缺失项，请按上方建议安装后重跑。")
        sys.exit(1)
    print("  结果: 全部通过 ✓  可以启动 yolo_camera.py（TRT 后端）")


if __name__ == "__main__":
    main()
