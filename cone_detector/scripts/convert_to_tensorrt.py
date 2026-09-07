#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
convert_to_tensorrt.py
======================
将 ONNX 模型转换为 TensorRT .engine 文件，支持三种精度模式：

  FP32  — 标准精度，默认
  FP16  — 半精度加速，Volta+ GPU 均支持（Jetson AGX / RTX 系列）
  INT8  — 8 位整数量化，需要校准数据集

用法（推荐 FP16）:
  python convert_to_tensorrt.py \
      --onnx    models/yolo26n_cone_sim.onnx \
      --output  models/yolo26n_cone.engine \
      --fp16 \
      --imgsz   640

INT8 用法（需要校准数据）:
  python convert_to_tensorrt.py \
      --onnx    models/yolo26n_cone_sim.onnx \
      --output  models/yolo26n_cone_int8.engine \
      --int8 \
      --calib-dir  scripts/train/dataset/images/val \
      --calib-num  100 \
      --imgsz   640

依赖（Jetson 已预装）:
  TensorRT >= 7.0
  pip install tensorrt pycuda

注意：
  .engine 文件必须在目标 GPU 架构上构建，不可跨平台迁移。
"""

import argparse
import os
import shutil
import sys
import time
from pathlib import Path
from typing import List, Optional

# ──────────────────────────────────────────────
#  TensorRT INT8 校准器（用于 PTQ 量化）
# ──────────────────────────────────────────────

class YOLOInt8Calibrator:
    """
    YOLO 系列模型的 INT8 校准器

    继承自 TensorRT IInt8EntropyCalibrator2，
    自动从校准图像目录中读取样本，计算激活分布直方图，
    用于 INT8 量化因子求解。
    """

    def __init__(self, image_dir: str, input_shape: tuple,
                 batch_size: int = 1, max_calib_images: int = 500,
                 cache_file: str = "calib_cache.bin"):
        """
        Args:
            image_dir:    校准图像目录
            input_shape:  (C, H, W)
            batch_size:  校准批大小
            max_calib_images: 最大校准样本数
            cache_file:   校准缓存路径（TensorRT 缓存后可跳过重复校准）
        """
        import cv2
        import numpy as np

        self.image_dir = Path(image_dir)
        self.input_shape = input_shape  # (C, H, W)
        self.batch_size = batch_size
        self.max_calib_images = max_calib_images
        self.cache_file = cache_file
        self.batch = None
        self._data_generator = self._build_data_generator()
        self.image_count = 0

        # 收集所有图像路径
        self.image_paths: List[Path] = []
        for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp"]:
            self.image_paths.extend(self.image_dir.rglob(ext))
        self.image_paths = sorted(self.image_paths)[:max_calib_images]

        if not self.image_paths:
            print(f"[calib] 警告: 校准目录为空: {image_dir}")
            print("[calib] INT8 校准将使用默认阈值，可能精度下降")

        print(f"[calib] 找到 {len(self.image_paths)} 张校准图像")

    def _preprocess(self, image_path: Path) -> "np.ndarray":
        """YOLO 标准预处理：Letterbox 缩放"""
        import cv2
        import numpy as np

        img = cv2.imread(str(image_path))
        if img is None:
            return np.zeros((self.input_shape[1], self.input_shape[2], 3), dtype=np.uint8)

        h, w = img.shape[:2]
        target_h, target_w = self.input_shape[1], self.input_shape[2]
        scale = min(target_w / w, target_h / h)
        new_w, new_h = int(w * scale), int(h * scale)

        resized = cv2.resize(img, (new_w, new_h))
        canvas = np.full((target_h, target_w, 3), 114, dtype=np.uint8)
        dx = (target_w - new_w) // 2
        dy = (target_h - new_h) // 2
        canvas[dy:dy + new_h, dx:dx + new_w] = resized

        # BGR→RGB，HWC→CHW，/255.0 归一化
        canvas = canvas[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
        return canvas

    def _build_data_generator(self):
        """无限循环数据生成器"""
        import numpy as np
        C, H, W = self.input_shape
        while True:
            batch = np.zeros((self.batch_size, C, H, W), dtype=np.float32)
            for i in range(self.batch_size):
                if self.image_paths:
                    path = self.image_paths[self.image_count % len(self.image_paths)]
                    batch[i] = self._preprocess(path)
                    self.image_count += 1
                else:
                    batch[i] = np.random.randn(C, H, W).astype(np.float32) * 0.01
            yield batch

    def get_batch(self, names: list, **kwargs) -> list:
        """TensorRT 每轮校准回调，返回一个 batch 的输入"""
        try:
            self.batch = next(self._data_generator)
            import ctypes
            return [int(ctypes.addressof(self.batch.ctypes.data))]
        except StopIteration:
            self._data_generator = self._build_data_generator()
            return self.get_batch(names)

    def get_batch_size(self) -> int:
        return self.batch_size

    def read_calibration_cache(self) -> Optional[bytes]:
        """读取校准缓存（加速重复构建）"""
        if Path(self.cache_file).exists():
            print(f"[calib] 读取校准缓存: {self.cache_file}")
            with open(self.cache_file, "rb") as f:
                return f.read()
        return None

    def write_calibration_cache(self, cache: bytes) -> None:
        """写入校准缓存"""
        with open(self.cache_file, "wb") as f:
            f.write(cache)
        print(f"[calib] 校准缓存已保存: {self.cache_file}")


# ──────────────────────────────────────────────
#  TensorRT Engine 构建
# ──────────────────────────────────────────────

def check_trt() -> tuple:
    """检查 TensorRT 是否可用，返回版本号"""
    try:
        import tensorrt as trt
        version = tuple(int(x) for x in trt.__version__.split(".")[:2])
        print(f"[trt] TensorRT 版本: {trt.__version__}  (major={version[0]})")
        return trt, version
    except ImportError:
        print("[trt] ✗ TensorRT 未安装")
        print("  Jetson: 已预装，source /usr/local/cuda/setup.bash 后重试")
        print("  Desktop: pip install tensorrt（或下载 deb 安装包）")
        sys.exit(1)


def build_engine(onnx_path: str, engine_path: str,
                 fp16: bool, int8: bool, int8_calibrator,
                 workspace_mb: int,
                 min_shape: tuple, opt_shape: tuple, max_shape: tuple,
                 verbose: bool) -> bool:
    """
    使用 TensorRT Python API 构建 engine
    """
    import tensorrt as trt

    trt, trt_version = check_trt()

    log_level = trt.Logger.VERBOSE if verbose else trt.Logger.WARNING
    logger = trt.Logger(log_level)

    explicit_batch = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = trt.Builder(logger).create_network(explicit_batch)
    parser = trt.OnnxParser(network, logger)

    print(f"[trt] 解析 ONNX: {onnx_path}")
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(f"[trt] 解析错误 {i}: {parser.get_error(i)}")
            return False

    builder = trt.Builder(logger)
    config = builder.create_builder_config()
    config.max_workspace_size = workspace_mb * (1 << 20)

    precision_flags = []
    if fp16:
        if builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
            precision_flags.append("FP16")
            print("[trt] ✓ FP16 模式已启用")
        else:
            print("[trt] ⚠ 当前平台不支持快速 FP16")

    if int8:
        if builder.platform_has_fast_int8:
            config.set_flag(trt.BuilderFlag.INT8)
            precision_flags.append("INT8")
            print("[trt] ✓ INT8 模式已启用")
        else:
            print("[trt] ⚠ 当前平台不支持 INT8")

    if not precision_flags:
        precision_flags = ["FP32"]
        print("[trt] 使用 FP32 精度")

    if int8 and int8_calibrator is not None:
        config.int8_calibrator = int8_calibrator
        print("[trt] ✓ INT8 校准器已绑定")

    input_tensor = network.get_input(0)
    input_name = input_tensor.name

    profile = builder.create_optimization_profile()
    profile.set_shape(input_name, min_shape, opt_shape, max_shape)
    config.add_optimization_profile(profile)
    print(f"[trt] Input shape:  min={min_shape}  opt={opt_shape}  max={max_shape}")

    print(f"[trt] 开始构建 engine（{'/'.join(precision_flags)}，"
          f"workspace={workspace_mb}MB，可能需要 3~10 分钟）...")
    t0 = time.time()

    if trt_version >= (8, 0):
        serialized = builder.build_serialized_network(network, config)
        if serialized is None:
            print("[trt] ✗ Engine 构建失败！")
            return False
        with open(engine_path, "wb") as f:
            f.write(serialized)
    else:
        engine = builder.build_engine(network, config)
        if engine is None:
            print("[trt] ✗ Engine 构建失败！")
            return False
        with open(engine_path, "wb") as f:
            f.write(engine.serialize())

    elapsed = time.time() - t0
    size_mb = os.path.getsize(engine_path) / (1 << 20)
    print(f"[trt] ✓ 构建完成！耗时: {elapsed:.1f}s  大小: {size_mb:.1f} MB")
    print(f"[trt] 保存路径: {engine_path}")
    return True


def build_via_trtexec(onnx_path: str, engine_path: str,
                      fp16: bool, int8: bool,
                      workspace_mb: int,
                      min_str: str, opt_str: str, max_str: str,
                      int8_calib: Optional[str] = None) -> bool:
    """通过 trtexec 命令行工具构建（更稳定，支持更多优化选项）"""
    trtexec = shutil.which("trtexec") or "/usr/src/tensortr/bin/trtexec"
    # 尝试常见路径
    for path in [
        "/usr/src/tensorrt/bin/trtexec",
        "/usr/local/tensorrt/bin/trtexec",
        "trtexec",
    ]:
        if os.path.exists(path):
            trtexec = path
            break

    if not os.path.exists(trtexec):
        print(f"[trt] ✗ trtexec 未找到（搜索了: {trtexec}）")
        print("[trt] 请确认 TensorRT 已安装且路径在 $PATH 中")
        return False

    cmd = [
        trtexec,
        f"--onnx={onnx_path}",
        f"--saveEngine={engine_path}",
        f"--workspace={workspace_mb}",
        f"--minShapes=images:{min_str}",
        f"--optShapes=images:{opt_str}",
        f"--maxShapes=images:{max_str}",
    ]
    if fp16:
        cmd.append("--fp16")
    if int8:
        cmd.append("--int8")
        if int8_calib:
            cmd.append(f"--calib={int8_calib}")

    print(f"[trt] 执行: {' '.join(cmd)}")
    ret = os.system(" ".join(cmd))
    if ret == 0:
        print(f"[trt] ✓ trtexec 完成: {engine_path}")
        return True
    print(f"[trt] ✗ trtexec 失败 (code={ret})")
    return False


def verify_engine(engine_path: str) -> bool:
    """验证 engine 可以正常反序列化"""
    try:
        import tensorrt as trt
        logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, "rb") as f:
            engine = trt.Runtime(logger).deserialize_cuda_engine(f.read())
        if engine is None:
            print("[verify] ✗ Engine 反序列化失败")
            return False
        print(f"[verify] ✓ Engine 反序列化成功，{engine.num_bindings} 个 I/O 绑定")
        inputs  = [engine.get_binding_name(i) for i in range(engine.num_bindings) if engine.binding_is_input(i)]
        outputs = [engine.get_binding_name(i) for i in range(engine.num_bindings) if not engine.binding_is_input(i)]
        print(f"[verify]   输入: {inputs}")
        print(f"[verify]   输出: {outputs}")
        return True
    except ImportError:
        print("[verify] ⚠ pycuda 未安装，跳过 engine 验证")
        return True


def parse_args():
    parser = argparse.ArgumentParser(
        description="ONNX → TensorRT Engine 转换工具（FP16 / INT8 量化）"
    )
    parser.add_argument("--onnx",       required=True, help="ONNX 模型路径")
    parser.add_argument("--output",      required=True, help="输出 .engine 路径")
    g = parser.add_mutually_exclusive_group()
    g.add_argument("--fp16",  action="store_true", help="FP16 半精度（Volta+ GPU 推荐）")
    g.add_argument("--int8",  action="store_true", help="INT8 8位量化（需校准数据）")
    parser.add_argument("--workspace",  type=int, default=4096, help="TensorRT workspace MB")
    parser.add_argument("--batch",      type=int, default=1,    help="最大批大小")
    parser.add_argument("--imgsz",      type=int, default=640, help="输入图像分辨率")
    parser.add_argument("--calib-dir",  type=str, default=None, help="INT8 校准图像目录")
    parser.add_argument("--calib-num",  type=int, default=500,  help="INT8 校准图像数量")
    parser.add_argument("--calib-cache",type=str, default="calib_cache.bin", help="校准缓存文件")
    parser.add_argument("--trtexec",    action="store_true",    help="使用 trtexec 命令行工具构建")
    parser.add_argument("--verbose",    action="store_true",    help="打印 TensorRT 详细日志")
    return parser.parse_args()


def main():
    args = parse_args()

    if not Path(args.onnx).exists():
        print(f"[ERROR] ONNX 文件不存在: {args.onnx}")
        sys.exit(1)

    if args.int8 and not args.calib_dir:
        print("[ERROR] INT8 模式需要 --calib-dir 提供校准数据目录")
        sys.exit(1)

    if args.calib_dir and not Path(args.calib_dir).exists():
        print(f"[ERROR] 校准目录不存在: {args.calib_dir}")
        sys.exit(1)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    C, H, W = 3, args.imgsz, args.imgsz
    min_shape = (1,  C, H, W)
    opt_shape = (args.batch, C, H, W)
    max_shape = (args.batch, C, H, W)
    min_str   = f"1x{C}x{H}x{W}"
    opt_str   = f"{args.batch}x{C}x{H}x{W}"
    max_str   = opt_str

    print("=" * 55)
    print("  ONNX → TensorRT Engine 转换")
    print("=" * 55)
    print(f"  ONNX 模型:    {args.onnx}")
    print(f"  输出 Engine: {args.output}")
    mode = "FP16" if args.fp16 else ("INT8" if args.int8 else "FP32")
    print(f"  精度模式:    {mode}")
    print(f"  批大小:       {args.batch}  尺寸: {H}×{W}  workspace: {args.workspace}MB")
    if args.int8:
        print(f"  校准目录:    {args.calib_dir}  数量: {args.calib_num}  缓存: {args.calib_cache}")
    print("=" * 55)

    # INT8 校准器
    calibrator = None
    if args.int8 and args.calib_dir:
        calibrator = YOLOInt8Calibrator(
            image_dir=args.calib_dir,
            input_shape=(C, H, W),
            batch_size=1,
            max_calib_images=args.calib_num,
            cache_file=args.calib_cache,
        )

    # 构建
    success = False
    if args.trtexec:
        success = build_via_trtexec(
            args.onnx, args.output,
            args.fp16, args.int8,
            args.workspace,
            min_str, opt_str, max_str,
        )
    else:
        success = build_engine(
            args.onnx, args.output,
            args.fp16, args.int8, calibrator,
            args.workspace,
            min_shape, opt_shape, max_shape,
            args.verbose,
        )

    if not success:
        sys.exit(1)

    # 验证
    verify_engine(args.output)

    size_mb = Path(args.output).stat().st_size / (1 << 20)
    mult = 8 if args.int8 else (2 if args.fp16 else 1)
    print()
    print("=" * 55)
    print("  模型大小参考")
    print("=" * 55)
    print(f"  当前 ({mode}):   {size_mb:.1f} MB")
    print(f"  FP16 估算:        {size_mb * 2 if args.int8 else size_mb:.1f} MB")
    print(f"  FP32 估算:        {size_mb * mult:.1f} MB")
    print("=" * 55)
    print()
    print("  ✓ 转换完成！")
    print(f"  路径: {args.output}")
    print()
    print("  下一步 → 部署:")
    print(f"  scp {args.output} jetson@<ip>:~/cone_detector/models/")
    print("  roslaunch cone_detector cone_detector.launch")
    print()


if __name__ == "__main__":
    main()
