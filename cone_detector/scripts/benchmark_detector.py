#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
benchmark_detector.py
=======================
对锥桶检测节点进行离线推理速度基准测试（不依赖 ROS）

用法:
  python3 benchmark_detector.py \
      --model  yolov6n_cone.engine \
      --backend tensorrt \
      --imgsz  640 \
      --runs   200

输出: 平均推理时延、FPS、P50/P95/P99 延迟
"""

import argparse
import time
import numpy as np


def run_benchmark(model_path: str, backend: str,
                  imgsz: int, runs: int, warmup: int):
    dummy = np.random.rand(1, 3, imgsz, imgsz).astype(np.float32)

    if backend == "tensorrt":
        import tensorrt as trt
        import pycuda.driver as cuda
        import pycuda.autoinit  # noqa

        logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(logger)
        with open(model_path, "rb") as f:
            engine = runtime.deserialize_cuda_engine(f.read())
        context = engine.create_execution_context()

        size = trt.volume(engine.get_binding_shape(0))
        h_in  = cuda.pagelocked_empty(size, np.float32)
        d_in  = cuda.mem_alloc(h_in.nbytes)
        out_size = trt.volume(engine.get_binding_shape(1))
        h_out = cuda.pagelocked_empty(out_size, np.float32)
        d_out = cuda.mem_alloc(h_out.nbytes)
        stream = cuda.Stream()
        bindings = [int(d_in), int(d_out)]

        def infer():
            np.copyto(h_in, dummy.ravel())
            cuda.memcpy_htod_async(d_in, h_in, stream)
            context.execute_async_v2(bindings=bindings,
                                     stream_handle=stream.handle)
            cuda.memcpy_dtoh_async(h_out, d_out, stream)
            stream.synchronize()

    else:  # onnx
        import onnxruntime as ort
        sess = ort.InferenceSession(model_path,
                                    providers=["CUDAExecutionProvider",
                                               "CPUExecutionProvider"])
        in_name = sess.get_inputs()[0].name

        def infer():
            sess.run(None, {in_name: dummy})

    # 预热
    print(f"[bench] 预热 {warmup} 次...")
    for _ in range(warmup):
        infer()

    # 计时
    print(f"[bench] 正式测试 {runs} 次...")
    latencies = []
    for _ in range(runs):
        t0 = time.perf_counter()
        infer()
        latencies.append((time.perf_counter() - t0) * 1000)

    arr = np.array(latencies)
    print("\n===== 推理性能报告 =====")
    print(f"  后端        : {backend}")
    print(f"  模型        : {model_path}")
    print(f"  输入尺寸    : {imgsz}x{imgsz}")
    print(f"  测试轮次    : {runs}")
    print(f"  平均延迟    : {arr.mean():.2f} ms  ({1000/arr.mean():.1f} FPS)")
    print(f"  最小延迟    : {arr.min():.2f} ms")
    print(f"  最大延迟    : {arr.max():.2f} ms")
    print(f"  P50         : {np.percentile(arr, 50):.2f} ms")
    print(f"  P95         : {np.percentile(arr, 95):.2f} ms")
    print(f"  P99         : {np.percentile(arr, 99):.2f} ms")
    print("========================\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model",   required=True)
    ap.add_argument("--backend", default="tensorrt",
                    choices=["tensorrt", "onnx"])
    ap.add_argument("--imgsz",  type=int, default=640)
    ap.add_argument("--runs",   type=int, default=200)
    ap.add_argument("--warmup", type=int, default=20)
    args = ap.parse_args()
    run_benchmark(args.model, args.backend,
                  args.imgsz, args.runs, args.warmup)
