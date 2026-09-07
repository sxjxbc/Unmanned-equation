#!/usr/bin/env python3
"""
YOLO 锥桶检测训练主脚本

锥桶分类:
    0 - red_cone    → TURN_LEFT（左转）
    1 - blue_cone   → TURN_RIGHT（右转）
    2 - yellow_cone → STOP（停止）

Usage:
    python train_cone.py [--model MODEL] [--data-yaml DATA_YAML]
                         [--epochs EPOCHS] [--imgsz IMGSZ] [--batch BATCH]
                         [--device DEVICE] [--half]

    # 示例（工控机 GPU 训练）
    cd C:/Users/subin/Desktop/锥桶检测
    python cone_detector/scripts/train/train_cone.py --data-yaml cone_detector/config/cone_data.yaml --epochs 100 --batch 16 --device 0
i无效率、9你、06*yi·09-
下、ei9·zuibewo+r9*，p0e-65vi,-90二五☃qviux
    # 可选模型（Ultralytics 自动下载预训练权重）：
    #   yolo11n.pt  ← 最快，推荐嵌入式平台
    #   yolo11s.pt  ← 精度略高
    #   yolov8n.pt  ← 备选
"""

import argparse
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

try:
    import yaml
except ImportError:
    print("错误: 未安装 pyyaml，请运行: pip install pyyaml")
    sys.exit(1)

# 添加 scripts 父目录以支持导入
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from ultralytics import YOLO
except ImportError:
    print("错误: 未安装 ultralytics，请运行: pip install ultralytics")
    sys.exit(1)

def parse_opt():
    parser = argparse.ArgumentParser(description="YOLO 锥桶三色检测训练脚本")
    parser.add_argument(
        "--model", type=str, default="yolo11n.pt",
        help="模型名称或路径，如 yolo11n.pt / yolov8n.pt（默认: yolo11n.pt，Ultralytics 自动下载）"
    )
    parser.add_argument(
        "--data-yaml", type=str, default="config/cone_data.yaml",
        help="数据集 YAML 配置文件路径"
    )
    parser.add_argument(
        "--epochs", type=int, default=100,
        help="训练总轮数"
    )
    parser.add_argument(
        "--imgsz", type=int, default=640,
        help="输入图像分辨率（推荐 416~640，实时场景可降至 416 提速）"
    )
    parser.add_argument(
        "--batch", type=int, default=16,
        help="批次大小（Jetson AGX Xavier 建议 8~16，根据显存调整）"
    )
    parser.add_argument(
        "--output-dir", type=str, default="runs/train",
        help="训练结果输出根目录"
    )
    parser.add_argument(
        "--device", type=str, default="auto",
        help="训练设备: 'auto' / 'cpu' / GPU id（如 '0'）"
    )
    parser.add_argument(
        "--workers", type=int, default=None,
        help="数据加载进程数（默认: CPU 核数 - 1）"
    )
    parser.add_argument(
        "--half", action="store_true",
        help="启用混合精度训练（FP16，节省显存/加速）"
    )
    parser.add_argument(
        "--lr0", type=float, default=0.001,
        help="初始学习率（默认 0.001，适用于 AdamW）"
    )
    parser.add_argument(
        "--save-period", type=int, default=10,
        help="每 N 轮保存一次权重（默认 10）"
    )
    parser.add_argument(
        "--patience", type=int, default=50,
        help="早停 patience（多少轮无提升则停止，默认 50）"
    )
    parser.add_argument(
        "--project", type=str, default=None,
        help="项目根目录（覆盖 output-dir）"
    )
    parser.add_argument(
        "--name", type=str, default=None,
        help="实验名称（子目录名）"
    )
    parser.add_argument(
        "--pretrained", action="store_true", default=True,
        help="使用预训练权重迁移学习（默认开启）"
    )
    return parser.parse_args()


def auto_device(device: str) -> str:
    """自动选择最佳计算设备"""
    if device.lower() != "auto":
        return device
    try:
        import torch
        if torch.cuda.is_available():
            return "0"
    except Exception:
        pass
    return "cpu"


def auto_workers(workers_arg: int = None) -> int:
    """自动计算合适的数据加载线程数"""
    try:
        cpu_count = os.cpu_count() or 1
        if workers_arg and workers_arg > 0:
            return workers_arg
        return max(1, cpu_count - 1)
    except Exception:
        return 4


def resolve_data_yaml(data_yaml_path: str, script_dir: Path) -> str:
    """
    解析数据集 YAML，确保 path/train/val 全为绝对路径，绕过 Ultralytics
    在 Windows 上的 datasets_dir 拼接问题 + Path 反斜杠歧义问题。

    策略：
    1. YAML 不存在 → 抛出错误
    2. 读取 YAML，提取 path/train/val/names 字段
    3. path/train/val 全转为绝对路径（基于 os.path，Windows 最可靠）
    4. 写入 resolved_data.yaml 到 script_dir，返回绝对路径
    """
    import os
    import yaml

    # ── 辅助：统一路径格式，避免 Windows 反斜杠 \t 被解析为 Tab ──────────
    def normalize(v):
        if not v:
            return None
        # 统一正斜杠，再转绝对路径（os.path 对两种分隔符都兼容）
        return os.path.abspath(str(v).replace("\\", "/"))

    # ── 定位 YAML 文件 ───────────────────────────────────────────────────
    yaml_path = script_dir / data_yaml_path
    if not yaml_path.exists():
        yaml_path = Path(data_yaml_path).resolve()
        if not yaml_path.exists():
            raise FileNotFoundError(
                f"数据集 YAML 不存在: {yaml_path}\n"
                f"请确认 --data-yaml 指向 config/cone_data.yaml"
            )

    with open(yaml_path, encoding="utf-8") as f:
        data = yaml.safe_load(f)

    # ── 解析三个路径字段，全转为绝对路径 ─────────────────────────────────
    raw_path   = str(data.get("path",   "")).strip()
    raw_train  = str(data.get("train",  "images/train")).strip()
    raw_val    = str(data.get("val",    "images/val")).strip()

    # path 为空时默认 dataset/
    dataset_root = normalize(raw_path) if raw_path else os.path.join(str(script_dir), "dataset")

    # train/val：如果是绝对路径直接用，否则拼在 dataset_root 下
    if os.path.isabs(raw_train):
        train_abs = normalize(raw_train)
    else:
        train_abs = os.path.join(dataset_root, raw_train.replace("\\", "/").lstrip("/"))

    if os.path.isabs(raw_val):
        val_abs = normalize(raw_val)
    else:
        val_abs = os.path.join(dataset_root, raw_val.replace("\\", "/").lstrip("/"))

    # 验证目录存在
    for label, abs_path in [("数据集根目录", dataset_root),
                            ("训练集",       train_abs),
                            ("验证集",       val_abs)]:
        if not os.path.exists(abs_path):
            raise FileNotFoundError(
                f"{label} 不存在: {abs_path}\n"
                f"请确认 config/cone_data.yaml 中路径正确"
            )

    # ── 构建 resolved YAML ───────────────────────────────────────────────
    resolved = {
        "path":  dataset_root,
        "train": train_abs,
        "val":   val_abs,
        "nc":    data.get("nc", 3),
        "names": data.get("names", ["red_cone", "blue_cone", "yellow_cone"]),
    }

    resolved_yaml = script_dir / "resolved_data.yaml"
    with open(resolved_yaml, "w", encoding="utf-8") as f:
        yaml.safe_dump(resolved, f, allow_unicode=True, default_flow_style=False, sort_keys=False)

    logging.info(f"数据集路径: {dataset_root}")
    logging.info(f"训练集路径: {train_abs}")
    logging.info(f"验证集路径: {val_abs}")
    logging.info(f"resolved YAML: {resolved_yaml}")
    return str(resolved_yaml)


def create_data_yaml_if_needed(data_yaml_path: str):
    """数据集 YAML 不存在时创建模板文件（已弃用，改为 resolve_data_yaml）"""
    pass


def main():
    opt = parse_opt()

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s: %(message)s"
    )

    # 自动命名实验目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = opt.name or f"cone_yolo_{timestamp}"
    output_dir = opt.project or opt.output_dir

    # 设备选择
    device = auto_device(opt.device)
    workers = auto_workers(opt.workers)

    # 脚本所在目录，用于解析 YAML 路径（不受 cwd 影响）
    script_dir = Path(__file__).parent.parent.resolve()

    logging.info("=" * 50)
    logging.info("YOLO 锥桶三色检测训练")
    logging.info("=" * 50)
    logging.info(f"模型:            {opt.model}")
    logging.info(f"数据集 YAML:    {opt.data_yaml}")
    logging.info(f"训练轮数:        {opt.epochs}")
    logging.info(f"图像尺寸:        {opt.imgsz}")
    logging.info(f"批次大小:        {opt.batch}")
    logging.info(f"计算设备:        {device}")
    logging.info(f"数据加载线程:    {workers}")
    logging.info(f"混合精度 FP16:   {opt.half}")
    logging.info(f"输出目录:        {output_dir}/{exp_name}")
    logging.info("=" * 50)

    # 解析数据集 YAML → 强制使用绝对路径，避免 Ultralytics 拼接 datasets/
    resolved_yaml = resolve_data_yaml(opt.data_yaml, script_dir)

    # 加载模型
    model = YOLO(opt.model)

    # 训练（使用 resolved YAML）
    results = model.train(
        data=resolved_yaml,
        epochs=opt.epochs,
        imgsz=opt.imgsz,
        batch=opt.batch,
        device=device,
        project=output_dir,
        name=exp_name,
        patience=opt.patience,
        workers=workers,
        optimizer="AdamW",
        lr0=opt.lr0,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3.0,
        mosaic=1.0,
        mixup=0.1,
        copy_paste=0.0,
        save=True,
        save_period=opt.save_period,
        half=opt.half,
        pretrained=opt.pretrained,
        verbose=True,
    )

    # 输出训练结果
    logging.info("")
    logging.info("=" * 50)
    logging.info("训练完成！")
    try:
        save_dir = Path(results.save_dir)
        best_pt = save_dir / "weights" / "best.pt"
        last_pt = save_dir / "weights" / "last.pt"
        logging.info(f"最佳权重:   {best_pt}")
        logging.info(f"最终权重:   {last_pt}")
        logging.info(f"验证结果:   {save_dir}")
    except Exception:
        logging.info("请手动查看 runs/ 目录获取权重文件路径")

    logging.info("")
    logging.info("下一步建议:")
    logging.info(f"  1. 导出 TensorRT 引擎:")
    logging.info(f"     python scripts/export/export_to_onnx.py --model runs/train/{exp_name}/weights/best.pt")
    logging.info(f"  2. 或直接导出 engine:")
    logging.info(f"     yolo export model=runs/train/{exp_name}/weights/best.pt format=engine imgsz={opt.imgsz} half={opt.half}")
    logging.info("=" * 50)


if __name__ == "__main__":
    main()
