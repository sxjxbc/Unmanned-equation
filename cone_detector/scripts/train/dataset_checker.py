#!/usr/bin/env python3
"""
锥桶检测数据集格式校验工具

检查项：
  1. data.yaml 语法正确性
  2. train/val 图像目录是否存在
  3. 每张图像是否有对应标签文件
  4. 标签格式是否符合 YOLO 标准 (class x_center y_center width height, 归一化)
  5. 类别 ID 是否在 [0, nc-1] 范围内
  6. 坐标值是否在 [0, 1] 范围内

Usage:
    python dataset_checker.py --data-yaml config/cone_data.yaml
    python dataset_checker.py --data-yaml config/cone_data.yaml --verbose
"""

import argparse
import sys
from pathlib import Path


# YOLO 锥桶 3 类定义（用于校验类别名称合理性）
EXPECTED_CLASSES = {
    0: ["red_cone", "红色锥桶"],
    1: ["blue_cone", "蓝色锥桶"],
    2: ["yellow_cone", "黄色锥桶"],
}


def parse_args():
    parser = argparse.ArgumentParser(description="数据集格式校验")
    parser.add_argument(
        "--data-yaml", type=str, required=True,
        help="数据集配置文件路径"
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="输出详细校验信息"
    )
    return parser.parse_args()


def load_yaml(yaml_path: Path):
    """手动解析简单 YAML（避免依赖）"""
    import yaml as _yaml
    with open(yaml_path, "r", encoding="utf-8") as f:
        return _yaml.safe_load(f)


def check_yaml(data_yaml: Path):
    """检查 YAML 配置"""
    errors = []
    warnings = []

    try:
        data = load_yaml(data_yaml)
    except FileNotFoundError:
        errors.append(f"YAML 文件不存在: {data_yaml}")
        return errors, warnings
    except Exception as e:
        errors.append(f"YAML 解析失败: {e}")
        return errors, warnings

    # 检查必须字段
    required = ["path", "train", "val", "nc", "names"]
    for field in required:
        if field not in data:
            errors.append(f"YAML 缺少必需字段: {field}")

    if "nc" in data:
        if data["nc"] != 3:
            errors.append(f"nc 应为 3（3 类锥桶），当前为 {data['nc']}")
        else:
            warnings.append(f"nc = 3 ✓")

    if "names" in data:
        names = data["names"]
        if isinstance(names, dict):
            for cid, name in names.items():
                if int(cid) not in EXPECTED_CLASSES:
                    warnings.append(f"未知类别 ID {cid}，期望 0~2")
                if "cone" not in str(name).lower():
                    warnings.append(f"类别名称 '{name}' 不包含 'cone'，请确认是否正确")
        warnings.append(f"类别映射: {names} ✓")

    # 检查数据集路径
    if "path" in data:
        dataset_root = Path(data["path"])
        if not dataset_root.is_absolute():
            # 相对路径相对于 YAML 所在目录
            dataset_root = (data_yaml.parent / dataset_root).resolve()
        data["_resolved_root"] = dataset_root

        for split in ["train", "val"]:
            if split in data:
                img_dir = dataset_root / data[split]
                if not img_dir.exists():
                    errors.append(f"[{split}] 图像目录不存在: {img_dir}")
                else:
                    warnings.append(f"[{split}] 图像目录存在: {img_dir} ✓")

    return errors, warnings


def check_labels(dataset_root: Path, split: str, nc: int, verbose: bool):
    """检查标签文件"""
    errors = []
    warnings = []
    img_dir = None

    for subdir in ["images", "images/train", "images/val"]:
        candidate = dataset_root / subdir.replace("images/", "").replace("/train", "/images/train").replace("/val", "/images/val")
        # 简化：直接用 images/train, images/val
        img_candidate = dataset_root / "images" / split
        label_candidate = dataset_root / "labels" / split

        if img_candidate.exists():
            img_dir = img_candidate
            label_dir = label_candidate
            break

    if img_dir is None:
        img_dir = dataset_root / "images" / split
        label_dir = dataset_root / "labels" / split

    if not img_dir.exists():
        errors.append(f"[{split}] 图像目录不存在: {img_dir}")
        return errors, warnings, 0, 0

    # 遍历图像
    img_files = sorted([
        f for f in img_dir.iterdir()
        if f.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]
    ])

    total = len(img_files)
    checked = 0
    label_errors = 0
    coordinate_errors = 0

    for img_file in img_files:
        checked += 1
        label_file = label_dir / f"{img_file.stem}.txt"

        if not label_file.exists():
            errors.append(f"缺失标签: {label_file.name}（对应图像: {img_file.name}）")
            label_errors += 1
            continue

        try:
            lines = label_file.read_text(encoding="utf-8").strip().split("\n")
        except Exception as e:
            errors.append(f"无法读取标签: {label_file}，错误: {e}")
            label_errors += 1
            continue

        for line_no, line in enumerate(lines, 1):
            line = line.strip()
            if not line:
                continue

            parts = line.split()
            if len(parts) != 5:
                errors.append(
                    f"{label_file.name}:{line_no} - 格式错误，期望 5 列 "
                    f"(class x_center y_center width height)，实际: {parts}"
                )
                coordinate_errors += 1
                continue

            try:
                cls = int(parts[0])
                x_c, y_c, w, h = map(float, parts[1:])
            except ValueError:
                errors.append(
                    f"{label_file.name}:{line_no} - 数值解析失败: {line}"
                )
                coordinate_errors += 1
                continue

            # 类别范围检查
            if cls < 0 or cls >= nc:
                errors.append(
                    f"{label_file.name}:{line_no} - 类别 ID {cls} 超出范围 [0, {nc-1}]"
                )
                coordinate_errors += 1

            # 坐标范围检查（YOLO 格式要求归一化坐标）
            for val, name in [(x_c, "x_center"), (y_c, "y_center"),
                               (w, "width"), (h, "height")]:
                if val < 0 or val > 1:
                    errors.append(
                        f"{label_file.name}:{line_no} - {name}={val} 超出 [0,1] 范围"
                    )
                    coordinate_errors += 1

        if verbose and checked % 50 == 0:
            print(f"  已校验 {checked}/{total} 张图像...")

    if total > 0:
        warnings.append(
            f"[{split}] {total} 张图像，{label_errors} 个标签缺失，"
            f"{coordinate_errors} 个格式/坐标错误"
        )
    else:
        warnings.append(f"[{split}] 未找到图像文件，请检查 images/{split}/ 目录")

    return errors, warnings, total, label_errors + coordinate_errors


def main():
    args = parse_args()
    data_yaml = Path(args.data_yaml)
    verbose = args.verbose

    print("=" * 55)
    print("锥桶检测数据集格式校验")
    print("=" * 55)
    print(f"配置文件: {data_yaml}")
    print()

    all_errors = []
    all_warnings = []

    # 1. 检查 YAML
    errors, warnings = check_yaml(data_yaml)
    all_errors.extend(errors)
    all_warnings.extend(warnings)

    if not data_yaml.exists():
        print(f"[FATAL] YAML 文件不存在: {data_yaml}")
        sys.exit(1)

    data = None
    try:
        data = load_yaml(data_yaml)
    except Exception as e:
        print(f"[FATAL] YAML 解析失败: {e}")
        sys.exit(1)

    # 解析数据集根路径
    if "path" in data:
        dataset_root = Path(data["path"])
        if not dataset_root.is_absolute():
            dataset_root = (data_yaml.parent / dataset_root).resolve()
    else:
        dataset_root = data_yaml.parent

    print(f"数据集根目录: {dataset_root}")
    nc = data.get("nc", 0)

    # 2. 检查 train 集
    print("\n--- 检查 train 集 ---")
    e, w, t_total, t_err = check_labels(dataset_root, "train", nc, verbose)
    all_errors.extend(e)
    all_warnings.extend(w)

    # 3. 检查 val 集
    print("\n--- 检查 val 集 ---")
    e, w, v_total, v_err = check_labels(dataset_root, "val", nc, verbose)
    all_errors.extend(e)
    all_warnings.extend(w)

    # 输出报告
    print("\n" + "=" * 55)
    print("校验结果汇总")
    print("=" * 55)

    if all_warnings:
        print("\n[INFO]")
        for w_item in all_warnings:
            print(f"  ✓ {w_item}")

    if all_errors:
        print(f"\n[ERROR] 共发现 {len(all_errors)} 个错误:")
        for i, e_item in enumerate(all_errors, 1):
            print(f"  {i}. {e_item}")
        print("\n请修复以上错误后重新训练。")
        sys.exit(1)
    else:
        print("\n[SUCCESS] 数据集格式校验通过！")
        print(f"  总计: train={t_total} 张, val={v_total} 张, 错误=0")
        sys.exit(0)


if __name__ == "__main__":
    main()
