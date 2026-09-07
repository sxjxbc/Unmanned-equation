#!/usr/bin/env python3
"""
锥桶检测数据集格式转换工具

支持格式:
    VOC XML  → YOLO txt
    COCO JSON → YOLO txt

Usage:
    # VOC XML → YOLO
    python prepare_dataset.py --input-dir dataset/VOC2012 \
        --format voc \
        --output-dir scripts/train/dataset \
        --class-names red_cone blue_cone yellow_cone

    # COCO JSON → YOLO
    python prepare_dataset.py --input-dir dataset/coco \
        --format coco \
        --output-dir scripts/train/dataset \
        --coco-anno annotations/instances_train.json
"""

import argparse
import json
import shutil
import sys
import xml.etree.ElementTree as ET
from pathlib import Path


# 默认锥桶类别映射
DEFAULT_CLASSES = ["red_cone", "blue_cone", "yellow_cone"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="数据集格式转换: VOC XML / COCO JSON → YOLO txt"
    )
    parser.add_argument(
        "--input-dir", type=str, required=True,
        help="原始数据集目录"
    )
    parser.add_argument(
        "--format", type=str, required=True, choices=["voc", "coco"],
        help="原始数据集格式: voc / coco"
    )
    parser.add_argument(
        "--output-dir", type=str, required=True,
        help="输出 YOLO 格式数据集目录"
    )
    parser.add_argument(
        "--class-names", type=str, nargs="+",
        default=DEFAULT_CLASSES,
        help="类别名称列表（顺序对应 YOLO class id）"
    )
    parser.add_argument(
        "--split", type=str, default="auto",
        choices=["auto", "train", "val", "test"],
        help="数据集划分"
    )
    parser.add_argument(
        "--coco-anno", type=str, default=None,
        help="COCO 标注文件路径"
    )
    parser.add_argument(
        "--copy-images", action="store_true", default=True,
        help="复制图像文件到输出目录"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="仅预览，不实际写入"
    )
    return parser.parse_args()


def parse_voc_xml(xml_path: Path, class_names: list):
    """解析 Pascal VOC XML，返回 YOLO 格式标注列表"""
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
    except ET.ParseError as e:
        print(f"  [WARN] XML 解析失败 {xml_path}: {e}")
        return []

    size = root.find("size")
    if size is None:
        return []

    try:
        w = int(size.find("width").text)
        h = int(size.find("height").text)
    except Exception:
        return []

    yolo_lines = []
    for obj in root.iter("object"):
        name_elem = obj.find("name")
        if name_elem is None:
            continue
        name = name_elem.text
        if name not in class_names:
            continue
        cls_id = class_names.index(name)
        bbox = obj.find("bndbox")
        if bbox is None:
            continue
        try:
            xmin = float(bbox.find("xmin").text)
            ymin = float(bbox.find("ymin").text)
            xmax = float(bbox.find("xmax").text)
            ymax = float(bbox.find("ymax").text)
        except Exception:
            continue

        x_c = (xmin + xmax) / 2.0 / w
        y_c = (ymin + ymax) / 2.0 / h
        bw = (xmax - xmin) / w
        bh = (ymax - ymin) / h
        yolo_lines.append((cls_id, x_c, y_c, bw, bh))

    return yolo_lines


def convert_voc(input_dir: Path, output_dir: Path, class_names: list,
                 copy_images: bool, dry_run: bool):
    """VOC 格式 → YOLO txt"""
    output_dir = Path(output_dir)
    img_out = output_dir / "images" / "train"
    lbl_out = output_dir / "labels" / "train"

    if not dry_run:
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)

    img_files = sorted(set(
        list(input_dir.rglob("*.jpg")) +
        list(input_dir.rglob("*.jpeg")) +
        list(input_dir.rglob("*.png"))
    ))

    if not img_files:
        print(f"[ERROR] 未找到图像文件: {input_dir}")
        return

    print(f"找到 {len(img_files)} 张图像")

    total_img, total_obj, skipped = 0, 0, 0
    for img_path in img_files:
        xml_path = img_path.with_suffix(".xml")
        if not xml_path.exists():
            skipped += 1
            continue

        yolo_lines = parse_voc_xml(xml_path, class_names)
        if not yolo_lines:
            skipped += 1
            continue

        total_img += 1
        total_obj += len(yolo_lines)

        if dry_run:
            print(f"  [DRY] {img_path.name} -> {len(yolo_lines)} 目标")
            continue

        lbl_file = lbl_out / f"{img_path.stem}.txt"
        with open(lbl_file, "w", encoding="utf-8") as f:
            for cls_id, x_c, y_c, bw, bh in yolo_lines:
                f.write(f"{cls_id} {x_c:.6f} {y_c:.6f} {bw:.6f} {bh:.6f}\n")

        if copy_images:
            shutil.copy2(img_path, img_out / img_path.name)

        if total_img % 100 == 0:
            print(f"  已处理 {total_img} 张图像...")

    print(f"\n转换完成: {total_img} 张图像, {total_obj} 个目标")
    if skipped:
        print(f"跳过: {skipped} 张（无对应 XML 或无有效标注）")


def convert_coco(input_dir: Path, output_dir: Path, class_names: list,
                  coco_anno: str, copy_images: bool, dry_run: bool):
    """COCO JSON 格式 → YOLO txt"""
    output_dir = Path(output_dir)
    img_out = output_dir / "images" / "train"
    lbl_out = output_dir / "labels" / "train"

    if not dry_run:
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)

    anno_file = Path(coco_anno)
    if not anno_file.is_absolute():
        anno_file = input_dir / coco_anno

    if not anno_file.exists():
        print(f"[ERROR] COCO 标注文件不存在: {anno_file}")
        return

    with open(anno_file, "r", encoding="utf-8") as f:
        coco_data = json.load(f)

    img_map = {img["id"]: img for img in coco_data["images"]}
    ann_by_img = {}
    for ann in coco_data["annotations"]:
        ann_by_img.setdefault(ann["image_id"], []).append(ann)

    cat_map = {}
    for cat in coco_data.get("categories", []):
        if cat["name"].strip() in class_names:
            cat_map[cat["id"]] = class_names.index(cat["name"].strip())

    if not cat_map:
        print("[WARN] 未找到匹配类别，请检查 --class-names 参数")

    total_img, total_obj = 0, 0
    for img_id, anns in ann_by_img.items():
        if img_id not in img_map:
            continue
        img_info = img_map[img_id]
        w, h = img_info["width"], img_info["height"]
        img_file = img_info["file_name"]

        yolo_lines = []
        for ann in anns:
            if ann["category_id"] not in cat_map:
                continue
            cls_id = cat_map[ann["category_id"]]
            x_min, y_min, bw, bh = ann["bbox"]
            x_c = (x_min + bw / 2) / w
            y_c = (y_min + bh / 2) / h
            yolo_lines.append((cls_id, x_c, y_c, bw / w, bh / h))
            total_obj += 1

        if not yolo_lines:
            continue

        total_img += 1
        if dry_run:
            print(f"  [DRY] {img_file} -> {len(yolo_lines)} 目标")
            continue

        stem = Path(img_file).stem
        lbl_file = lbl_out / f"{stem}.txt"
        with open(lbl_file, "w", encoding="utf-8") as f:
            for cls_id, x_c, y_c, bw, bh in yolo_lines:
                f.write(f"{cls_id} {x_c:.6f} {y_c:.6f} {bw:.6f} {bh:.6f}\n")

        if copy_images:
            src_img = input_dir / img_file
            if src_img.exists():
                shutil.copy2(src_img, img_out / Path(img_file).name)

        if total_img % 100 == 0:
            print(f"  已处理 {total_img} 张图像...")

    print(f"\n转换完成: {total_img} 张图像, {total_obj} 个目标")


def main():
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    if not input_dir.exists():
        print(f"[ERROR] 输入目录不存在: {input_dir}")
        sys.exit(1)

    print("=" * 55)
    print("锥桶检测数据集格式转换工具")
    print("=" * 55)
    print(f"输入格式: {args.format.upper()}")
    print(f"输入目录: {input_dir}")
    print(f"输出目录: {output_dir}")
    print(f"类别映射: {args.class_names}")
    print(f"模式: {'[DRY RUN] 预览' if args.dry_run else '正式转换'}")
    print("=" * 55)

    if args.format == "voc":
        convert_voc(input_dir, output_dir, args.class_names,
                    args.copy_images, args.dry_run)
    elif args.format == "coco":
        convert_coco(input_dir, output_dir, args.class_names,
                     args.coco_anno, args.copy_images, args.dry_run)

    if not args.dry_run:
        data_yaml = output_dir / "data.yaml"
        content = f"# YOLO 格式数据集配置 - 锥桶检测\n# 由 prepare_dataset.py 自动生成\n\npath: {output_dir.resolve()}\ntrain: images/train\nval: images/val\n\nnc: {len(args.class_names)}\nnames:\n"
        for i, name in enumerate(args.class_names):
            content += f"    {i}: {name}\n"
        data_yaml.write_text(content, encoding="utf-8")
        print(f"\n[INFO] 已生成数据集配置: {data_yaml}")
        print("[INFO] 可将此路径填入 train_cone.py 的 --data-yaml 参数")


if __name__ == "__main__":
    main()
