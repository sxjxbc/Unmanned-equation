#!/home/nvidia/miniconda3/envs/yolov5/bin/python

import rospy
import cv2
import numpy as np
import torch
import os
import sys
import subprocess
import signal
import time
from datetime import datetime
import threading
import traceback

# ---------------------- 新增 ROS 相关导入 ----------------------
import rospy
from msgs.msg import ConeDetection
from std_msgs.msg import Header

# ---------------------- 新增：导入CAN UDP相关模块 ----------------------
import socket
import struct

# ------------------------------------------------------------

sys.path.append('/home/nvidia/yolov5')  # 替换为你的YOLO路径

# ---------------------- 配置参数 ----------------------
weight_path = "/home/nvidia/yolov5/weights1/best.pt"  # YOLO权重路径

# 摄像头内参（标定时的参数）
FX = 435.01  # 焦距x（像素）
FY = 432.67  # 焦距y（像素）
CX = 627.13  # 主点x坐标（像素，图像中心x）
CY = 362.05  # 主点y坐标（像素，图像中心y）

# 分辨率
CALIBRATED_RESOLUTION = (1280, 720)  # 标定时的分辨率

# 物理参数
BASELINE = 18.7  # 双目摄像头光心间距（cm，实测值）
CAMERA_HEIGHT = 104.0  # 摄像头光心距地面高度（cm）
TARGET_HEIGHT = 19.0  # 锥桶标签中心距地面高度（cm）
HEIGHT_DIFF = CAMERA_HEIGHT - TARGET_HEIGHT  # 高度差

# 测距修正系数
BASE_ANGLE_CORRECTION = 0.78  # 基础视角修正系数（建议范围：0.98-1.08）
HEIGHT_CORRECTION_STRENGTH = 0.08  # 高度相关修正强度（建议范围：0.05-0.25）
DISTANCE_CORRECTION = 0.001  # 距离相关修正（建议范围：0.01-0.05）

# 设备配置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 全局变量（存储摄像头帧）
left_frame = None
right_frame = None
frame_lock = threading.Lock()
running = True

# 新增：用于记录前一帧是否识别到yellow标签
prev_has_yellow = False
prev_has_red_blue = False  # 新增：记录前一帧是否有红色或蓝色

# 新增：ROS发布器全局变量
cone_pub = None

# 新增：CAN UDP客户端全局变量
can_client = None
can_client_ok = False

# 新增：显示窗口相关变量
display_enabled = True  # 是否启用显示窗口
window_name_left = "Left Camera - Color & Distance"
window_name_right = "Right Camera - Color & Distance"
window_name_combined = "Stereo Vision - Combined View"


# ---------------------- CAN UDP客户端类 ----------------------
class CANUDPClient:
    """CAN UDP客户端，用于发送CAN错误帧"""

    def __init__(self):
        self.sock = None
        self.dest_ip = "192.168.0.7"
        self.dest_port = 20001  # CAN1端口20001
        self.self_port = 11311  # 使用相同的端口11311
        self.addr_to = None

    def init(self, dest_ip="192.168.0.7", dest_port=20001, self_port=11311):
        """初始化UDP客户端"""
        try:
            self.dest_ip = dest_ip
            self.dest_port = dest_port
            self.self_port = self_port

            print(f"🔧 初始化UDP: 目标={dest_ip}:{dest_port}, 本地端口={self_port}")

            # 创建UDP socket
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

            # 设置socket选项
            self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

            # 绑定本地端口
            self.sock.bind(('', self_port))

            # 设置目标地址
            self.addr_to = (dest_ip, dest_port)

            print("✅ CAN UDP客户端初始化成功")
            return True

        except Exception as e:
            print(f"❌ CAN UDP客户端初始化失败: {e}")
            return False

    def send_can_frame(self, can_id, data):
        """发送CAN帧 - 使用USR-CANET200的13字节透传协议"""
        if self.sock is None:
            print("❌ UDP socket未初始化")
            return False

        if len(data) != 8:
            print(f"❌ CAN数据必须为8字节，当前为{len(data)}字节")
            return False

        try:
            # 根据USR-CANET200协议构建13字节数据包
            packet = bytearray(13)

            # 帧信息字节 (第1字节)
            # Bit7: FF - 0=标准帧, 1=扩展帧
            # Bit6: RTR - 0=数据帧, 1=远程帧
            # Bit5-4: 保留位 (00)
            # Bit3-0: 数据长度 (0-8)
            frame_info = 0x00  # 标准帧 + 数据帧 + 保留位00

            # 设置数据长度
            data_length = len(data)
            frame_info |= (data_length & 0x0F)

            # 设置帧类型 (标准帧/扩展帧)
            if can_id > 0x7FF:  # 扩展帧
                frame_info |= 0x80
            else:  # 标准帧
                frame_info |= 0x00

            packet[0] = frame_info

            # CAN ID (第2-5字节) - 大端序，高位在前
            if can_id > 0x7FF:  # 扩展帧 (29位)
                packet[1] = (can_id >> 24) & 0xFF
                packet[2] = (can_id >> 16) & 0xFF
                packet[3] = (can_id >> 8) & 0xFF
                packet[4] = can_id & 0xFF
            else:  # 标准帧 (11位)
                packet[1] = 0x00
                packet[2] = 0x00
                packet[3] = (can_id >> 8) & 0xFF
                packet[4] = can_id & 0xFF

            # CAN数据 (第6-13字节)
            for i in range(8):
                if i < len(data):
                    packet[5 + i] = data[i]
                else:
                    packet[5 + i] = 0x00  # 不足8字节补0

            # 发送数据包
            sent = self.sock.sendto(bytes(packet), self.addr_to)

            if sent == len(packet):
                # 打印实际发送的数据内容
                data_hex = ' '.join(['{:02X}'.format(b) for b in data])
                print("✅ 发送CAN帧: ID=0x{:08X}, 数据=[{}]".format(can_id, data_hex))
                return True
            else:
                print("❌ 发送CAN帧失败")
                return False

        except Exception as e:
            print("❌ 发送CAN帧时出错: {}".format(e))
            return False

    def close(self):
        """关闭socket"""
        if self.sock:
            self.sock.close()
            self.sock = None


# ---------------------- 模型加载与检测 ----------------------
def load_model(weights):
    """加载YOLO模型"""
    from models.experimental import attempt_load
    try:
        model = attempt_load(weights, device=device)
        print(f"模型加载成功（设备：{device}）")
        return model
    except Exception as e:
        print(f"模型加载失败：{e}")
        raise



def validate_color(self, im0, xyxy, expected_class, names):
    x1, y1, x2, y2 = map(int, xyxy)
    roi = im0[y1:y2, x1:x2]
    if roi.size == 0:
        return expected_class

    # 转换为HSV
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    # 颜色范围
    # 黄色范围（最激进，优先识别为黄色）
    lower_yellow = np.array([18, 50, 60])       # 从偏黄的橙色开始
    upper_yellow = np.array([35, 255, 255])     # 到偏黄的绿色结束

# 红色范围（严格收紧，避免与黄色重叠）
    lower_red1 = np.array([0, 150, 80])        # 提高饱和度要求，排除偏黄的红色
    upper_red1 = np.array([5, 255, 255])       # 收紧上限，避免与黄色重叠
    lower_red2 = np.array([175, 150, 80])      # 提高饱和度要求
    upper_red2 = np.array([180, 255, 255])

    # 创建掩码
    mask_red = cv2.bitwise_or(
        cv2.inRange(hsv, lower_red1, upper_red1),
        cv2.inRange(hsv, lower_red2, upper_red2)
    )
    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)

    # 计算比例
    total_pixels = mask_red.size
    red_ratio = np.sum(mask_red > 0) / total_pixels
    yellow_ratio = np.sum(mask_yellow > 0) / total_pixels

    print(f"颜色分析 - 红:{red_ratio:.3f} 黄:{yellow_ratio:.3f} 预测:{expected_class}({names[expected_class]})")

    # 更保守的修正逻辑
    if expected_class == 0:  # 模型说是红色
        # 只有当黄色明显占优时才修正
        if yellow_ratio > 0.2 and yellow_ratio > red_ratio * 2.0:
            print(f"保守修正: {names[expected_class]} -> yellow (黄色明显主导)")
            return 2
    
    # 如果是黄色预测，但红色很多，修正为红色（防止红色被识别为黄色）
    elif expected_class == 2:  # 模型说是黄色
        if red_ratio > 0.2 and red_ratio > yellow_ratio * 2.0:
            print(f"反向修正: {names[expected_class]} -> red (红色明显主导)")
            return 0

    return expected_class

def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
    """将检测框坐标从处理后图像映射到原始图像"""
    if ratio_pad is None:
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, [0, 2]] -= pad[0]
    coords[:, [1, 3]] -= pad[1]
    coords[:, :4] /= gain
    clip_coords(coords, img0_shape)
    return coords


def clip_coords(boxes, shape):
    """裁剪边界框到图像范围内"""
    if isinstance(boxes, torch.Tensor):
        boxes[:, 0].clamp_(0, shape[1])
        boxes[:, 1].clamp_(0, shape[0])
        boxes[:, 2].clamp_(0, shape[1])
        boxes[:, 3].clamp_(0, shape[0])
    else:
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, shape[1])
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, shape[0])


def detect_objects(model, img):
    """检测图像中的锥桶并返回中心点坐标"""
    from utils.general import non_max_suppression
    from utils.augmentations import letterbox

    img0 = img.copy()  # 原始图像
    h, w = img0.shape[:2]

    # 预处理：调整尺寸
    img = letterbox(img0, new_shape=640, auto=False)[0]
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR转RGB，HWC转CHW
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).to(device).float() / 255.0  # 归一化
    if img.ndimension() == 3:
        img = img.unsqueeze(0)  # 添加batch维度

    # 推理
    with torch.no_grad():
        pred = model(img, augment=False)[0]

    # 后处理：非极大值抑制
    pred = non_max_suppression(pred, 0.5, 0.4)

    detections = []
    for det in pred:
        if len(det):
            # 将检测框映射回原始图像
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()

            for *xyxy, conf, cls in det:
                # 获取类别名称
                cls_name = model.names[int(cls)] if hasattr(model, 'names') else f'class_{int(cls)}'
                if cls_name not in ['red', 'blue', 'yellow']:
                    continue  # 只保留锥桶类别

                # 计算检测框坐标
                x1, y1, x2, y2 = map(float, xyxy)
                # 调整中心点（红色锥桶底部较宽，中心下移）
                center_x = (x1 + x2) / 2  # 水平中心（x坐标）
                if cls_name == 'red':
                    center_y = y1 + (y2 - y1) * 0.65  # 垂直中心（y坐标，下移至65%高度）
                else:
                    center_y = (y1 + y2) / 2  # 其他锥桶用中心

                detections.append({
                    'label': cls_name,
                    'center': (center_x, center_y),  # 中心点像素坐标
                    'bbox': (x1, y1, x2, y2),  # 边界框
                    'confidence': float(conf)
                })

    return detections


# ---------------------- 坐标计算（核心逻辑） ----------------------
def calculate_coordinates(left_center, right_center, label):
    """
    计算锥桶标签的二维坐标（y, x）
    y：垂直距离（深度，cm）
    x：水平偏移量（cm，正值为左，负值为右）
    """
    # 提取左右图中心点坐标
    left_x, left_y = left_center
    right_x, right_y = right_center

    # 计算视差
    disparity = left_x - right_x
    print(f"\n{label}锥桶坐标计算:")
    print(f"  左图中心: ({left_x:.1f}, {left_y:.1f})px")
    print(f"  右图中心: ({right_x:.1f}, {right_y:.1f})px")
    print(f"  视差: {disparity:.1f}px")

    if abs(disparity) <= 1e-6:
        print("  视差过小，无法计算坐标")
        return None

    # 计算像素高度差
    pixel_height_from_center = abs(left_y - CY)

    # 高度修正因子（物体在图像中的位置影响）
    normalized_height = pixel_height_from_center / CALIBRATED_RESOLUTION[1]
    height_correction = 1.0 + normalized_height * HEIGHT_CORRECTION_STRENGTH

    # 计算原始深度
    z_depth = (BASELINE * FX) / abs(disparity)

    # 距离修正（针对远距离的精度衰减）
    distance_factor = 1.0 + (z_depth / 1000) * DISTANCE_CORRECTION

    # 综合修正
    total_correction = BASE_ANGLE_CORRECTION * height_correction * distance_factor
    z_depth_corrected = z_depth * total_correction

    # 最终距离计算
    y_distance = np.sqrt(z_depth_corrected ** 2 + HEIGHT_DIFF ** 2)
    x_offset = - ((left_x - CX) * BASELINE) / disparity

    # 详细输出
    print(f"  原始深度: {z_depth:.1f}cm")
    print(f"  基础修正: {BASE_ANGLE_CORRECTION}")
    print(f"  高度修正: {height_correction:.3f}")
    print(f"  距离修正: {distance_factor:.3f}")
    print(f"  总修正系数: {total_correction:.3f}")
    print(f"  修正后深度: {z_depth_corrected:.1f}cm")
    print(f"  垂直距离（y）: {y_distance:.1f}cm")
    print(f"  水平偏移（x）: {x_offset:.1f}cm")
    print(f"  坐标: ({y_distance:.1f}cm, {x_offset:.1f}cm)")

    return (round(y_distance, 1), round(x_offset, 1))


# ---------------------- 新增：图像显示函数 ----------------------
# ---------------------- 新增：图像显示函数 ----------------------
def draw_detections_with_distance(image, detections, coord_results, is_left=True):
    """在图像上绘制检测框和距离信息"""
    display_img = image.copy()
    
    # 定义颜色
    color_red = (0, 0, 255)      # 红色 - BGR
    color_blue = (255, 0, 0)     # 蓝色
    color_yellow = (0, 255, 255) # 黄色
    color_white = (255, 255, 255) # 白色
    color_black = (0, 0, 0)      # 黑色
    color_green = (0, 255, 0)    # 绿色
    
    # 为每个检测结果绘制信息
    for det in detections:
        label = det['label']
        bbox = det['bbox']
        confidence = det['confidence']
        center = det['center']
        
        # 根据标签选择颜色
        if label == 'red':
            color = color_red
        elif label == 'blue':
            color = color_blue
        elif label == 'yellow':
            color = color_yellow
        else:
            color = color_white
        
        # 绘制边界框
        x1, y1, x2, y2 = map(int, bbox)
        cv2.rectangle(display_img, (x1, y1), (x2, y2), color, 2)
        
        # 绘制中心点
        center_x, center_y = map(int, center)
        cv2.circle(display_img, (center_x, center_y), 5, color_green, -1)
        
        # 查找对应的坐标结果 - 改进匹配逻辑
        coord_info = ""
        for result in coord_results:
            # 通过标签和中心点位置来匹配
            if (result['label'] == label and 
                abs(result['left_bbox' if is_left else 'right_bbox'][0] - x1) < 5 and
                abs(result['left_bbox' if is_left else 'right_bbox'][1] - y1) < 5):
                y_dist, x_offset = result['coords']
                coord_info = f"Dist:{y_dist}cm X:{x_offset}cm"
                break
        
        # 准备显示文本
        label_text = f"{label} {confidence:.2f}"
        if coord_info:
            info_text = coord_info
        else:
            info_text = "No match"
        
        # 计算文本位置
        text_y1 = max(y1 - 10, 20)  # 确保不超出图像顶部
        text_y2 = text_y1 - 25
        
        # 绘制文本背景 - 为标签文本
        (text_width1, text_height1), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        (text_width2, text_height2), _ = cv2.getTextSize(info_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        
        max_width = max(text_width1, text_width2)
        
        # 绘制标签文本背景
        cv2.rectangle(display_img, 
                     (x1, text_y2 - text_height1 - 5), 
                     (x1 + max_width + 10, text_y2 + 5), 
                     color, -1)
        
        # 绘制距离信息背景
        cv2.rectangle(display_img, 
                     (x1, text_y1 - text_height2 - 5), 
                     (x1 + max_width + 10, text_y1 + 5), 
                     color_black, -1)
        
        # 绘制标签文本
        cv2.putText(display_img, label_text, (x1+5, text_y2), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_black, 2)
        
        # 绘制距离信息
        cv2.putText(display_img, info_text, (x1+5, text_y1), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_white, 2)
    
    # 添加摄像头标识和统计信息
    cam_text = "Left Camera" if is_left else "Right Camera"
    cv2.putText(display_img, cam_text, (20, 40), 
               cv2.FONT_HERSHEY_SIMPLEX, 1, color_white, 2)
    
    # 显示检测到的锥桶数量
    count_text = f"Cones: {len(detections)}"
    cv2.putText(display_img, count_text, (20, 80), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_white, 2)
    
    # 显示有距离信息的锥桶数量
    matched_count = sum(1 for det in detections for result in coord_results 
                       if (result['label'] == det['label'] and 
                           abs(result['left_bbox' if is_left else 'right_bbox'][0] - det['bbox'][0]) < 5))
    distance_text = f"With Distance: {matched_count}"
    cv2.putText(display_img, distance_text, (20, 110), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_green, 2)
    
    return display_img


def create_combined_view(left_img, right_img, coord_results):
    """创建左右图像合并视图"""
    # 调整图像大小以适应显示
    scale_factor = 0.6
    new_width = int(left_img.shape[1] * scale_factor)
    new_height = int(left_img.shape[0] * scale_factor)
    
    left_resized = cv2.resize(left_img, (new_width, new_height))
    right_resized = cv2.resize(right_img, (new_width, new_height))
    
    # 创建合并图像
    combined = np.vstack([left_resized, right_resized])
    
    # 添加标题
    cv2.putText(combined, "STEREO VISION - Cone Detection with Distance", 
               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    # 添加坐标结果汇总
    if coord_results:
        cv2.putText(combined, "DETECTED CONES:", (10, 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        y_offset = 100
        for i, result in enumerate(coord_results):
            color = result['label']
            y_dist, x_offset = result['coords']
            
            # 根据颜色设置文本颜色
            if color == 'red':
                text_color = (0, 0, 255)
            elif color == 'blue':
                text_color = (255, 0, 0)
            elif color == 'yellow':
                text_color = (0, 255, 255)
            else:
                text_color = (255, 255, 255)
                
            color_text = f"{color.upper()}: Distance={y_dist}cm, Offset={x_offset}cm"
            cv2.putText(combined, color_text, (10, y_offset + i*30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)
    else:
        cv2.putText(combined, "No cones detected with distance", (10, 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    return combined


# ---------------------- 摄像头捕获 ----------------------
def camera_capture(cam_id, is_left):
    """摄像头捕获线程"""
    global left_frame, right_frame, running

    # ffmpeg命令捕获摄像头
    cmd = [
        "ffmpeg",
        "-f", "v4l2",
        "-input_format", "yuyv422",
        "-video_size", f"{CALIBRATED_RESOLUTION[0]}x{CALIBRATED_RESOLUTION[1]}",
        "-framerate", "25",
        "-i", f"/dev/video{cam_id}",
        "-f", "rawvideo",
        "-pix_fmt", "bgr24",
        "-"
    ]

    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=10 ** 8)

    try:
        while running:
            # 读取一帧数据
            frame_size = CALIBRATED_RESOLUTION[0] * CALIBRATED_RESOLUTION[1] * 3
            raw_frame = process.stdout.read(frame_size)
            if not raw_frame:
                break

            # 转换为OpenCV格式
            frame = np.frombuffer(raw_frame, dtype=np.uint8).reshape(
                (CALIBRATED_RESOLUTION[1], CALIBRATED_RESOLUTION[0], 3)
            )

            # 更新全局帧变量
            with frame_lock:
                if is_left:
                    left_frame = frame
                else:
                    right_frame = frame

            time.sleep(0.01)  # 控制帧率

    except Exception as e:
        print(f"摄像头{cam_id}错误: {e}")
    finally:
        process.terminate()


# ---------------------- 新增：CAN帧发送函数 ----------------------
def send_can_error_frame():
    """发送CAN错误帧（黄色锥桶）"""
    global can_client, can_client_ok

    if not can_client_ok:
        return False

    # CAN参数
    CAN_ID_ERROR = 0x200  # ID设置为200
    ERROR_DATA = [0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01]

    return can_client.send_can_frame(CAN_ID_ERROR, ERROR_DATA)


def send_can_normal_frame():
    """发送CAN正常帧（红色和蓝色锥桶）"""
    global can_client, can_client_ok

    if not can_client_ok:
        return False

    # CAN参数
    CAN_ID_NORMAL = 0x200  # ID设置为200
    NORMAL_DATA = [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00]

    return can_client.send_can_frame(CAN_ID_NORMAL, NORMAL_DATA)


# ---------------------- 信号处理与主函数 ----------------------
def signal_handler(sig, frame):
    """处理中断信号（Ctrl+C）"""
    global running, can_client
    print("\n收到中断信号，正在停止...")
    running = False
    # 关闭显示窗口
    if display_enabled:
        cv2.destroyAllWindows()
    # 关闭CAN客户端
    if can_client:
        can_client.close()
    sys.exit(0)


def main():
    global running, prev_has_yellow, prev_has_red_blue, cone_pub, can_client, can_client_ok

    # 1. 初始化CAN UDP客户端
    print("🔧 初始化CAN UDP客户端...")
    can_client = CANUDPClient()
    if not can_client.init():
        print("⚠️ CAN UDP客户端初始化失败，将继续运行但不发送CAN帧")
        can_client_ok = False
    else:
        can_client_ok = True

    # ---------------------- 新增：ROS初始化 ----------------------
    print("初始化ROS节点...")
    rospy.init_node('camera_yolo_node', anonymous=False)
    cone_pub = rospy.Publisher('/perception/cones', ConeDetection, queue_size=50)
    print("ROS节点已启动，话题：/perception/cones")
    # ---------------------------------------------------------

    # 注册信号处理
    signal.signal(signal.SIGINT, signal_handler)

    # 加载模型
    print("加载YOLO模型...")
    model = load_model(weight_path)

    # 启动双摄像头捕获线程
    print("启动双摄像头（设备0和1）...")
    left_thread = threading.Thread(target=camera_capture, args=(0, True))
    right_thread = threading.Thread(target=camera_capture, args=(1, False))
    left_thread.daemon = True
    right_thread.daemon = True
    left_thread.start()
    right_thread.start()

    # 等待摄像头初始化
    time.sleep(2)

    # CAN发送相关变量
    last_yellow_can_time = 0
    last_red_blue_can_time = 0
    can_send_interval = 0.1  # 每100ms发送一次CAN帧

    # FPS计算变量
    fps_counter = 0
    fps_timer = time.time()
    current_fps = 0

    try:
        print("\n=== 开始实时坐标检测 ===")
        print("输出格式：<颜色>锥桶坐标: (y, x) cm （y:深度, x:水平偏移，左侧为正）")
        print("测距修正参数:")
        print(f"  基础视角修正: {BASE_ANGLE_CORRECTION}")
        print(f"  高度修正强度: {HEIGHT_CORRECTION_STRENGTH}")
        print(f"  距离修正: {DISTANCE_CORRECTION}")
        print("当连续两帧识别到yellow标签时，将额外输出'stop'")
        print("同时发布ROS消息到 /perception/cones")
        print("识别到黄色锥桶时发送CAN错误帧（ID:0x200, 数据:0101010101010101）")
        print("识别到红色或蓝色锥桶时发送CAN正常帧（ID:0x200, 数据:0000000000000000）")
        print("图像窗口将显示实时检测结果和距离信息")
        print("按 'q' 在图像窗口中退出，或按 Ctrl+C 在终端退出\n")

        # 创建显示窗口
        if display_enabled:
            cv2.namedWindow(window_name_left, cv2.WINDOW_NORMAL)
            cv2.namedWindow(window_name_right, cv2.WINDOW_NORMAL)
            cv2.namedWindow(window_name_combined, cv2.WINDOW_NORMAL)
            # 调整窗口大小
            cv2.resizeWindow(window_name_left, 800, 450)
            cv2.resizeWindow(window_name_right, 800, 450)
            cv2.resizeWindow(window_name_combined, 800, 900)

        while running and not rospy.is_shutdown():
            # 等待左右帧都准备好
            with frame_lock:
                if left_frame is None or right_frame is None:
                    time.sleep(0.001)  # 更短的等待时间
                    continue
                current_left = left_frame.copy()
                current_right = right_frame.copy()

            # 检测锥桶
            left_dets = detect_objects(model, current_left)
            right_dets = detect_objects(model, current_right)

            # 匹配左右图中的锥桶（同颜色+Y坐标相近）
            matched_pairs = []
            right_used = set()
            for left_idx, left_det in enumerate(left_dets):
                for right_idx, right_det in enumerate(right_dets):
                    if right_idx in right_used:
                        continue
                    if left_det['label'] == right_det['label']:
                        # 垂直方向Y坐标差异小于30像素（确保是同一物体）
                        y_diff = abs(left_det['center'][1] - right_det['center'][1])
                        if y_diff < 30:
                            matched_pairs.append((left_idx, right_idx))
                            right_used.add(right_idx)
                            break

            # 计算坐标并输出结果
            coord_results = []
            current_has_yellow = False  # 记录当前帧是否识别到yellow
            current_has_red_blue = False  # 记录当前帧是否识别到红色或蓝色

            for left_idx, right_idx in matched_pairs:
                left_det = left_dets[left_idx]
                right_det = right_dets[right_idx]

                # 检查颜色类型
                if left_det['label'] == 'yellow':
                    current_has_yellow = True
                elif left_det['label'] in ['red', 'blue']:
                    current_has_red_blue = True

                # 计算（y, x）坐标
                coords = calculate_coordinates(
                    left_det['center'],
                    right_det['center'],
                    left_det['label']
                )
                if coords:
                    coord_results.append({
                        'label': left_det['label'],
                        'coords': coords,
                        'left_bbox': left_det['bbox'],
                        'right_bbox': right_det['bbox']
                    })

                    # ---------------------- 新增：发布ROS消息 ----------------------
                    try:
                        # 坐标转换：cm → m，坐标系调整
                        y_m = coords[0] / 100.0  # 深度 → x轴（前方）
                        x_m = coords[1] / 100.0  # 水平偏移 → y轴（左方）

                        msg = ConeDetection()
                        msg.header = Header(stamp=rospy.Time.now(), frame_id="camera")
                        msg.color = left_det['label'].lower()  # 确保小写
                        msg.x = y_m  # 相机坐标系：x=前方距离
                        msg.y = x_m  # y=左侧距离（左正右负）
                        msg.z = 0.0  # 地面高度设为0

                        cone_pub.publish(msg)
                        print(f"[ROS发布] {msg.color}锥桶 - x:{msg.x:.2f}m y:{msg.y:.2f}m")

                    except Exception as e:
                        print(f"ROS消息发布失败: {e}")
                    # ---------------------------------------------------------

            # 新增：检查连续两帧是否都识别到yellow，是则输出stop并发送CAN错误帧
            if current_has_yellow and prev_has_yellow:
                print("\n" + "=" * 20)
                print("              stop")  # 居中显示stop
                print("=" * 20 + "\n")

                # 识别到黄色锥桶时发送CAN错误帧
                current_time = time.time()
                if current_time - last_yellow_can_time >= can_send_interval:
                    if can_client_ok:
                        send_can_error_frame()
                    last_yellow_can_time = current_time

            # 新增：检查连续两帧是否都识别到红色或蓝色，是则发送CAN正常帧
            if current_has_red_blue and prev_has_red_blue:
                # 识别到红色或蓝色锥桶时发送CAN正常帧
                current_time = time.time()
                if current_time - last_red_blue_can_time >= can_send_interval:
                    if can_client_ok:
                        send_can_normal_frame()
                    last_red_blue_can_time = current_time

            # 更新前一帧的颜色识别状态
            prev_has_yellow = current_has_yellow
            prev_has_red_blue = current_has_red_blue

            # 终端输出汇总结果
            if coord_results:
                print("\n=== 实时坐标结果 ===")
                for res in coord_results:
                    print(f"{res['label']}锥桶坐标: {res['coords']} cm")

            # ---------------------- 新增：图像显示 ----------------------
            if display_enabled:
                # 计算FPS
                fps_counter += 1
                if time.time() - fps_timer >= 1.0:
                    current_fps = fps_counter
                    fps_counter = 0
                    fps_timer = time.time()

                # 绘制检测结果
                left_display = draw_detections_with_distance(current_left, left_dets, coord_results, is_left=True)
                right_display = draw_detections_with_distance(current_right, right_dets, coord_results, is_left=False)
                
                # 更新FPS显示
                fps_text = f"FPS: {current_fps}"
                cv2.putText(left_display, fps_text, (20, 80), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(right_display, fps_text, (20, 80), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # 创建合并视图
                combined_display = create_combined_view(left_display, right_display, coord_results)
                
                # 显示图像
                cv2.imshow(window_name_left, left_display)
                cv2.imshow(window_name_right, right_display)
                cv2.imshow(window_name_combined, combined_display)
                
                # 检查按键
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == ord('Q'):
                    print("收到退出信号，正在停止...")
                    running = False
                    break

    except Exception as e:
        print(f"运行错误: {e}")
        traceback.print_exc()
    finally:
        running = False
        if display_enabled:
            cv2.destroyAllWindows()
        if can_client:
            can_client.close()
        print("程序已停止")


if __name__ == '__main__':
    main()
