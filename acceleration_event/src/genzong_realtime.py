#!/usr/bin/env python
# -*- coding: utf-8 -*-
import rospy
import numpy as np
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Float64, Header, String
import tf.transformations as tf_trans
import math
import time
import os
import sys
import csv
import threading
import json
from data_quality import (PoseBuffer, fresh, monotonic, map_line_to_local,
                          centerline_from_status, finite, LongitudinalProgress)
from datetime import datetime

# 新增：导入fssim消息类型
try:
    from fssim_common.msg import Cmd, State
    FSSIM_AVAILABLE = True
    rospy.loginfo("✅ fssim_common包可用")
except ImportError:
    FSSIM_AVAILABLE = False
    rospy.logwarn("⚠️ fssim_common包未安装，将无法使用FSSIM仿真器功能")

# 新增：导入UDP通信相关模块
import socket
import struct

# 直线加速赛VCU转向CAN协议
# CAN ID: 0x210
# 数据: uint16 little-endian
CAN_ID_STEERING = 0x210

class CANUDPClient:
    """CAN UDP客户端，用于发送CAN帧"""
    
    def __init__(self):
        self.sock = None
        self.dest_ip = "192.168.0.7"  # 目标IP地址
        self.dest_port = 20005  # CAN1端口20001
        self.self_port = 11312  # 使用不同的本地端口，避免冲突
        self.addr_to = None

    def init(self, dest_ip=None, dest_port=None, self_port=None):
        """初始化UDP客户端，支持ROS参数覆盖，保持原车端默认值。"""
        try:
            if dest_ip is None:
                dest_ip = rospy.get_param("~can_dest_ip", "192.168.0.7")
            if dest_port is None:
                dest_port = int(rospy.get_param("~can_dest_port", 20005))
            if self_port is None:
                self_port = int(rospy.get_param("~can_self_port", 11312))

            self.dest_ip = dest_ip
            self.dest_port = dest_port
            self.self_port = self_port

            rospy.loginfo("🔧 初始化UDP: 目标=%s:%d, 本地端口=%d", dest_ip, dest_port, self_port)

            # 创建UDP socket
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

            # 设置socket选项
            self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

            # 绑定本地端口
            self.sock.bind(('', self_port))

            # 设置目标地址
            self.addr_to = (dest_ip, dest_port)

            rospy.loginfo("✅ CAN UDP客户端初始化成功")
            return True

        except Exception as e:
            rospy.logerr("❌ CAN UDP客户端初始化失败: %s", str(e))
            return False

    def send_steering_frame(self, steering_angle, stop=False):
        """发送直线加速赛转向CAN帧。

        正常转向：输入steering_angle为弧度，±90°映射到raw 0~1000。
        停车：复用同一条0x210帧，把raw置为2000，不增加额外CAN帧。
        """
        if self.sock is None:
            rospy.logwarn("❌ UDP socket未初始化")
            return False

        try:
            if stop:
                steering_scaled = 2000
                log_angle = 0.0
            else:
                # 正常转向：±90° -> 0~1000
                steering_angle = max(
                    -math.pi / 2.0,
                    min(math.pi / 2.0, float(steering_angle))
                )
                steering_scaled = int(round(
                    500.0 + steering_angle * 1000.0 / math.pi
                ))
                steering_scaled = max(0, min(1000, steering_scaled))
                log_angle = steering_angle

            data = bytearray(8)
            data[0] = steering_scaled & 0xFF
            data[1] = (steering_scaled >> 8) & 0xFF

            # 根据USR-CANET200协议构建13字节数据包
            packet = bytearray(13)

            # 帧信息字节 (第1字节)
            # Bit7: FF - 0=标准帧, 1=扩展帧
            # Bit6: RTR - 0=数据帧, 1=远程帧  
            # Bit5-4: 保留位 (00)
            # Bit3-0: 数据长度 (0-8)
            frame_info = 0x00  # 标准帧 + 数据帧 + 保留位00
            frame_info |= 0x08  # 数据长度=8字节

            packet[0] = frame_info

            # CAN ID (第2-5字节) - 大端序，高位在前
            # 标准帧 (11位)
            packet[1] = 0x00
            packet[2] = 0x00
            packet[3] = (CAN_ID_STEERING >> 8) & 0xFF
            packet[4] = CAN_ID_STEERING & 0xFF

            # CAN数据 (第6-13字节)
            for i in range(8):
                packet[5 + i] = data[i]

            # 发送数据包
            sent = self.sock.sendto(bytes(packet), self.addr_to)

            if sent == len(packet):
                # 打印实际发送的数据内容
                data_hex = ' '.join(['{:02X}'.format(b) for b in data[:2]])  # 只显示前2字节
                normalized = steering_scaled / 500.0 - 1.0
                rospy.logdebug(
                    "✅ 发送转向CAN: ID=0x{:03X}, angle={:.3f}°, norm={:.3f}, raw={}, data=[{}]".format(
                        CAN_ID_STEERING,
                        math.degrees(log_angle),
                        normalized,
                        steering_scaled,
                        data_hex,
                    )
                )
                return True
            else:
                rospy.logwarn("❌ 发送CAN帧失败")
                return False

        except Exception as e:
            rospy.logerr("❌ 发送转向角CAN帧时出错: %s", str(e))
            return False

    def close(self):
        """关闭socket"""
        if self.sock:
            self.sock.close()
            self.sock = None
            rospy.loginfo("✅ CAN UDP客户端已关闭")

class ControlLogger:
    """控制日志记录器"""
    
    def __init__(self, log_dir="/tmp/path_tracking_logs"):
        """初始化日志记录器"""
        self.log_dir = log_dir
        self.csv_file = None
        self.csv_writer = None
        
        # 创建日志目录
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        
        # 创建日志文件
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_filename = os.path.join(log_dir, "control_log_{}.csv".format(timestamp))
        
        try:
            self.csv_file = open(log_filename, 'w')
            self.csv_writer = csv.writer(self.csv_file)
            
            # 写入CSV表头
            headers = [
                'timestamp',           # 时间戳
                'elapsed_time',        # 运行时间(秒)
                'vehicle_x',           # 车辆X坐标
                'vehicle_y',           # 车辆Y坐标
                'vehicle_yaw',         # 车辆航向角(弧度)
                'vehicle_speed',       # 新增：车辆速度
                'lateral_error',       # 横向偏差
                'heading_error',       # 航向偏差(弧度)
                'steering_angle',      # 转向角(弧度)
                'throttle',           # 新增：油门值
                'slam_received',       # 是否接收到SLAM定位(0/1)
                'fssim_received',      # 新增：是否接收到FSSIM定位(0/1)
                'path_received',       # 是否接收到路径(0/1)
                'lookahead_found',     # 是否找到前瞻点(0/1)
                'controller_started',  # 控制器是否启动(0/1)
                'target_point_index',  # 目标点索引
                'total_distance',      # 兼容列名：固定赛道方向的有符号进度
                'lookahead_distance',  # 前瞻距离
                'visited_points',      # 已访问点数量
                'total_path_points',   # 路径总点数
                'udp_sent',           # UDP是否发送成功
                'control_mode',       # 控制模式(slam/fssim)
                'local_centerline_active',
                'local_centerline_pairs',
                'local_centerline_k',
                'local_centerline_b',
                'lookahead_mode',
                'pose_age',
                'input_reason',
                'stop_cmd',
                'centerline_source',
                'centerline_confidence',
                'centerline_blend_weight',
                'observation_age',
                'finish_progress',
                'finish_reference_mode',
                'finish_reference_x',
                'finish_reference_y',
                'finish_axis_yaw',
                'finish_pose_stamp',
                'finish_confirm_count',
                'finish_reference_stamp'
            ]
            self.csv_writer.writerow(headers)
            
            rospy.loginfo("✅ 日志文件创建成功: %s", log_filename)
            self.start_time = time.time()
            
        except Exception as e:
            rospy.logerr("❌ 创建日志文件失败: %s", str(e))
            self.csv_file = None
            self.csv_writer = None
    
    def log_control_data(self, data_dict):
        """记录控制数据"""
        if self.csv_writer is None:
            return
        
        try:
            # 计算运行时间
            elapsed_time = time.time() - self.start_time
            
            # 准备数据行
            row = [
                datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],  # 时间戳(毫秒精度)
                "{:.3f}".format(elapsed_time),
                "{:.6f}".format(data_dict.get('vehicle_x', 0.0)),
                "{:.6f}".format(data_dict.get('vehicle_y', 0.0)),
                "{:.6f}".format(data_dict.get('vehicle_yaw', 0.0)),
                "{:.6f}".format(data_dict.get('vehicle_speed', 0.0)),  # 新增：车辆速度
                "{:.6f}".format(data_dict.get('lateral_error', 0.0)),
                "{:.6f}".format(data_dict.get('heading_error', 0.0)),
                "{:.6f}".format(data_dict.get('steering_angle', 0.0)),
                "{:.6f}".format(data_dict.get('throttle', 0.0)),      # 新增：油门值
                1 if data_dict.get('slam_received', False) else 0,
                1 if data_dict.get('fssim_received', False) else 0,   # 新增：FSSIM接收状态
                1 if data_dict.get('path_received', False) else 0,
                1 if data_dict.get('lookahead_found', False) else 0,
                1 if data_dict.get('controller_started', False) else 0,
                data_dict.get('target_point_index', -1),
                "{:.3f}".format(data_dict.get('total_distance', 0.0)),
                "{:.3f}".format(data_dict.get('lookahead_distance', 0.0)),
                data_dict.get('visited_points', 0),
                data_dict.get('total_path_points', 0),
                1 if data_dict.get('udp_sent', False) else 0,
                data_dict.get('control_mode', 'unknown'),              # 控制模式
                1 if data_dict.get('local_centerline_active', False) else 0,
                data_dict.get('local_centerline_pairs', 0),
                "{:.6f}".format(data_dict.get('local_centerline_k', 0.0)),
                "{:.6f}".format(data_dict.get('local_centerline_b', 0.0)),
                data_dict.get('lookahead_mode', 'global'),
                data_dict.get('pose_age', ''),
                data_dict.get('input_reason', ''),
                data_dict.get('stop_cmd', 0),
                data_dict.get('centerline_source', ''),
                data_dict.get('centerline_confidence', 0.0),
                data_dict.get('centerline_blend_weight', 0.0),
                data_dict.get('observation_age', ''),
                data_dict.get('finish_progress', ''),
                data_dict.get('finish_reference_mode', ''),
                data_dict.get('finish_reference_x', ''),
                data_dict.get('finish_reference_y', ''),
                data_dict.get('finish_axis_yaw', ''),
                data_dict.get('finish_pose_stamp', ''),
                data_dict.get('finish_confirm_count', ''),
                data_dict.get('finish_reference_stamp', '')
            ]
            
            # 写入数据
            self.csv_writer.writerow(row)
            self.csv_file.flush()  # 立即刷新到文件
            
        except Exception as e:
            rospy.logerr("❌ 写入日志数据失败: %s", str(e))
    
    def close(self):
        """关闭日志文件"""
        if self.csv_file:
            try:
                self.csv_file.close()
                rospy.loginfo("✅ 日志文件已关闭")
            except Exception as e:
                rospy.logerr("❌ 关闭日志文件失败: %s", str(e))

class PathTrackingController:
    def __init__(self):
        try:
            rospy.init_node('path_tracking_controller', anonymous=True)
            
            # ================== 控制与赛道参数 ==================
            # 无人直线加速赛：赛道宽度3m，锥桶间隔5m，实车最高约15km/h
            self.track_width = rospy.get_param('~track_width', 3.0)
            self.max_vehicle_speed_mps = rospy.get_param('~max_vehicle_speed_mps', 15.0 / 3.6)

            # 控制器内部限幅：直线赛道先限制到约±20°，避免小误差触发大幅转向。
            # 不改变VCU协议，0x210仍按原协议发送。
            self.max_steering = rospy.get_param('~max_steering', 0.28)
            # Independent fixed-axis finish progress, checked before path validity.
            self.finish_distance = rospy.get_param('~finish_distance', 75.0)
            self.finish_start_mode = rospy.get_param('~finish_start_mode', 'first_valid_pose')
            if self.finish_start_mode not in ('first_valid_pose', 'configured'):
                raise ValueError('finish_start_mode must be first_valid_pose or configured')
            finish_origin = None
            if self.finish_start_mode == 'configured':
                finish_origin = (rospy.get_param('~finish_start_x', 0.0),
                                 rospy.get_param('~finish_start_y', 0.0))
            self.finish_progress = LongitudinalProgress(
                self.finish_distance, rospy.get_param('~finish_axis_yaw', 0.0),
                finish_origin, rospy.get_param('~finish_confirm_samples', 1))
            self.finish_triggered = False
            self.total_distance = 0.0
            self.pose_stamp = None
            rospy.loginfo('Finish progress: mode=%s origin=%s axis=%.3frad distance=%.2fm confirm=%d',
                          self.finish_start_mode, str(self.finish_progress.origin),
                          self.finish_progress.axis_yaw, self.finish_progress.distance,
                          self.finish_progress.confirm_samples)
            self.steering_sign = rospy.get_param('~steering_sign', 1.0)
            # 20Hz控制周期下，单周期最多变化约2.3°，降低反打/过冲。
            self.max_steering_step = rospy.get_param('~max_steering_step', 0.012)

            # FSSIM调试速度设为接近实车上限
            self.fixed_speed = rospy.get_param('~fixed_speed', self.max_vehicle_speed_mps)

            # 直线赛道稳态PD：
            # 横向误差作为主项；航向误差降权，避免前瞻点方位角与横向误差重复放大。
            self.kp_lateral = rospy.get_param('~kp_lateral', 0.48)
            self.kd_lateral = rospy.get_param('~kd_lateral', 0.05)
            self.kp_heading = rospy.get_param('~kp_heading', 0.10)
            self.kd_heading = rospy.get_param('~kd_heading', 0.01)
            self.error_filter_alpha = rospy.get_param('~error_filter_alpha', 0.12)

            rospy.loginfo(
                "🎯 直线PD控制器 v2: K_lat=%.2f Kd_lat=%.2f K_head=%.2f Kd_head=%.2f max=%.1f° step=%.1f°",
                self.kp_lateral, self.kd_lateral,
                self.kp_heading, self.kd_heading,
                math.degrees(self.max_steering),
                math.degrees(self.max_steering_step)
            )

            # 航向项自适应权重：
            # 车辆接近中心线时主要看横向误差，防止前瞻点方位角造成大幅转向；
            # 车辆偏得较远时再逐步恢复航向项，帮助车辆重新对正。
            self.heading_weight_min = rospy.get_param('~heading_weight_min', 0.35)
            self.heading_weight_full_lateral_error = rospy.get_param(
                '~heading_weight_full_lateral_error', 0.80)

            # 防止误差微分项在定位/路径更新瞬间产生“微分冲击”。
            self.max_lateral_derivative = rospy.get_param('~max_lateral_derivative', 0.60)
            self.max_heading_derivative = rospy.get_param('~max_heading_derivative', 0.35)

            # 速度相关前瞻：15km/h时约6.4m，限制在4.5~7m
            self.lookahead_min = rospy.get_param('~lookahead_min', 2.5)
            self.lookahead_max = rospy.get_param('~lookahead_max', 4.0)
            self.lookahead_speed_gain = rospy.get_param('~lookahead_speed_gain', 0.35)
            self.lookahead_distance = self.lookahead_min

            # The planner is the only cone-pairing and road-estimation authority.
            self.local_blend_two_pairs = rospy.get_param('~local_blend_two_pairs', 0.65)
            self.local_blend_one_pair = rospy.get_param('~local_blend_one_pair', 0.45)
            self.local_blend_single_side = rospy.get_param('~local_blend_single_side', 0.20)

            self.local_centerline_k = 0.0
            self.local_centerline_b = 0.0
            # Latest planner estimate in map; local k/b are derived for logging.
            self.local_centerline_map_k = 0.0
            self.local_centerline_map_b = 0.0
            self.local_centerline_observation_stamp = None
            self.local_centerline_confidence = 0.0
            self.local_centerline_pairs = 0
            self.local_centerline_active = False

            # ================== 实车速度估计 ==================
            self.last_slam_pose_x = None
            self.last_slam_pose_y = None
            self.last_slam_pose_time = None
            self.slam_speed_estimate = 0.0

            # 车辆状态 - 支持多种定位源
            self.vehicle_x = 0.0
            self.vehicle_y = 0.0
            self.vehicle_yaw = 0.0
            self.vehicle_speed = 0.0  # 新增：车辆速度
            self.pose_received = False
            self.initial_x = None
            self.initial_y = None
            self.total_distance = 0.0
            
            # 定位源管理 - 智能选择
            self.fssim_available = FSSIM_AVAILABLE
            self.fssim_pose_received = False
            self.slam_pose_received = False
            self.last_fssim_time = None
            self.last_slam_time = None
            self.localization_mode = rospy.get_param('~localization_mode', 'slam')
            if self.localization_mode not in ('slam', 'fssim'):
                raise ValueError('localization_mode must be slam or fssim')
            self.pose_timeout = rospy.get_param('~pose_timeout', 0.5)
            self.path_timeout = rospy.get_param('~path_timeout', 0.5)
            self.path_observation_timeout = rospy.get_param('~path_observation_timeout', 0.5)
            self.slam_buffer = PoseBuffer(rospy.get_param('~pose_max_speed', 12.0))
            self.fssim_buffer = PoseBuffer(rospy.get_param('~pose_max_speed', 12.0))
            self.input_lock = threading.RLock()
            self.input_reason = 'missing'
            self.pose_age = None
            self.path_receive_time = None
            self.planning_status = None
            self.status_receive_time = None
            self.fssim_timeout = 2.0  # FSSIM数据超时时间（秒）
            self.control_mode = "none"  # 当前使用的定位源
            
            # 独立的车辆状态变量，避免数据覆盖
            self.fssim_x = 0.0
            self.fssim_y = 0.0
            self.fssim_yaw = 0.0
            self.fssim_speed = 0.0
            
            self.slam_x = 0.0
            self.slam_y = 0.0
            self.slam_yaw = 0.0
            
            # 路径信息
            self.current_path = None
            self.path_received = False
            self.target_point_index = 0
            self.visited_indices = set()
            self.visit_threshold = 1.0
            
            # 控制状态
            self.prev_lateral_error = 0.0
            self.prev_heading_error = 0.0
            self.filtered_lateral_error = 0.0
            self.filtered_heading_error = 0.0
            self.prev_steering_angle = 0.0
            self.prev_time = rospy.Time.now()
            
            # 启动标志 - 基于路径和定位判断
            self.is_started = False
            self.start_time = None
            
            # 初始化日志记录器
            log_dir = rospy.get_param('~log_dir', '/tmp/path_tracking_logs')
            self.logger = ControlLogger(log_dir)
            
            # 初始化CAN UDP客户端
            self.can_client = CANUDPClient()
            self.enable_vehicle_output = rospy.get_param('~enable_vehicle_output', False)
            self.can_client_ok = self.can_client.init() if self.enable_vehicle_output else False
            if not self.enable_vehicle_output:
                rospy.logwarn('VALIDATION MODE: UDP/CAN, ROS steering and FSSIM output disabled')
            if not self.can_client_ok:
                rospy.logwarn("⚠️ CAN UDP客户端初始化失败，将继续运行但不发送CAN帧")
            else:
                rospy.loginfo("✅ CAN UDP客户端初始化成功 - 将发送转向角数据(ID:0x210)")
            
            # 发布器和订阅器
            self.quality_pub = rospy.Publisher('/control/input_status', String, queue_size=2)
            self.planning_status_sub = rospy.Subscriber('/planning/status', String, self.planning_status_callback, queue_size=5)
            self.steering_pub = (rospy.Publisher('/control/steering', Float64, queue_size=10)
                                 if self.enable_vehicle_output else None)
            
            # 新增：FSSIM控制命令发布器
            if self.fssim_available and self.enable_vehicle_output:
                self.cmd_pub = rospy.Publisher('/fssim/cmd', Cmd, queue_size=10)
                rospy.loginfo("✅ FSSIM控制命令发布器已创建: /fssim/cmd")
            else:
                self.cmd_pub = None
                rospy.logwarn("⚠️ FSSIM不可用，将无法发布控制命令到仿真器")
            
            # 定位订阅器 - 同时订阅两种定位源
            if self.fssim_available:
                self.fssim_pose_sub = rospy.Subscriber('/fssim/base_pose_ground_truth', State, self.fssim_pose_callback)
                rospy.loginfo("✅ 订阅FSSIM定位数据: /fssim/base_pose_ground_truth")
            
            self.slam_pose_sub = rospy.Subscriber('/vehicle_pose', PoseStamped, self.slam_pose_callback)
            rospy.loginfo("✅ 订阅SLAM定位数据: /vehicle_pose")
            
            self.path_sub = rospy.Subscriber('/planned_path', Path, self.path_callback)
            
            # 定位源检查定时器
            self.source_check_timer = rospy.Timer(rospy.Duration(1.0), lambda event: rospy.loginfo_throttle(5.0, 'Input source=%s reason=%s', self.localization_mode, self.input_reason))
            
            # 控制定时器
            self.control_timer = rospy.Timer(rospy.Duration(0.05), self.control_loop)  # 20Hz
            
            rospy.loginfo("🚗 实时中心线跟踪控制器已初始化")
            rospy.loginfo("📏 赛道宽度=%.2fm，锥桶间隔=5m，最大速度=%.2fm/s(15km/h)",
                          self.track_width, self.max_vehicle_speed_mps)
            rospy.loginfo("中心线唯一来源: 与当前路径匹配的 /planning/status，按观测质量和年龄加权")
            rospy.loginfo("🔭 前瞻距离：%.1f~%.1fm，随速度变化",
                          self.lookahead_min, self.lookahead_max)
            rospy.loginfo("📊 日志记录已启用，日志目录: %s", log_dir)
            if self.can_client_ok:
                rospy.loginfo("🔌 CAN UDP通信已启用 - 转向角将通过CAN帧发送")
            if self.fssim_available:
                rospy.loginfo("🎮 FSSIM仿真器支持已启用")
            rospy.loginfo("⏱️ FSSIM数据超时时间: %.1f秒", self.fssim_timeout)
            rospy.loginfo("🔄 等待车辆状态和路径信息...")
            
        except Exception as e:
            rospy.logerr("控制器初始化失败: %s", str(e))
            import traceback
            rospy.logerr("详细错误: %s", traceback.format_exc())
            raise
    
    def check_position_sources(self, event=None):
        # Every control cycle, select one complete measurement snapshot.
        buffer = self.slam_buffer if self.localization_mode == 'slam' else self.fssim_buffer
        sample = buffer.latest(rospy.Time.now().to_sec(), self.pose_timeout)
        self.pose_received = bool(sample and sample['valid'])
        self.input_reason = sample['reason'] if sample else 'missing'
        self.pose_age = sample['age'] if sample else None
        self.control_mode = self.localization_mode if self.pose_received else 'none'
        if self.pose_received:
            self.vehicle_x, self.vehicle_y = sample['x'], sample['y']
            self.vehicle_yaw, self.vehicle_speed = sample['yaw'], sample['speed']
            self.pose_stamp = sample['stamp']
            self.check_start_condition()

    def slam_pose_callback(self, msg):
        q = msg.pose.orientation
        values = [q.x, q.y, q.z, q.w]
        if not all((not math.isnan(v) and not math.isinf(v)) for v in values) or sum(v*v for v in values) < 1e-12:
            self.slam_buffer.reason = 'invalid_orientation'
            return
        _, _, yaw = tf_trans.euler_from_quaternion(values)
        accepted = self.slam_buffer.add(msg.header.stamp.to_sec(),
                                       msg.pose.position.x, msg.pose.position.y, yaw)
        self.slam_pose_received = accepted

    def fssim_pose_callback(self, msg):
        speed = None
        if hasattr(msg, 'vx') and hasattr(msg, 'vy'):
            speed = math.hypot(msg.vx, msg.vy)
        else:
            for field in ('u', 'speed', 'vel'):
                if hasattr(msg, field):
                    speed = abs(float(getattr(msg, field)))
                    break
        header = getattr(msg, 'header', None)
        stamp = header.stamp.to_sec() if header is not None else 0.0
        self.fssim_pose_received = self.fssim_buffer.add(stamp, msg.x, msg.y, msg.yaw, speed=speed)

    def planning_status_callback(self, msg):
        with self.input_lock:
            self._planning_status_callback(msg)

    def _planning_status_callback(self, msg):
        try:
            status = json.loads(msg.data)
            if not isinstance(status, dict):
                self.planning_status = None
                return
            self.planning_status = status
            self.status_receive_time = monotonic()
        except (TypeError, ValueError):
            self.planning_status = None

    def path_is_fresh(self):
        now = rospy.Time.now().to_sec()
        status = self.planning_status or {}
        return bool(self.path_received and self.current_path and
                    fresh(self.current_path.header.stamp.to_sec(), now, self.path_timeout) and
                    fresh(self.path_receive_time, monotonic(), self.path_timeout) and
                    fresh(self.status_receive_time, monotonic(), self.path_timeout) and
                    status.get('valid') is True and
                    status.get('path_stamp') == self.current_path.header.stamp.to_sec() and
                    centerline_from_status(status, now, self.path_observation_timeout) is not None)

    def publish_input_status(self, reason):
        finish = self.finish_status()
        status = dict(
            source=self.localization_mode, pose_age=self.pose_age,
            reason=reason, stop_latched=self.finish_triggered)
        status.update(finish)
        self.quality_pub.publish(String(data=json.dumps(status)))
        if reason != 'valid':
            finish.update(pose_age=self.pose_age, input_reason=reason,
                          stop_cmd=int(self.finish_triggered), control_mode=self.control_mode,
                          total_distance=self.total_distance)
            self.logger.log_control_data(finish)

    def hold_normal_stop(self):
        # Normal finish remains latched even when tracking inputs disappear.
        if self.can_client_ok:
            self.can_client.send_steering_frame(0.0, stop=True)
        self.publish_steering_command(0.0)
        if self.enable_vehicle_output and self.localization_mode == 'fssim' and self.cmd_pub is not None:
            cmd = Cmd()
            cmd.dc, cmd.delta = 0.0, 0.0
            self.cmd_pub.publish(cmd)

    def update_speed_source(self):
        # Speed belongs to the same snapshot as pose. Clamp only lookahead.
        pass

    def update_dynamic_lookahead(self):
        v = min(max(self.vehicle_speed, 0.0), self.max_vehicle_speed_mps)
        self.lookahead_distance = self.lookahead_min + self.lookahead_speed_gain * v
        self.lookahead_distance = max(self.lookahead_min,
                                      min(self.lookahead_max, self.lookahead_distance))
        return self.lookahead_distance

    def world_to_vehicle(self, world_x, world_y):
        """世界坐标 -> 车辆坐标，车辆坐标定义为X前、Y右。"""
        dx = world_x - self.vehicle_x
        dy = world_y - self.vehicle_y
        c = math.cos(self.vehicle_yaw)
        s = math.sin(self.vehicle_yaw)
        return dx * c + dy * s, -dx * s + dy * c

    def vehicle_to_world(self, local_x, local_y):
        """车辆坐标 -> 世界坐标。"""
        c = math.cos(self.vehicle_yaw)
        s = math.sin(self.vehicle_yaw)
        return (self.vehicle_x + local_x * c - local_y * s,
                self.vehicle_y + local_x * s + local_y * c)

    def local_centerline_result(self, confidence):
        if math.cos(self.vehicle_yaw) + self.local_centerline_map_k * math.sin(self.vehicle_yaw) <= 1e-6:
            return None
        line = map_line_to_local(self.local_centerline_map_k, self.local_centerline_map_b,
                                 self.vehicle_x, self.vehicle_y, self.vehicle_yaw)
        if line is None:
            return None
        self.local_centerline_k, self.local_centerline_b = line
        return dict(k=line[0], b=line[1], pairs=self.local_centerline_pairs,
                    confidence=confidence, source=self.planning_status['centerline']['source'],
                    observation_stamp=self.local_centerline_observation_stamp)

    def estimate_local_centerline(self):
        """Project the matched planner estimate; never independently pair or fit cones."""
        self.local_centerline_active = False
        if not self.path_is_fresh():
            return None
        line = centerline_from_status(self.planning_status, rospy.Time.now().to_sec(),
                                      self.path_observation_timeout)
        if line is None:
            return None
        self.local_centerline_map_k = line['map_k']
        self.local_centerline_map_b = line['map_b']
        self.local_centerline_pairs = line['pairs']
        self.local_centerline_observation_stamp = line['observation_stamp']
        self.local_centerline_confidence = line['confidence']
        result = self.local_centerline_result(line['confidence'])
        self.local_centerline_active = result is not None
        return result

    def get_local_centerline_target(self, lookahead):
        result = self.estimate_local_centerline()
        if result is None:
            return None
        target_y = result['k'] * lookahead + result['b']
        wx, wy = self.vehicle_to_world(lookahead, target_y)
        return dict(x=wx, y=wy, k=result['k'], b=result['b'], pairs=result['pairs'],
                    confidence=result['confidence'], source=result['source'],
                    observation_stamp=result['observation_stamp'])

    def local_blend_weight(self, target):
        # Confidence already incorporates the age of the original observation.
        if target is None:
            return 0.0
        confidence = target.get('confidence', 0.0)
        if not finite(confidence) or confidence <= 0.0:
            return 0.0
        if target['source'] == 'two_pairs':
            base = self.local_blend_two_pairs
        elif target['source'] == 'one_pair':
            base = self.local_blend_one_pair
        elif target['source'] == 'single_side':
            base = self.local_blend_single_side
        else:
            return 0.0
        return max(0.0, min(1.0, base)) * min(1.0, confidence)

    def update_visited_points(self):
        """更新已访问的路径点"""
        if not self.current_path or not self.pose_received:
            return
        
        # 检查车辆是否通过了路径上的点
        for i, pose in enumerate(self.current_path.poses):
            if i in self.visited_indices:
                continue
                
            # 计算车辆到路径点的距离
            dx = pose.pose.position.x - self.vehicle_x
            dy = pose.pose.position.y - self.vehicle_y
            distance = math.sqrt(dx*dx + dy*dy)
            
            # 如果距离小于阈值，标记为已访问
            if distance < self.visit_threshold:
                self.visited_indices.add(i)
                rospy.loginfo("📍 通过路径点 %d，距离: %.2f m", i, distance)
    
    def path_callback(self, msg):
        with self.input_lock:
            self._path_callback(msg)

    def _path_callback(self, msg):
        """接收实时规划路径：路径点数量相同也必须更新。"""
        if len(msg.poses) <= 0:
            rospy.logwarn_throttle(2.0, "接收到空路径")
            self.path_received = False
            return

        # 实时规划器会以固定点数高频发布，因此不能再用“长度是否变化”判断新路径。
        # 直接用消息时间戳和首点位置判断，并保留路径为最新版本。
        old_stamp = self.current_path.header.stamp if self.current_path is not None else rospy.Time(0)
        new_stamp = msg.header.stamp
        is_new = self.current_path is None or new_stamp > old_stamp

        if not is_new:
            return

        # 确保路径方向与当前车辆行驶方向一致；实时局部路径通常已经是前向的。
        if self.pose_received:
            self.align_path_with_vehicle(msg)

        self.current_path = msg
        self.path_receive_time = monotonic()
        self.path_received = True

        # 实时局部路径每次都是从车辆当前位置重新生成，旧的 visited_indices 不再具有意义。
        self.target_point_index = 0
        self.visited_indices = set()

        rospy.logdebug_throttle(1.0,
                                "🛣️ 实时更新路径，%d点，stamp=%.3f",
                                len(msg.poses), new_stamp.to_sec())

        self.check_start_condition()

    def check_start_condition(self):
        """检查启动条件：当同时接收到路径和车辆定位时启动控制"""
        if not self.is_started and self.pose_received and self.path_received:
            self.is_started = True
            self.start_time = time.time()
            rospy.loginfo("✅ 启动条件满足：已接收到路径和车辆定位，开始路径跟踪")
            rospy.loginfo("📍 当前定位源: %s", self.control_mode)
            rospy.loginfo("🚗 车辆初始位置: (%.2f, %.2f), 初始航向: %.2f°", 
                         self.vehicle_x, self.vehicle_y, math.degrees(self.vehicle_yaw))
            rospy.loginfo("🛣️ 路径点数: %d", len(self.current_path.poses))
        else:
            # 调试信息：显示为什么没有启动
            if not self.is_started:
                rospy.loginfo_throttle(5, "⏳ 启动条件检查: pose_received=%s, path_received=%s, is_started=%s", 
                                      self.pose_received, self.path_received, self.is_started)

    def align_path_with_vehicle(self, path_msg):
        """确保路径方向与车辆初始方向一致"""
        if not self.pose_received:
            return

        # 计算路径第一个点的航向角
        first_point = path_msg.poses[0].pose.position
        second_point = path_msg.poses[1].pose.position if len(path_msg.poses) > 1 else first_point

        path_direction = math.atan2(second_point.y - first_point.y, 
                                  second_point.x - first_point.x)

        # 计算车辆当前航向与路径方向的夹角
        yaw_diff = path_direction - self.vehicle_yaw
        while yaw_diff > math.pi:
            yaw_diff -= 2 * math.pi
        while yaw_diff < -math.pi:
            yaw_diff += 2 * math.pi

        # 如果路径方向与车辆初始方向相差超过90度，则反转路径
        if abs(yaw_diff) > math.radians(90):
            rospy.loginfo("路径方向与车辆初始方向相反，反转路径")
            path_msg.poses = path_msg.poses[::-1]  # 反转路径点顺序

    def find_closest_point(self):
        """找到路径上最近的点"""
        if not self.current_path or not self.pose_received:
            return -1
        
        min_distance = float('inf')
        closest_index = 0
        
        for i, pose in enumerate(self.current_path.poses):
            dx = pose.pose.position.x - self.vehicle_x
            dy = pose.pose.position.y - self.vehicle_y
            distance = math.sqrt(dx*dx + dy*dy)
            
            if distance < min_distance:
                min_distance = distance
                closest_index = i
        
        return closest_index
    
    def find_lookahead_point(self):
        """使用Pure Pursuit算法找到前瞻点"""
        if not self.current_path or not self.pose_received:
            return None
        
        # 根据速度动态前瞻
        lookahead = self.update_dynamic_lookahead()
        
        # 首先找到最近的路径点作为起始点
        closest_index = self.find_closest_point()
        
        # 从最近点开始向前搜索前瞻点
        best_point = None
        best_distance = float('inf')
        best_index = -1
        
        # 搜索范围：从最近点开始向前搜索
        search_start = max(0, closest_index)
        search_end = min(len(self.current_path.poses), search_start + 20)  # 限制搜索范围
        
        for i in range(search_start, search_end):
            # 跳过已经访问过的点（只有当距离很近时才跳过）
            if i in self.visited_indices:
                pose = self.current_path.poses[i]
                dx = pose.pose.position.x - self.vehicle_x
                dy = pose.pose.position.y - self.vehicle_y
                distance = math.sqrt(dx*dx + dy*dy)
                if distance < self.visit_threshold * 2:  # 如果距离很近，跳过
                    continue
            
            pose = self.current_path.poses[i]
            dx = pose.pose.position.x - self.vehicle_x
            dy = pose.pose.position.y - self.vehicle_y
            distance = math.sqrt(dx*dx + dy*dy)
            
            # 寻找距离前瞻距离最近的点
            if distance >= lookahead * 0.8:  # 至少要有80%的前瞻距离
                distance_diff = abs(distance - lookahead)
                if distance_diff < best_distance:
                    best_distance = distance_diff
                    best_point = pose
                    best_index = i
        
        # 如果没有找到合适的前瞻点，选择未访问的最远点
        if best_point is None:
            for i in range(len(self.current_path.poses) - 1, -1, -1):
                if i not in self.visited_indices:
                    pose = self.current_path.poses[i]
                    dx = pose.pose.position.x - self.vehicle_x
                    dy = pose.pose.position.y - self.vehicle_y
                    distance = math.sqrt(dx*dx + dy*dy)
                    
                    if distance > self.visit_threshold:  # 确保点不在车辆附近
                        best_point = pose
                        best_index = i
                        break
        
        if best_point is not None:
            self.target_point_index = best_index
        
        return best_point
    
    def calculate_lateral_error(self, target_point):
        """计算横向偏差"""
        if not target_point:
            return 0.0
        
        # 计算车辆到目标点的向量
        dx = target_point.pose.position.x - self.vehicle_x
        dy = target_point.pose.position.y - self.vehicle_y
        
        # 计算横向偏差（垂直于车辆朝向的距离）
        lateral_error = -dx * math.sin(self.vehicle_yaw) + dy * math.cos(self.vehicle_yaw)
        
        return lateral_error
    
    def calculate_heading_error(self, target_point):
        """计算航向偏差"""
        if not target_point:
            return 0.0
        
        # 计算目标点的航向角
        dx = target_point.pose.position.x - self.vehicle_x
        dy = target_point.pose.position.y - self.vehicle_y
        target_yaw = math.atan2(dy, dx)
        
        # 计算航向偏差
        heading_error = target_yaw - self.vehicle_yaw
        
        # 将角度限制在[-π, π]范围内
        while heading_error > math.pi:
            heading_error -= 2 * math.pi
        while heading_error < -math.pi:
            heading_error += 2 * math.pi
        
        return heading_error
    
    def calculate_errors_from_xy(self, target_x, target_y):
        dx = target_x - self.vehicle_x
        dy = target_y - self.vehicle_y
        lateral_error = -dx * math.sin(self.vehicle_yaw) + dy * math.cos(self.vehicle_yaw)
        target_yaw = math.atan2(dy, dx)
        heading_error = target_yaw - self.vehicle_yaw
        while heading_error > math.pi:
            heading_error -= 2 * math.pi
        while heading_error < -math.pi:
            heading_error += 2 * math.pi
        return lateral_error, heading_error

    def filter_tracking_errors(self, lateral_error, heading_error):
        a = self.error_filter_alpha
        self.filtered_lateral_error = (1.0 - a) * self.filtered_lateral_error + a * lateral_error
        self.filtered_heading_error = (1.0 - a) * self.filtered_heading_error + a * heading_error
        return self.filtered_lateral_error, self.filtered_heading_error

    def pid_control(self, lateral_error, heading_error, dt):
        """直线赛道稳健型横向+航向PD。

        设计原则：
        1) 横向误差是主控制量；
        2) 航向误差降权，避免与“前瞻点方位角”重复放大；
        3) 接近中心线时进一步降低航向项，减少左右摆动；
        4) 微分项限幅，避免路径/定位更新造成瞬时大舵角；
        5) 保留原有转向限幅和单周期变化率限制；
        6) CAN协议完全不变。
        """
        if dt <= 0.0:
            dt = 0.05

        lateral_derivative = (lateral_error - self.prev_lateral_error) / dt
        lateral_derivative = max(
            -self.max_lateral_derivative,
            min(self.max_lateral_derivative, lateral_derivative)
        )
        lateral_control = (
            self.kp_lateral * lateral_error +
            self.kd_lateral * lateral_derivative
        )

        heading_derivative = (heading_error - self.prev_heading_error) / dt
        heading_derivative = max(
            -self.max_heading_derivative,
            min(self.max_heading_derivative, heading_derivative)
        )

        # 航向项自适应降权：
        # |e_y|很小时仅保留 heading_weight_min；
        # 偏差接近0.8m时逐步恢复到100%。
        lateral_mag = abs(lateral_error)
        if self.heading_weight_full_lateral_error > 1e-6:
            heading_ratio = lateral_mag / self.heading_weight_full_lateral_error
        else:
            heading_ratio = 1.0
        heading_ratio = max(0.0, min(1.0, heading_ratio))
        heading_weight = (
            self.heading_weight_min +
            (1.0 - self.heading_weight_min) * heading_ratio
        )

        heading_control = heading_weight * (
            self.kp_heading * heading_error +
            self.kd_heading * heading_derivative
        )

        raw_steering = (
            lateral_control + heading_control
        ) * self.steering_sign

        steering = max(
            -self.max_steering,
            min(self.max_steering, raw_steering)
        )

        # 限制单周期变化，防止“左/右打死”式快速反打。
        delta = steering - self.prev_steering_angle
        if delta > self.max_steering_step:
            steering = self.prev_steering_angle + self.max_steering_step
        elif delta < -self.max_steering_step:
            steering = self.prev_steering_angle - self.max_steering_step

        rospy.logdebug_throttle(
            1.0,
            "控制分解: e_y=%.3f, e_psi=%.2f°, w_psi=%.2f, lat=%.2f°, head=%.2f°, raw=%.2f°, out=%.2f°",
            lateral_error,
            math.degrees(heading_error),
            heading_weight,
            math.degrees(lateral_control),
            math.degrees(heading_control),
            math.degrees(raw_steering),
            math.degrees(steering)
        )

        self.prev_lateral_error = lateral_error
        self.prev_heading_error = heading_error
        self.prev_steering_angle = steering
        return steering

    def publish_fssim_control(self, steering_angle):
        """发布控制指令到FSSIM仿真器"""
        if not self.enable_vehicle_output or not self.fssim_available or self.cmd_pub is None:
            return 0.0
        
        try:
            # 计算固定速度对应的油门值（简单线性映射）
            # 假设速度范围 0-5 m/s 对应油门 0-1
            throttle = min(1.0, max(0.0, self.fixed_speed / 5.0))
            
            cmd_msg = Cmd()
            cmd_msg.dc = throttle
            cmd_msg.delta = steering_angle
            self.cmd_pub.publish(cmd_msg)
            
            rospy.logdebug_throttle(5, "🎮 发布FSSIM控制命令: 油门=%.3f, 转向角=%.3f°", 
                                  throttle, math.degrees(steering_angle))
            
            return throttle
            
        except Exception as e:
            rospy.logerr("发布FSSIM控制命令时出错: %s", str(e))
            return 0.0
    
    def update_finish_distance(self):
        """Update signed progress once per accepted localization timestamp."""
        if self.finish_triggered:
            return
        if not self.pose_received:
            self.finish_progress.confirm_count = 0
            return
        had_origin = self.finish_progress.origin is not None
        self.finish_progress.update(self.pose_stamp, self.vehicle_x, self.vehicle_y)
        self.total_distance = self.finish_progress.progress
        self.finish_triggered = self.finish_progress.finished
        if not had_origin and self.finish_progress.origin is not None:
            rospy.loginfo('Finish reference locked: map=(%.3f, %.3f), axis=%.3frad, stamp=%.3f',
                          self.finish_progress.origin[0], self.finish_progress.origin[1],
                          self.finish_progress.axis_yaw, self.finish_progress.reference_stamp)

    def finish_status(self):
        progress = self.finish_progress
        origin = progress.origin or (None, None)
        return dict(finish_progress=progress.progress, finish_reference_mode=self.finish_start_mode,
                    finish_reference_x=origin[0], finish_reference_y=origin[1],
                    finish_axis_yaw=progress.axis_yaw, finish_pose_stamp=progress.last_stamp,
                    finish_confirm_count=progress.confirm_count,
                    finish_reference_stamp=progress.reference_stamp)

    def control_loop(self, event):
        with self.input_lock:
            self._control_loop(event)

    def _control_loop(self, event):
        """20Hz闭环：全局路径作基准，实时锥桶中心线作局部纠偏。"""
        self.check_position_sources()
        if not self.finish_triggered:
            self.update_finish_distance()
        if self.finish_triggered:
            self.hold_normal_stop()
            self.publish_input_status('normal_stop_latched')
            return
        if not self.pose_received or not self.path_is_fresh():
            reason = self.input_reason if not self.pose_received else 'path_or_observation_stale'
            self.publish_input_status(reason)
            rospy.logwarn_throttle(1.0, 'Tracking inhibited: %s; RES/VCU action required', reason)
            return
        self.publish_input_status('valid')
        self.update_speed_source()
        self.update_dynamic_lookahead()

        log_data = {
            'pose_age': self.pose_age,
            'input_reason': self.input_reason,
            'vehicle_x': self.vehicle_x,
            'vehicle_y': self.vehicle_y,
            'vehicle_yaw': self.vehicle_yaw,
            'vehicle_speed': self.vehicle_speed,
            'slam_received': self.slam_pose_received,
            'fssim_received': self.fssim_pose_received,
            'path_received': self.path_received,
            'controller_started': self.is_started,
            'total_distance': self.total_distance,
            'lookahead_distance': self.lookahead_distance,
            'visited_points': len(self.visited_indices),
            'total_path_points': len(self.current_path.poses) if self.current_path else 0,
            'target_point_index': self.target_point_index,
            'lateral_error': 0.0,
            'heading_error': 0.0,
            'steering_angle': 0.0,
            'throttle': 0.0,
            'lookahead_found': False,
            'udp_sent': False,
            'control_mode': self.control_mode,
            'local_centerline_active': False,
            'local_centerline_pairs': 0,
            'local_centerline_k': 0.0,
            'local_centerline_b': 0.0,
            'lookahead_mode': 'global'
        }
        log_data.update(self.finish_status())

        if not self.pose_received:
            rospy.logwarn_throttle(10, "⏳ 等待车辆定位数据...")
            self.logger.log_control_data(log_data)
            return
        if not self.path_received:
            rospy.logwarn_throttle(10, "⏳ 未接收到全局参考路径...")
            self.logger.log_control_data(log_data)
            return
        if not self.is_started:
            rospy.loginfo_throttle(5, "⏳ 等待启动条件：路径 + 定位")
            self.logger.log_control_data(log_data)
            return

        now = rospy.Time.now()
        dt = (now - self.prev_time).to_sec()
        self.prev_time = now
        if dt <= 0:
            return

        # 全局路径基准
        global_target = self.find_lookahead_point()
        if global_target is None:
            self.logger.log_control_data(log_data)
            return
        global_lateral, global_heading = self.calculate_errors_from_xy(
            global_target.pose.position.x, global_target.pose.position.y)
        log_data['lookahead_found'] = True

        # One authoritative target: follow the published path, including its
        # entry transition. Evidence quality gates freshness; it never switches
        # tracking to a second, unblended straight-line target.
        lateral_error, heading_error = global_lateral, global_heading
        line = centerline_from_status(self.planning_status, now.to_sec(),
                                      self.path_observation_timeout)
        if line is None:
            return
        log_data['lookahead_mode'] = 'planned_path_only'
        log_data['centerline_source'] = line['source']
        log_data['centerline_confidence'] = line['confidence']
        log_data['centerline_blend_weight'] = 0.0
        log_data['observation_age'] = now.to_sec() - line['observation_stamp']

        lateral_error, heading_error = self.filter_tracking_errors(
            lateral_error, heading_error)
        log_data['lateral_error'] = lateral_error
        log_data['heading_error'] = heading_error

        steering_angle = self.pid_control(lateral_error, heading_error, dt)

        # The independent finish branch already returned before tracking began.
        safe_steering_angle = steering_angle
        log_data['steering_angle'] = safe_steering_angle
        log_data['stop_cmd'] = 0

        throttle = 0.0
        if self.control_mode == "fssim":
            throttle = self.publish_fssim_control(safe_steering_angle)
            log_data['throttle'] = throttle

        if self.can_client_ok:
            log_data['udp_sent'] = self.can_client.send_steering_frame(
                safe_steering_angle,
                stop=False
            )

        self.logger.log_control_data(log_data)
        self.publish_steering_command(safe_steering_angle)

        source_text = '规划路径'
        pair_count = line['pairs']
        rospy.loginfo_throttle(
            1.0,
            "🚗 v=%.2fkm/h | source=%s | pairs=%d | e_y=%.3fm e_psi=%.2f° | Ld=%.2fm | δ=%.2f°" %
            (self.vehicle_speed * 3.6, source_text, pair_count,
             lateral_error, math.degrees(heading_error),
             self.lookahead_distance, math.degrees(safe_steering_angle)))

    def publish_steering_command(self, steering_angle):
        """发布转向控制指令 (弧度制)"""
        # 创建Float64消息，包含转向角（弧度）
        steering_msg = Float64()
        steering_msg.data = steering_angle
        
        # 发布转向指令
        if self.enable_vehicle_output and self.steering_pub is not None:
            self.steering_pub.publish(steering_msg)

    def shutdown(self):
        """关闭控制器"""
        rospy.loginfo("正在关闭路径跟踪控制器...")
        
        try:
            if self.can_client_ok:
                self.can_client.send_steering_frame(0.0, stop=True)
        except Exception as e:
            rospy.logerr("发送最后零转向时出错: %s", str(e))

        try:
            # 发送零控制指令到FSSIM（如果使用）
            if self.enable_vehicle_output and self.fssim_available and self.cmd_pub is not None:
                rospy.loginfo("发送最后的零控制指令到FSSIM...")
                cmd_msg = Cmd()
                cmd_msg.dc = 0.0
                cmd_msg.delta = 0.0
                self.cmd_pub.publish(cmd_msg)
        except Exception as e:
            rospy.logerr("发送零控制指令到FSSIM时出错: %s", str(e))
        
        try:
            # 关闭日志记录器
            if hasattr(self, 'logger'):
                self.logger.close()
        except Exception as e:
            rospy.logerr("关闭日志记录器时出错: %s", str(e))
        
        try:
            # 关闭CAN UDP客户端
            if hasattr(self, 'can_client'):
                self.can_client.close()
        except Exception as e:
            rospy.logerr("关闭CAN UDP客户端时出错: %s", str(e))
        
        try:
            # 停止ROS定时器
            if hasattr(self, 'control_timer'):
                self.control_timer.shutdown()
            if hasattr(self, 'source_check_timer'):
                self.source_check_timer.shutdown()
        except Exception as e:
            rospy.logerr("停止定时器时出错: %s", str(e))
        
        rospy.loginfo("路径跟踪控制器已关闭")

def main():
    """主函数，确保正确清理资源"""
    controller = None
    try:
        controller = PathTrackingController()
        rospy.loginfo("🚀 路径跟踪控制器启动成功 - 智能定位源选择")
        rospy.loginfo("📍 定位策略: 使用配置的固定来源，每个控制周期读取最新快照")
        rospy.loginfo("📊 日志记录功能已启用")
        if controller.can_client_ok:
            rospy.loginfo("🔌 CAN UDP通信功能已启用")
        if controller.fssim_available:
            rospy.loginfo("🎮 FSSIM仿真器支持已启用")
        
        # 注册关闭钩子
        rospy.on_shutdown(controller.shutdown)
        
        rospy.spin()
        
    except rospy.ROSInterruptException:
        rospy.loginfo("接收到ROS中断信号")
    except KeyboardInterrupt:
        rospy.loginfo("用户中断程序")
    except Exception as e:
        rospy.logerr("控制器运行错误: %s", str(e))
        import traceback
        rospy.logerr("详细错误: %s", traceback.format_exc())
    finally:
        # 确保在finally块中清理资源
        if controller:
            try:
                controller.shutdown()
            except Exception as e:
                rospy.logerr("在finally块中关闭控制器时出错: %s", str(e))

if __name__ == '__main__':
    main()
