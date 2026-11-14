#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
FSAE Formula Student Autonomous Vehicle
雷达版建图模块：订阅 /clustered_points (PoseArray)，发布 /cone_map (ConeArray)

适配：
  - 雷达感知：clustering.py
  - 直线加速规划：zhixian.py

坐标系：
  雷达坐标系（ROS VLP-16）：X 前，Y 左，Z 上
  车辆坐标系：X 前，Y 右，Z 上
  世界坐标系：X 前（初始车头方向），Y 右，原点在初始化位置
"""

from __future__ import print_function
import rospy
import numpy as np
import math
import cv2
from sensor_msgs.msg import NavSatFix, Imu
from geometry_msgs.msg import PoseStamped, PoseArray
from std_msgs.msg import Header, String
import threading
import sys
import time
import json
from gps_integrity import GpsIntegrityGuard
from heading_integrity import HeadingIntegrityGuard


def visualization_wait_ms(started, now, frequency):
    """Budget GUI refresh only; ROS callback rates remain independent."""
    period = 1.0 / frequency
    elapsed = max(0.0, now-started)
    return max(1, int(max(0.0, period-elapsed)*1000.0))

if sys.version_info[0] < 3:
    reload(sys)
    sys.setdefaultencoding('utf-8')

from mapping.msg import ConeArray, Cone


class CoordinateTransformer(object):
    """GPS 坐标转换"""

    @staticmethod
    def gps_to_cartesian(lat, lon, origin_lat, origin_lon):
        R = 6371000
        lat_rad = math.radians(lat)
        lon_rad = math.radians(lon)
        origin_lat_rad = math.radians(origin_lat)
        origin_lon_rad = math.radians(origin_lon)
        x = R * (lon_rad - origin_lon_rad) * math.cos(origin_lat_rad)
        y = R * (lat_rad - origin_lat_rad)
        return x, y

    @staticmethod
    def geographic_to_vehicle_frame(x_geo, y_geo, origin_yaw_deg):
        """地理东北坐标 -> 车辆初始坐标系（X 前，Y 右）"""
        yaw_rad = math.radians(origin_yaw_deg)
        x_vehicle = y_geo * math.cos(yaw_rad) + x_geo * math.sin(yaw_rad)
        y_vehicle = -y_geo * math.sin(yaw_rad) + x_geo * math.cos(yaw_rad)
        return x_vehicle, y_vehicle


class ConeObject(object):
    """锥筒对象"""

    def __init__(self, x, y, timestamp):
        self.x = x
        self.y = y
        self.observations = [(x, y, timestamp)]
        self.last_update = timestamp
        self.confidence = 0.2
        self.std_dev_x = 0.0
        self.std_dev_y = 0.0

    def update(self, x, y, timestamp):
        distance = math.sqrt((x - self.x) ** 2 + (y - self.y) ** 2)

        if distance > 2.0:
            outlier_weight = 0.1
            self.x = self.x * (1 - outlier_weight) + x * outlier_weight
            self.y = self.y * (1 - outlier_weight) + y * outlier_weight
        else:
            self.observations.append((x, y, timestamp))
            max_observations = 10
            if len(self.observations) > max_observations:
                self.observations = self.observations[-max_observations:]

            weights = np.exp(np.linspace(-1, 0, len(self.observations)))
            weights = weights / weights.sum()
            x_values = [obs[0] for obs in self.observations]
            y_values = [obs[1] for obs in self.observations]
            self.x = np.average(x_values, weights=weights)
            self.y = np.average(y_values, weights=weights)

            if len(self.observations) > 2:
                self.std_dev_x = np.std(x_values)
                self.std_dev_y = np.std(y_values)

        self.last_update = timestamp
        self.confidence = min(1.0, len(self.observations) / 5.0)


class FSAESLAMNode(object):
    """雷达版建图节点"""

    def __init__(self):
        rospy.init_node('fsae_slam_node', anonymous=False)

        # ---------- 初始化参数 ----------
        self.imu_init_wait_time = rospy.get_param('~imu_init_wait_time', 3.0)
        self.gps_lat_min = rospy.get_param('~gps_lat_min', 30.0)
        self.gps_lat_max = rospy.get_param('~gps_lat_max', 40.0)
        self.gps_lon_min = rospy.get_param('~gps_lon_min', 110.0)
        self.gps_lon_max = rospy.get_param('~gps_lon_max', 125.0)
        self.gps_guard = GpsIntegrityGuard(
            max_speed_mps=rospy.get_param('~gps_max_speed_mps', 8.0),
            jump_tolerance_m=rospy.get_param('~gps_jump_tolerance_m', 2.0),
            max_gap_s=rospy.get_param('~gps_max_gap_s', 1.0),
            max_age_s=rospy.get_param('~gps_max_age_s', 0.5),
            bootstrap_samples=rospy.get_param('~gps_bootstrap_samples', 3),
            lat_min=self.gps_lat_min, lat_max=self.gps_lat_max,
            lon_min=self.gps_lon_min, lon_max=self.gps_lon_max)
        self.last_gps_stamp = rospy.Time(0)
        self.imu_integrity = HeadingIntegrityGuard(
            max_rate_dps=rospy.get_param('~imu_max_yaw_rate_dps', 180.0),
            jump_tolerance_deg=rospy.get_param('~imu_jump_tolerance_deg', 5.0))

        # ---------- 雷达外参 ----------
        # 雷达在车辆坐标系中的安装位置（X 前，Y 右，Z 上）
        self.lidar_offset_x = rospy.get_param('~lidar_offset_x', 1.76618)
        self.lidar_offset_y = rospy.get_param('~lidar_offset_y', -0.11304)
        self.lidar_offset_z = rospy.get_param('~lidar_offset_z', 0.42850)

        # 雷达 Y 轴方向：VLP-16 在 ROS 中 Y 向左，车辆 Y 向右 → 取反
        # 如果雷达 Y 向右，设为 1.0
        self.lidar_y_sign = rospy.get_param('~lidar_y_sign', -1.0)

        # 雷达安装偏航角（度），如果雷达 X 轴与车辆 X 轴不重合
        self.lidar_yaw_offset_deg = rospy.get_param('~lidar_yaw_offset_deg', 0.5)

        # ---------- 建图参数 ----------
        self.cone_merge_distance = rospy.get_param('~cone_merge_distance', 0.8)
        self.cone_timeout = rospy.get_param('~cone_timeout', 3.0)
        # 车辆本体的雷达回波不能作为赛道锥桶。范围为车辆坐标系，单位米。
        self.self_filter_rear = rospy.get_param('~self_filter_rear', 1.2)
        self.self_filter_front = rospy.get_param('~self_filter_front', 2.7)
        self.self_filter_half_width = rospy.get_param('~self_filter_half_width', 0.9)

        # ---------- 可视化 ----------
        self.enable_visualization = rospy.get_param('~enable_visualization', True)
        self.visualization_hz = float(rospy.get_param('~visualization_hz', 5.0))
        if not np.isfinite(self.visualization_hz) or self.visualization_hz <= 0:
            raise ValueError('visualization_hz must be finite and positive')
        self.vis_map_size = rospy.get_param('~vis_map_size', 60)
        self.window_width = 1000
        self.window_height = 1000
        self.pixels_per_meter = self.window_width / (2.0 * self.vis_map_size)
        self.camera_center_x = 0.0
        self.camera_center_y = 0.0
        self.camera_smoothing = 0.15

        # ---------- 状态 ----------
        self.state_lock = threading.Lock()
        self._is_initialized = False
        self.init_start_time = None
        self.origin_lat = None
        self.origin_lon = None
        self.origin_yaw = None
        self.current_lat = None
        self.current_lon = None
        self.current_yaw = None
        self.current_x = 0.0
        self.current_y = 0.0

        self.cone_map = []
        self.last_cone_observation_stamp = rospy.Time(0)
        self.cone_lock = threading.Lock()
        self.transformer = CoordinateTransformer()

        # ---------- 统计 ----------
        self.gps_msg_count = 0
        self.imu_msg_count = 0
        self.cone_detection_count = 0
        self.cone_merge_count = 0
        self.cone_new_count = 0
        self.last_lidar_total = 0
        self.last_lidar_filtered = 0
        self.last_lidar_accepted = 0
        self.last_lidar_points = []

        # ---------- 可视化控制 ----------
        self.vis_shutdown = threading.Event()

        # ---------- 订阅/发布 ----------
        self.gps_sub = rospy.Subscriber('/GPS_data', NavSatFix, self.gps_callback, queue_size=10)
        self.imu_sub = rospy.Subscriber('/imu_data', Imu, self.imu_callback, queue_size=50)

        # 雷达感知输出：PoseArray，每个 pose 是一个锥筒
        self.cone_sub = rospy.Subscriber('/clustered_points', PoseArray, self.cone_callback, queue_size=50)

        # 发布接口保持不变，规划器 zhixian.py 无需改动
        self.vehicle_pose_pub = rospy.Publisher('/vehicle_pose', PoseStamped, queue_size=10)
        self.cone_map_pub = rospy.Publisher('/cone_map', ConeArray, queue_size=10)
        self.gps_status_pub = rospy.Publisher('/localization/gps_status', String, queue_size=1)
        self.imu_status_pub = rospy.Publisher('/localization/imu_status', String, queue_size=1)
        self.cone_timer = rospy.Timer(rospy.Duration(0.2), self.cone_maintenance_callback)
        self.gps_status_timer = rospy.Timer(rospy.Duration(1.0), self.publish_gps_status)

        print("\n" + "=" * 60)
        rospy.loginfo("[FSAE SLAM-LiDAR] Node started")
        rospy.loginfo("  雷达坐标系: X 前, Y 左, Z 上 (ROS VLP-16)")
        rospy.loginfo("  车辆坐标系: X 前, Y 右, Z 上")
        rospy.loginfo("  世界坐标系: X 前, Y 右, 原点=初始化位置")
        rospy.loginfo("[雷达外参] X:%.2f Y:%.2f Z:%.2f y_sign:%.1f yaw_offset:%.1f°",
                      self.lidar_offset_x, self.lidar_offset_y, self.lidar_offset_z,
                      self.lidar_y_sign, self.lidar_yaw_offset_deg)
        rospy.loginfo("[建图参数] 合并距离:%.2fm", self.cone_merge_distance)
        rospy.loginfo("[车身回波过滤] X:[-%.2f, %.2f]m, |Y|<=%.2fm",
                      self.self_filter_rear, self.self_filter_front, self.self_filter_half_width)
        print("=" * 60)
        rospy.loginfo("[Init] Waiting for IMU stabilization: %.1f seconds", self.imu_init_wait_time)
        print("=" * 60 + "\n")

    # ============================================================
    # 状态快照
    # ============================================================
    def get_state_snapshot(self):
        with self.state_lock:
            return {
                'is_initialized': self._is_initialized,
                'current_lat': self.current_lat,
                'current_lon': self.current_lon,
                'current_yaw': self.current_yaw,
                'current_x': self.current_x,
                'current_y': self.current_y,
                'gps_stamp': self.last_gps_stamp,
                'gps_fresh': self.gps_guard.is_fresh(rospy.Time.now().to_sec()),
                'imu_fresh': self.imu_integrity.is_fresh(rospy.Time.now().to_sec()),
                'origin_lat': self.origin_lat,
                'origin_lon': self.origin_lon,
                'origin_yaw': self.origin_yaw,
                'init_start_time': self.init_start_time,
                'gps_msg_count': self.gps_msg_count,
                'imu_msg_count': self.imu_msg_count
            }

    # ============================================================
    # GPS 回调
    # ============================================================
    def gps_callback(self, msg):
        with self.state_lock:
            self.gps_msg_count += 1

            if self.gps_msg_count == 1:
                rospy.loginfo("[GPS] First data received!")

            if not self.gps_guard.accept(msg.latitude, msg.longitude,
                                         msg.header.stamp.to_sec(), rospy.Time.now().to_sec()):
                if not self._is_initialized:
                    self.init_start_time = None
                if self.gps_guard.last_reason != 'warming_up':
                    rospy.logwarn_throttle(2.0, '[GPS] Rejected: %s; distance=%s limit=%s',
                                           self.gps_guard.last_reason,
                                           self.gps_guard.last_distance_m,
                                           self.gps_guard.last_limit_m)
                return
            self.current_lat, self.current_lon = msg.latitude, msg.longitude
            self.last_gps_stamp = msg.header.stamp

            if not self._is_initialized:
                self.check_initialization()
                return

            x_geo, y_geo = self.transformer.gps_to_cartesian(
                self.current_lat, self.current_lon,
                self.origin_lat, self.origin_lon
            )
            self.current_x, self.current_y = self.transformer.geographic_to_vehicle_frame(
                x_geo, y_geo, self.origin_yaw
            )

        self.publish_vehicle_state()

    def publish_gps_status(self, _event):
        with self.state_lock:
            status = self.gps_guard.snapshot(rospy.Time.now().to_sec())
            imu_status = self.imu_integrity.snapshot(rospy.Time.now().to_sec())
        self.gps_status_pub.publish(String(json.dumps(status, allow_nan=False, sort_keys=True)))
        self.imu_status_pub.publish(String(json.dumps(imu_status, allow_nan=False, sort_keys=True)))

    # ============================================================
    # IMU 回调
    # ============================================================
    def imu_callback(self, msg):
        with self.state_lock:
            self.imu_msg_count += 1

            if self.imu_msg_count == 1:
                rospy.loginfo("[IMU] First data received!")

            if not self.imu_integrity.accept(msg.orientation.z, msg.header.stamp.to_sec(),
                                             rospy.Time.now().to_sec()):
                if not self._is_initialized:
                    self.init_start_time = None
                if self.imu_integrity.last_reason != 'warming_up':
                    rospy.logwarn_throttle(2.0, '[IMU] Rejected heading discontinuity: %s',
                                           self.imu_integrity.last_reason)
                return
            self.current_yaw = msg.orientation.z

            if not self._is_initialized:
                self.check_initialization()

    # ============================================================
    # 雷达锥筒回调：处理 PoseArray
    # ============================================================
    def cone_callback(self, msg):
        """
        雷达感知输出的是 PoseArray，每个 pose 是一个锥筒的位置。
        位置在雷达坐标系下（/velodyne），无颜色。
        """
        self.cone_detection_count += 1
        detections = []
        filtered = 0
        for pose in msg.poses:
            lidar_x = pose.position.x
            lidar_y = pose.position.y
            lidar_z = pose.position.z
            if not np.all(np.isfinite((lidar_x, lidar_y, lidar_z))):
                filtered += 1
                continue
            vehicle_x, vehicle_y = self.transform_lidar_to_vehicle(lidar_x, lidar_y)
            if self.is_vehicle_return(vehicle_x, vehicle_y):
                filtered += 1
                continue
            detections.append((vehicle_x, vehicle_y))

        with self.cone_lock:
            self.last_lidar_total = len(msg.poses)
            self.last_lidar_filtered = filtered
            self.last_lidar_accepted = len(detections)
            self.last_lidar_points = detections

        state = self.get_state_snapshot()
        if not state['is_initialized']:
            if detections:
                rospy.logwarn_throttle(5.0, "[Map] LiDAR data received, waiting for GPS/IMU initialization")
            return
        if state['current_yaw'] is None:
            return
        if not state['gps_fresh'] or not state['imu_fresh']:
            rospy.logwarn_throttle(2.0, '[Map] GPS/IMU expired; skip world-frame cone updates')
            return

        # 使用接收时间管理超时；消息时间戳可能来自不同的时钟或录包。
        timestamp = rospy.Time.now().to_sec()
        with self.cone_lock:
            for vehicle_x, vehicle_y in detections:
                world_x, world_y = self.transform_vehicle_to_world(vehicle_x, vehicle_y, state)
                self._merge_cone_unlocked(world_x, world_y, timestamp)
            if detections:
                source_stamp = msg.header.stamp
                now_stamp = rospy.Time.now()
                # Zero-stamped legacy producers use receipt time; valid source
                # stamps retain age across the map, planner and controller.
                if source_stamp.to_sec() <= 0:
                    source_stamp = now_stamp
                if (source_stamp.to_sec() <= now_stamp.to_sec()+0.1 and
                        source_stamp.to_sec() > self.last_cone_observation_stamp.to_sec()):
                    self.last_cone_observation_stamp = source_stamp
            self._prune_cones_unlocked(timestamp)
        # 一帧雷达数据只发布一次完整地图，避免下游收到逐个增加的半帧结果。
        self.publish_cone_map()

    def is_vehicle_return(self, vehicle_x, vehicle_y):
        return (-self.self_filter_rear <= vehicle_x <= self.self_filter_front and
                abs(vehicle_y) <= self.self_filter_half_width)

    def cone_maintenance_callback(self, _event):
        if not self.get_state_snapshot()['is_initialized']:
            return
        with self.cone_lock:
            changed = self._prune_cones_unlocked(rospy.Time.now().to_sec())
        if changed:
            self.publish_cone_map()

    # ============================================================
    # GPS 有效性
    # ============================================================
    def is_gps_valid(self, lat, lon):
        if lat is None or lon is None:
            return False
        return (self.gps_lat_min <= lat <= self.gps_lat_max and
                self.gps_lon_min <= lon <= self.gps_lon_max)

    # ============================================================
    # 初始化
    # ============================================================
    def check_initialization(self):
        current_time = rospy.Time.now()
        # Start the existing wait only after BOTH streams are usable. During
        # device startup, heading may look stable while GPS is still zero.
        # Counting that period used to freeze the old heading at the first
        # valid GPS fix, just before the real heading passed its own gate.
        if (not self.gps_guard.is_fresh(current_time.to_sec()) or
                not self.imu_integrity.is_fresh(current_time.to_sec())):
            self.init_start_time = None
            return

        if (self.current_lat is None or self.current_lon is None or
                self.current_yaw is None):
            self.init_start_time = None
            return

        if not self.is_gps_valid(self.current_lat, self.current_lon):
            self.init_start_time = None
            return

        if self.init_start_time is None:
            self.init_start_time = current_time
            rospy.loginfo("[Init] GPS/IMU valid; starting continuous initialization wait...")
            return

        elapsed_time = (current_time - self.init_start_time).to_sec()
        if elapsed_time < 0:
            self.init_start_time = current_time
            return
        if elapsed_time < self.imu_init_wait_time:
            return

        self.origin_lat = self.current_lat
        self.origin_lon = self.current_lon
        self.origin_yaw = self.current_yaw
        self.current_x = 0.0
        self.current_y = 0.0
        self._is_initialized = True
        self.camera_center_x = 0.0
        self.camera_center_y = 0.0

        print("\n" + "*" * 60)
        rospy.loginfo("[Init] *** INITIALIZED SUCCESSFULLY ***")
        rospy.loginfo("[世界坐标系原点] GPS: (%.6f, %.6f)", self.origin_lat, self.origin_lon)
        rospy.loginfo("[世界坐标系X轴] 初始航向: %.1f度 (0=北, 90=东)", self.origin_yaw)
        print("*" * 60 + "\n")

    # ============================================================
    # 雷达坐标 -> 世界坐标
    # ============================================================
    def transform_lidar_to_world(self, lidar_x, lidar_y, lidar_z, state):
        """
        雷达坐标系（ROS VLP-16）：X 前，Y 左，Z 上
        车辆坐标系：X 前，Y 右，Z 上

        步骤：
        1. 雷达坐标 -> 车辆坐标（Y 取反 + 加安装偏移 + 可选偏航旋转）
        2. 车辆当前坐标 -> 世界坐标（按相对航向旋转 + 加车辆世界位置）
        """
        vehicle_x, vehicle_y = self.transform_lidar_to_vehicle(lidar_x, lidar_y)
        return self.transform_vehicle_to_world(vehicle_x, vehicle_y, state)

    def transform_lidar_to_vehicle(self, lidar_x, lidar_y):
        # ---------- 第一步：雷达 -> 车辆 ----------
        # Y 取反：雷达 Y 左，车辆 Y 右
        lx = lidar_x
        ly = self.lidar_y_sign * lidar_y

        # 如果雷达有安装偏航角，先旋转
        if abs(self.lidar_yaw_offset_deg) > 1e-6:
            yaw_off = math.radians(self.lidar_yaw_offset_deg)
            cos_off = math.cos(yaw_off)
            sin_off = math.sin(yaw_off)
            rx = lx * cos_off - ly * sin_off
            ry = lx * sin_off + ly * cos_off
            lx, ly = rx, ry

        # 加安装偏移
        vehicle_x = lx + self.lidar_offset_x
        vehicle_y = ly + self.lidar_offset_y
        return vehicle_x, vehicle_y

    def transform_vehicle_to_world(self, vehicle_x, vehicle_y, state):
        # ---------- 第二步：车辆当前 -> 世界 ----------
        yaw_relative = state['current_yaw'] - state['origin_yaw']
        while yaw_relative > 180:
            yaw_relative -= 360
        while yaw_relative < -180:
            yaw_relative += 360

        yaw_rad = math.radians(yaw_relative)
        cos_yaw = math.cos(yaw_rad)
        sin_yaw = math.sin(yaw_rad)

        world_x_rel = vehicle_x * cos_yaw - vehicle_y * sin_yaw
        world_y_rel = vehicle_x * sin_yaw + vehicle_y * cos_yaw

        world_x = state['current_x'] + world_x_rel
        world_y = state['current_y'] + world_y_rel

        return world_x, world_y

    # ============================================================
    # 建图更新：只用距离合并
    # ============================================================
    def _merge_cone_unlocked(self, x, y, timestamp):
        best_cone = None
        best_distance = float('inf')
        for cone in self.cone_map:
            d = math.hypot(cone.x - x, cone.y - y)
            if d < self.cone_merge_distance and d < best_distance:
                best_cone = cone
                best_distance = d

        if best_cone is not None:
            best_cone.update(x, y, timestamp)
            self.cone_merge_count += 1
        else:
            self.cone_map.append(ConeObject(x, y, timestamp))
            self.cone_new_count += 1

    def _prune_cones_unlocked(self, timestamp):
        before = len(self.cone_map)
        self.cone_map = [cone for cone in self.cone_map
                         if timestamp - cone.last_update < self.cone_timeout]
        return len(self.cone_map) != before

    # ============================================================
    # 发布车辆位姿
    # ============================================================
    def publish_vehicle_state(self):
        state = self.get_state_snapshot()
        if not state['is_initialized'] or not state['gps_fresh'] or not state['imu_fresh']:
            return

        pose_msg = PoseStamped()
        pose_msg.header = Header()
        pose_msg.header.stamp = state['gps_stamp']
        pose_msg.header.frame_id = "map"
        pose_msg.pose.position.x = state['current_x']
        pose_msg.pose.position.y = state['current_y']
        pose_msg.pose.position.z = 0.0

        if state['origin_yaw'] is not None and state['current_yaw'] is not None:
            yaw_relative = state['current_yaw'] - state['origin_yaw']
            while yaw_relative > 180:
                yaw_relative -= 360
            while yaw_relative < -180:
                yaw_relative += 360
            yaw_rad = math.radians(yaw_relative)
            pose_msg.pose.orientation.z = math.sin(yaw_rad / 2.0)
            pose_msg.pose.orientation.w = math.cos(yaw_rad / 2.0)

        self.vehicle_pose_pub.publish(pose_msg)

    # ============================================================
    # 发布锥筒地图（保持和 zhixian.py 兼容）
    # ============================================================
    def publish_cone_map(self):
        cone_array_msg = ConeArray()
        cone_array_msg.header = Header()
        cone_array_msg.header.frame_id = "map"

        with self.cone_lock:
            # A republished/maintained map must not make old detections fresh.
            cone_array_msg.header.stamp = self.last_cone_observation_stamp
            for cone_obj in self.cone_map:
                cone_msg = Cone()
                cone_msg.color = "unknown"
                cone_msg.x = cone_obj.x
                cone_msg.y = cone_obj.y
                cone_msg.confidence = cone_obj.confidence
                cone_array_msg.cones.append(cone_msg)

        self.cone_map_pub.publish(cone_array_msg)

    # ============================================================
    # 可视化（保留原有逻辑，加 unknown 颜色映射）
    # ============================================================
    def update_camera_center(self, vehicle_x, vehicle_y):
        self.camera_center_x += (vehicle_x - self.camera_center_x) * self.camera_smoothing
        self.camera_center_y += (vehicle_y - self.camera_center_y) * self.camera_smoothing

    def world_to_screen(self, world_x, world_y):
        relative_x = world_x - self.camera_center_x
        relative_y = world_y - self.camera_center_y
        screen_x = int(self.window_width / 2 + relative_y * self.pixels_per_meter)
        screen_y = int(self.window_height / 2 - relative_x * self.pixels_per_meter)
        return screen_x, screen_y

    def draw_init_screen(self):
        img = np.zeros((self.window_height, self.window_width, 3), dtype=np.uint8)
        img[:] = (40, 40, 40)

        state = self.get_state_snapshot()

        cv2.putText(img, "SYSTEM INITIALIZING (LiDAR)", (150, 100),
                    cv2.FONT_HERSHEY_DUPLEX, 1.2, (0, 255, 255), 2)

        y_pos = 200
        line_height = 40

        if state['init_start_time'] is None:
            cv2.putText(img, "Waiting for first data...", (200, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1)
        else:
            elapsed = (rospy.Time.now() - state['init_start_time']).to_sec()
            text = "Time: %.1f / %.1f seconds" % (elapsed, self.imu_init_wait_time)
            cv2.putText(img, text, (200, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
            y_pos += line_height

            gps_color = (0, 255, 0) if state['current_lat'] else (0, 0, 255)
            gps_text = "RECEIVED" if state['current_lat'] else "WAITING"
            cv2.putText(img, "GPS Data: " + gps_text, (200, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, gps_color, 1)
            y_pos += line_height

            imu_color = (0, 255, 0) if state['current_yaw'] is not None else (0, 0, 255)
            imu_text = "RECEIVED" if state['current_yaw'] is not None else "WAITING"
            cv2.putText(img, "IMU Data: " + imu_text, (200, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, imu_color, 1)
            y_pos += line_height

            if state['current_lat']:
                valid = self.is_gps_valid(state['current_lat'], state['current_lon'])
                valid_color = (0, 255, 0) if valid else (0, 0, 255)
                valid_text = "YES" if valid else "OUT OF RANGE"
                cv2.putText(img, "GPS Valid: " + valid_text, (200, y_pos),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, valid_color, 1)

        with self.cone_lock:
            lidar_total = self.last_lidar_total
            lidar_filtered = self.last_lidar_filtered
            lidar_points = list(self.last_lidar_points)
        cv2.putText(img, "LiDAR: %d poses, %d filtered" % (lidar_total, lidar_filtered),
                    (200, y_pos + line_height), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (255, 255, 255), 1)

        # 定位尚未初始化时，仍实时显示雷达局部坐标中的有效检测。
        cv2.rectangle(img, (580, 500), (980, 900), (100, 100, 100), 1)
        cv2.putText(img, "LiDAR local preview (waiting for pose)", (580, 480),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        center_x, center_y = 780, 700
        cv2.circle(img, (center_x, center_y), 6, (0, 0, 255), -1)
        cv2.arrowedLine(img, (center_x, center_y), (center_x, center_y - 28),
                        (0, 255, 255), 2)
        for vehicle_x, vehicle_y in lidar_points:
            px = int(center_x + vehicle_y * 8)
            py = int(center_y - vehicle_x * 8)
            if 585 <= px <= 975 and 505 <= py <= 895:
                cv2.circle(img, (px, py), 5, (0, 255, 0), -1)

        return img

    def draw_map_screen(self):
        img = np.zeros((self.window_height, self.window_width, 3), dtype=np.uint8)
        img[:] = (30, 30, 30)

        state = self.get_state_snapshot()
        self.update_camera_center(state['current_x'], state['current_y'])

        # 网格
        grid_spacing = 10
        grid_pixels = int(grid_spacing * self.pixels_per_meter)
        offset_x = int((self.camera_center_x % grid_spacing) * self.pixels_per_meter)
        offset_y = int((self.camera_center_y % grid_spacing) * self.pixels_per_meter)

        for i in range(-offset_x, self.window_width, grid_pixels):
            if 0 <= i < self.window_width:
                cv2.line(img, (i, 0), (i, self.window_height), (60, 60, 60), 1)
        for i in range(offset_y, self.window_height, grid_pixels):
            if 0 <= i < self.window_height:
                cv2.line(img, (0, i), (self.window_width, i), (60, 60, 60), 1)

        # 原点
        origin_x, origin_y = self.world_to_screen(0, 0)
        if -50 <= origin_x < self.window_width + 50 and -50 <= origin_y < self.window_height + 50:
            cv2.drawMarker(img, (origin_x, origin_y), (255, 255, 255),
                           cv2.MARKER_CROSS, 20, 2)
            cv2.putText(img, "ORIGIN", (origin_x + 10, origin_y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        with self.cone_lock:
            for cone in self.cone_map:
                cone_color = (0, 255, 0)
                cx, cy = self.world_to_screen(cone.x, cone.y)

                if -20 <= cx < self.window_width + 20 and -20 <= cy < self.window_height + 20:
                    radius = 6
                    if cone.confidence < 0.5:
                        faded_color = tuple(int(c * 0.5) for c in cone_color)
                        cv2.circle(img, (cx, cy), radius, faded_color, -1)
                        cv2.circle(img, (cx, cy), radius, (128, 128, 128), 1)
                        cv2.circle(img, (cx, cy), radius + 3, (0, 255, 255), 1)
                    else:
                        cv2.circle(img, (cx, cy), radius, cone_color, -1)
                        cv2.circle(img, (cx, cy), radius, (255, 255, 255), 1)

        # 车辆
        if state['origin_yaw'] is not None and state['current_yaw'] is not None:
            yaw_relative = state['current_yaw'] - state['origin_yaw']
            while yaw_relative > 180:
                yaw_relative -= 360
            while yaw_relative < -180:
                yaw_relative += 360
            yaw_rad = math.radians(yaw_relative)

            vehicle_length = 2.0
            vehicle_width = 1.0

            front_x = state['current_x'] + vehicle_length * math.cos(yaw_rad)
            front_y = state['current_y'] + vehicle_length * math.sin(yaw_rad)
            left_x = state['current_x'] + vehicle_width / 2 * math.cos(yaw_rad + math.pi / 2)
            left_y = state['current_y'] + vehicle_width / 2 * math.sin(yaw_rad + math.pi / 2)
            right_x = state['current_x'] + vehicle_width / 2 * math.cos(yaw_rad - math.pi / 2)
            right_y = state['current_y'] + vehicle_width / 2 * math.sin(yaw_rad - math.pi / 2)

            pts = np.array([
                self.world_to_screen(front_x, front_y),
                self.world_to_screen(left_x, left_y),
                self.world_to_screen(right_x, right_y)
            ], np.int32)

            cv2.fillPoly(img, [pts], (0, 0, 255))
            cv2.polylines(img, [pts], True, (255, 255, 255), 2)

            arrow_end_x = state['current_x'] + 4.0 * math.cos(yaw_rad)
            arrow_end_y = state['current_y'] + 4.0 * math.sin(yaw_rad)
            arrow_start = self.world_to_screen(state['current_x'], state['current_y'])
            arrow_end = self.world_to_screen(arrow_end_x, arrow_end_y)
            cv2.arrowedLine(img, arrow_start, arrow_end, (0, 255, 255), 2, tipLength=0.3)

        # 信息面板
        cv2.rectangle(img, (10, 10), (420, 230), (0, 100, 0), -1)
        cv2.rectangle(img, (10, 10), (420, 230), (0, 255, 0), 2)

        y_pos = 40
        cv2.putText(img, "FSAE SLAM - LiDAR", (20, y_pos),
                    cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 2)
        y_pos += 30

        cv2.putText(img, "Vehicle: (%.1f, %.1f)m" % (state['current_x'], state['current_y']),
                    (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        y_pos += 25

        yaw_val = state['current_yaw'] - state['origin_yaw'] if (state['current_yaw'] and state['origin_yaw']) else 0
        cv2.putText(img, "Yaw: %.1f deg" % yaw_val,
                    (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        y_pos += 25

        with self.cone_lock:
            cone_text = "Cones: %d" % len(self.cone_map)
            lidar_text = "LiDAR: %d poses, %d filtered, %d usable" % (
                self.last_lidar_total, self.last_lidar_filtered, self.last_lidar_accepted)
        cv2.putText(img, cone_text, (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        y_pos += 25

        cv2.putText(img, lidar_text, (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        y_pos += 25

        cv2.putText(img, "Merges: %d" % self.cone_merge_count,
                    (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        y_pos += 25

        cv2.putText(img, "New: %d" % self.cone_new_count,
                    (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 165, 0), 1)

        # 比例尺
        scale_length_m = 10
        scale_length_px = int(scale_length_m * self.pixels_per_meter)
        scale_x = self.window_width - 150
        scale_y = self.window_height - 30
        cv2.line(img, (scale_x, scale_y), (scale_x + scale_length_px, scale_y), (255, 255, 255), 2)
        cv2.putText(img, "10m", (scale_x + 30, scale_y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        return img

    def visualization_loop(self):
        cv2.namedWindow('FSAE SLAM (LiDAR)', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('FSAE SLAM (LiDAR)', self.window_width, self.window_height)

        rospy.loginfo("[Visualization] OpenCV window created")

        while not rospy.is_shutdown() and not self.vis_shutdown.is_set():
            try:
                frame_started = time.time()
                state = self.get_state_snapshot()
                if not state['is_initialized']:
                    img = self.draw_init_screen()
                else:
                    img = self.draw_map_screen()

                cv2.imshow('FSAE SLAM (LiDAR)', img)

                key = cv2.waitKey(visualization_wait_ms(
                    frame_started, time.time(), self.visualization_hz))
                if key == 27 or key == ord('q'):
                    rospy.loginfo("[Visualization] User requested shutdown")
                    break

            except Exception as e:
                rospy.logerr("[Visualization] Error: %s", str(e))
                import traceback
                traceback.print_exc()
                time.sleep(0.5)

        cv2.destroyAllWindows()
        rospy.loginfo("[Visualization] Closed")

    def run(self):
        if not self.enable_visualization:
            rospy.loginfo('[System] Headless mapping; callbacks remain active')
            rospy.spin()
            return
        rospy.loginfo("[System] Starting ROS callback thread...")
        ros_thread = threading.Thread(target=rospy.spin)
        ros_thread.daemon = True
        ros_thread.start()

        time.sleep(0.5)
        rospy.loginfo("[System] Starting OpenCV visualization...")

        try:
            self.visualization_loop()
        except KeyboardInterrupt:
            rospy.loginfo("[System] Keyboard interrupt")
        finally:
            self.vis_shutdown.set()
            rospy.signal_shutdown("Visualization closed")


def main():
    try:
        node = FSAESLAMNode()
        node.run()
    except rospy.ROSInterruptException:
        pass
    except KeyboardInterrupt:
        rospy.loginfo("[System] Shutting down...")
    except Exception as e:
        rospy.logerr("[Error] %s", str(e))
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
