#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
FSAE Formula Student Autonomous Vehicle
雷达版建图模块：订阅 /clustered_points (PoseArray)，发布 /cone_map (PoseArray), position.z = confidence

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
from geometry_msgs.msg import PoseStamped, PoseArray, Pose
from std_msgs.msg import Header
from collections import deque
from mapping.msg import ConeArray
import threading
import sys
import time

if sys.version_info[0] < 3:
    reload(sys)
    sys.setdefaultencoding('utf-8')



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


def interpolate_samples(samples, stamp, tolerance, angular=False):
    """Return values at acquisition time, with bounded nearest fallback."""
    if not samples:
        return None
    samples = sorted(samples, key=lambda sample: sample[0])
    for left, right in zip(samples, samples[1:]):
        if left[0] <= stamp <= right[0] and right[0] > left[0]:
            if max(stamp-left[0], right[0]-stamp) > tolerance:
                return None
            fraction = (stamp-left[0]) / (right[0]-left[0])
            delta = np.asarray(right[1:]) - np.asarray(left[1:])
            if angular:
                delta = (delta + 180.0) % 360.0 - 180.0
            return np.asarray(left[1:]) + fraction * delta
    nearest = min(samples, key=lambda sample: abs(sample[0]-stamp))
    return np.asarray(nearest[1:]) if abs(nearest[0]-stamp) <= tolerance else None


class ConeObject(object):
    def __init__(self, x, y, timestamp, confidence=1.0):
        self.x = x
        self.y = y
        self.observations = [(x, y, timestamp, confidence)]
        self.last_update = timestamp
        self.confidence = confidence / 3.0
        self.std_dev_x = 0.0
        self.std_dev_y = 0.0

    def update(self, x, y, timestamp, confidence=1.0):
        distance = math.sqrt((x - self.x) ** 2 + (y - self.y) ** 2)
        if distance > 2.0:
            w = 0.1
            self.x = self.x * (1 - w) + x * w
            self.y = self.y * (1 - w) + y * w
        else:
            self.observations.append((x, y, timestamp, confidence))
            max_obs = 10
            if len(self.observations) > max_obs:
                self.observations = self.observations[-max_obs:]
            weights = np.exp(np.linspace(-1, 0, len(self.observations))) * np.array([max(o[3], 1e-3) for o in self.observations])
            weights = weights / weights.sum()
            xs = [o[0] for o in self.observations]
            ys = [o[1] for o in self.observations]
            self.x = np.average(xs, weights=weights)
            self.y = np.average(ys, weights=weights)
            if len(self.observations) > 2:
                self.std_dev_x = float(np.std(xs))
                self.std_dev_y = float(np.std(ys))
        self.last_update = max(self.last_update, timestamp)
        self.confidence = float(np.mean([o[3] for o in self.observations])) * min(1.0, len(self.observations) / 3.0)


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

        # ---------- 雷达外参 ----------
        # 雷达在车辆坐标系中的安装位置（X 前，Y 右，Z 上）
        self.lidar_offset_x = rospy.get_param('~lidar_offset_x', 2.11552)
        self.lidar_offset_y = rospy.get_param('~lidar_offset_y', -0.02212)
        self.lidar_offset_z = rospy.get_param('~lidar_offset_z', 0.28870)

        # 雷达 Y 轴方向：VLP-16 在 ROS 中 Y 向左，车辆 Y 向右 → 取反
        # 如果雷达 Y 向右，设为 1.0
        self.lidar_y_sign = rospy.get_param('~lidar_y_sign', -1.0)

        # 雷达安装偏航角（度），如果雷达 X 轴与车辆 X 轴不重合
        self.lidar_yaw_offset_deg = rospy.get_param('~lidar_yaw_offset_deg', 0.5)

        # ---------- 建图参数 ----------
        self.cone_merge_distance = rospy.get_param('~cone_merge_distance', 0.8)
        self.cone_timeout = rospy.get_param('~cone_timeout', 3.0)
        self.confidence_decay_time = rospy.get_param('~confidence_decay_time', 1.5)



        # ---------- 可视化 ----------
        self.vis_map_size = rospy.get_param('~vis_map_size', 60)
        self.window_width = 1000
        self.window_height = 1000
        self.pixels_per_meter = self.window_width / (2.0 * self.vis_map_size)
        self.camera_center_x = 0.0
        self.camera_center_y = 0.0
        self.camera_smoothing = 0.15

        # ---------- 状态 ----------
        self.pose_time_tolerance = float(rospy.get_param('~pose_time_tolerance', 0.25))
        if (not all(np.isfinite(v) for v in (self.pose_time_tolerance, self.cone_timeout, self.confidence_decay_time)) or
                min(self.pose_time_tolerance, self.cone_timeout, self.confidence_decay_time) <= 0):
            raise ValueError('Invalid map timing parameters')
        self.gps_history = deque(maxlen=300)
        self.yaw_history = deque(maxlen=1000)
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
        self.cone_lock = threading.Lock()
        self.transformer = CoordinateTransformer()

        # ---------- 统计 ----------
        self.gps_msg_count = 0
        self.imu_msg_count = 0
        self.cone_detection_count = 0
        self.cone_merge_count = 0
        self.cone_new_count = 0

        # ---------- 可视化控制 ----------
        self.vis_shutdown = threading.Event()

        # Publishers exist before callbacks and periodic maintenance start.
        self.vehicle_pose_pub = rospy.Publisher('/vehicle_pose', PoseStamped, queue_size=10)
        self.cone_map_pub = rospy.Publisher('/cone_map', PoseArray, queue_size=1)
        self.cone_input_mode = rospy.get_param('~cone_input_mode', 'quality')
        if self.cone_input_mode == 'quality':
            self.cone_sub = rospy.Subscriber('/lidar/cones', ConeArray, self.quality_cone_callback, queue_size=1)
        elif self.cone_input_mode == 'legacy':
            self.cone_sub = rospy.Subscriber('/clustered_points', PoseArray, self.cone_callback, queue_size=1)
        else:
            raise ValueError('cone_input_mode must be quality or legacy')
        self.gps_sub = rospy.Subscriber('/GPS_data', NavSatFix, self.gps_callback, queue_size=10)
        self.imu_sub = rospy.Subscriber('/imu_data', Imu, self.imu_callback, queue_size=50)
        self.map_timer = rospy.Timer(rospy.Duration(0.1), self.map_timer_callback)

        print("\n" + "=" * 60)
        rospy.loginfo("[FSAE SLAM-LiDAR] Node started")
        rospy.loginfo("  雷达坐标系: X 前, Y 左, Z 上 (ROS VLP-16)")
        rospy.loginfo("  车辆坐标系: X 前, Y 右, Z 上")
        rospy.loginfo("  世界坐标系: X 前, Y 右, 原点=初始化位置")
        rospy.loginfo("[雷达外参] X:%.2f Y:%.2f Z:%.2f y_sign:%.1f yaw_offset:%.1f°",
                      self.lidar_offset_x, self.lidar_offset_y, self.lidar_offset_z,
                      self.lidar_y_sign, self.lidar_yaw_offset_deg)
        rospy.loginfo("[建图参数] 合并距离:%.2fm 锥桶超时:%.2fs 置信度衰减:%.2fs",
                      self.cone_merge_distance, self.cone_timeout, self.confidence_decay_time)
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
            self.current_lat = msg.latitude
            self.current_lon = msg.longitude

            if self.is_gps_valid(msg.latitude, msg.longitude):
                stamp = msg.header.stamp.to_sec() if not msg.header.stamp.is_zero() else rospy.Time.now().to_sec()
                self.gps_history.append((stamp, msg.latitude, msg.longitude))
            if self.gps_msg_count == 1:
                rospy.loginfo("[GPS] First data received!")

            if not self.is_gps_valid(self.current_lat, self.current_lon):
                return

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

    # ============================================================
    # IMU 回调
    # ============================================================
    def imu_callback(self, msg):
        with self.state_lock:
            self.imu_msg_count += 1
            if not np.isfinite(msg.orientation.z):
                return
            self.current_yaw = msg.orientation.z
            stamp = msg.header.stamp.to_sec() if not msg.header.stamp.is_zero() else rospy.Time.now().to_sec()
            self.yaw_history.append((stamp, self.current_yaw))

            if self.imu_msg_count == 1:
                rospy.loginfo("[IMU] First data received!")

            if not self._is_initialized:
                self.check_initialization()

    # ============================================================
    # 雷达锥筒回调：处理 PoseArray
    # ============================================================
    def quality_cone_callback(self, msg):
        observations = [(c.x, c.y, 0.0, c.confidence) for c in msg.cones]
        self.process_cones(msg.header, observations)

    def cone_callback(self, msg):
        observations = [(p.position.x, p.position.y, p.position.z, 1.0) for p in msg.poses]
        self.process_cones(msg.header, observations)

    def state_at_time(self, stamp):
        with self.state_lock:
            if not self._is_initialized:
                return None
            gps = interpolate_samples(list(self.gps_history), stamp, self.pose_time_tolerance)
            yaw = interpolate_samples(list(self.yaw_history), stamp, self.pose_time_tolerance, True)
            if gps is None or yaw is None:
                return None
            gx, gy = self.transformer.gps_to_cartesian(gps[0], gps[1], self.origin_lat, self.origin_lon)
            x, y = self.transformer.geographic_to_vehicle_frame(gx, gy, self.origin_yaw)
            return dict(current_x=x, current_y=y, current_yaw=float(yaw[0]), origin_yaw=self.origin_yaw)

    def process_cones(self, header, observations):
        if header.frame_id.lstrip('/') != 'velodyne':
            rospy.logwarn_throttle(1.0, 'Ignoring cones outside velodyne frame')
            return
        stamp = header.stamp.to_sec() if not header.stamp.is_zero() else rospy.Time.now().to_sec()
        age = rospy.Time.now().to_sec() - stamp
        if age < -self.pose_time_tolerance or age >= self.cone_timeout:
            rospy.logwarn_throttle(1.0, 'Ignoring stale or future lidar frame')
            return
        state = self.state_at_time(stamp)
        if state is None:
            rospy.logwarn_throttle(1.0, 'No time-matched GPS/IMU for lidar frame')
            return
        for x, y, z, confidence in observations:
            if not all(np.isfinite(v) for v in (x, y, z, confidence)) or not 0 < confidence <= 1:
                continue
            wx, wy = self.transform_lidar_to_world(x, y, z, state)
            self.update_cone_map_optimized(wx, wy, stamp, confidence)
        self.publish_cone_map()

    def map_timer_callback(self, event):
        now = rospy.Time.now().to_sec()
        with self.cone_lock:
            self.cone_map = [c for c in self.cone_map if 0 <= now-c.last_update < self.cone_timeout]
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

        if self.init_start_time is None:
            self.init_start_time = current_time
            rospy.loginfo("[Init] Starting initialization...")
            return

        elapsed_time = (current_time - self.init_start_time).to_sec()
        if elapsed_time < self.imu_init_wait_time:
            return

        if (self.current_lat is None or self.current_lon is None or
                self.current_yaw is None):
            return

        if not self.is_gps_valid(self.current_lat, self.current_lon):
            self.init_start_time = None
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

        # ---------- 第二步：车辆当前 -> 世界 ----------
        yaw_relative = state['current_yaw'] - state['origin_yaw']
        while yaw_relative > 180:
            yaw_relative -= 360
        while yaw_relative < -180:
            yaw_relative += 360

        yaw_rad = math.radians(-yaw_relative)
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
    def update_cone_map_optimized(self, x, y, timestamp, confidence=1.0):
        with self.cone_lock:
            best_cone = None
            best_distance = float('inf')
            for cone in self.cone_map:
                d = math.sqrt((cone.x - x) ** 2 + (cone.y - y) ** 2)
                if d < self.cone_merge_distance and d < best_distance:
                    best_cone = cone
                    best_distance = d

            if best_cone is not None:
                best_cone.update(x, y, timestamp, confidence)
                self.cone_merge_count += 1
            else:
                self.cone_map.append(ConeObject(x, y, timestamp, confidence))
                self.cone_new_count += 1

            self.cone_map = [c for c in self.cone_map
                            if (timestamp - c.last_update) < self.cone_timeout]


    # ============================================================
    # 发布车辆位姿
    # ============================================================
    def publish_vehicle_state(self):
        state = self.get_state_snapshot()

        pose_msg = PoseStamped()
        pose_msg.header = Header()
        pose_msg.header.stamp = rospy.Time.now()
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
            yaw_rad = math.radians(-yaw_relative)
            pose_msg.pose.orientation.z = math.sin(yaw_rad / 2.0)
            pose_msg.pose.orientation.w = math.cos(yaw_rad / 2.0)

        self.vehicle_pose_pub.publish(pose_msg)

    # ============================================================
    # 发布锥筒地图（保持和 zhixian.py 兼容）
    # ============================================================
    def publish_cone_map(self):
        out = PoseArray()
        out.header = Header()
        out.header.stamp = rospy.Time.now()
        out.header.frame_id = 'map'
        with self.cone_lock:
            now = rospy.Time.now().to_sec()
            for cone in self.cone_map:
                p = Pose()
                p.position.x = cone.x
                p.position.y = cone.y
                age = max(0.0, now - cone.last_update)
                freshness = math.exp(-age / max(self.confidence_decay_time, 1e-3))
                p.position.z = cone.confidence * freshness
                p.orientation.w = 1.0
                out.poses.append(p)
        self.cone_map_pub.publish(out)

    # ============================================================
    # 可视化：按置信度区分显示
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
                cx, cy = self.world_to_screen(cone.x, cone.y)
                if -20 <= cx < self.window_width + 20 and -20 <= cy < self.window_height + 20:
                    radius = 6
                    if cone.confidence >= 0.5:
                        cv2.circle(img, (cx, cy), radius, (200, 200, 200), -1)
                        cv2.circle(img, (cx, cy), radius, (255, 255, 255), 1)
                    else:
                        cv2.circle(img, (cx, cy), radius, (120, 120, 120), -1)
                        cv2.circle(img, (cx, cy), radius, (128, 128, 128), 1)
                        cv2.circle(img, (cx, cy), radius + 3, (0, 255, 255), 1)

        # 车辆
        if state['origin_yaw'] is not None and state['current_yaw'] is not None:
            yaw_relative = state['current_yaw'] - state['origin_yaw']
            while yaw_relative > 180:
                yaw_relative -= 360
            while yaw_relative < -180:
                yaw_relative += 360
            yaw_rad = math.radians(-yaw_relative)

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
        cv2.putText(img, cone_text, (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
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
                state = self.get_state_snapshot()
                if not state['is_initialized']:
                    img = self.draw_init_screen()
                else:
                    img = self.draw_map_screen()

                cv2.imshow('FSAE SLAM (LiDAR)', img)

                key = cv2.waitKey(50)
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
