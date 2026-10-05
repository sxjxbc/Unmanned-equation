#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
FSAE Formula Student Autonomous Vehicle
雷达版建图模块：默认订阅 /lidar/cones (ConeArray)，可选旧 PoseArray 输入，发布 /cone_map (ConeArray)

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
import threading
import sys
import time
from data_quality import PoseBuffer, fresh, gps_quality_valid, StableOrigin

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
    """Confidence combines independent frame support and current detection quality."""
    def __init__(self, x, y, timestamp, confidence=1.0):
        self.x, self.y = x, y
        self.observations = [(x, y, timestamp)]
        self.last_update = timestamp
        self.detection_confidence = confidence
        self.confidence = confidence * 0.4
        self.std_dev_x = self.std_dev_y = 0.0

    def update(self, x, y, timestamp, confidence=1.0):
        if timestamp <= self.last_update:
            return False
        self.observations = (self.observations + [(x, y, timestamp)])[-10:]
        weights = np.exp(np.linspace(-1, 0, len(self.observations)))
        weights /= weights.sum()
        xs, ys = [o[0] for o in self.observations], [o[1] for o in self.observations]
        self.x, self.y = np.average(xs, weights=weights), np.average(ys, weights=weights)
        self.std_dev_x, self.std_dev_y = np.std(xs), np.std(ys)
        self.last_update = timestamp
        self.detection_confidence = confidence
        self.confidence = confidence * min(1.0, 0.4 + 0.15 * (len(self.observations) - 1))
        return True


class FSAESLAMNode(object):
    """雷达版建图节点"""

    def __init__(self):
        rospy.init_node('fsae_slam_node', anonymous=False)

        # ---------- 初始化参数 ----------
        self.imu_init_wait_time = rospy.get_param('~imu_init_wait_time', 3.0)
        self.gps_max_horizontal_variance = rospy.get_param('~gps_max_horizontal_variance', 0.25)
        self.gps_allow_unknown_covariance = rospy.get_param('~gps_allow_unknown_covariance', False)
        if not np.isfinite(self.gps_max_horizontal_variance) or self.gps_max_horizontal_variance <= 0:
            raise ValueError('GPS variance limit must be positive and finite')
        self.initializer = StableOrigin(self.imu_init_wait_time,
            rospy.get_param('~init_min_samples', 10),
            rospy.get_param('~init_max_gap', 0.5),
            rospy.get_param('~init_position_tolerance', 0.30),
            math.radians(rospy.get_param('~init_yaw_tolerance_deg', 3.0)))
        self.gps_lat_min = rospy.get_param('~gps_lat_min', 30.0)
        self.gps_lat_max = rospy.get_param('~gps_lat_max', 40.0)
        self.gps_lon_min = rospy.get_param('~gps_lon_min', 110.0)
        self.gps_lon_max = rospy.get_param('~gps_lon_max', 125.0)

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

        # ---------- 可视化 ----------
        self.enable_visualization = rospy.get_param('~enable_visualization', True)
        self.vis_map_size = rospy.get_param('~vis_map_size', 60)
        self.window_width = 1000
        self.window_height = 1000
        self.pixels_per_meter = self.window_width / (2.0 * self.vis_map_size)
        self.camera_center_x = 0.0
        self.camera_center_y = 0.0
        self.camera_smoothing = 0.15

        # ---------- 状态 ----------
        self.state_lock = threading.Lock()
        self.pose_history = PoseBuffer(rospy.get_param('~pose_max_speed', 12.0))
        self.imu_history = PoseBuffer()
        self.alignment_tolerance = rospy.get_param('~alignment_tolerance', 0.15)
        self.sensor_timeout = rospy.get_param('~sensor_timeout', 0.5)
        self.last_gps_stamp = None
        self.last_imu_stamp = None
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
        self.cone_lock = threading.RLock()
        self.transformer = CoordinateTransformer()

        # ---------- 统计 ----------
        self.gps_msg_count = 0
        self.imu_msg_count = 0
        self.cone_detection_count = 0
        self.cone_merge_count = 0
        self.cone_new_count = 0

        # ---------- 可视化控制 ----------
        self.vis_shutdown = threading.Event()

        # Publishers must exist before a subscriber can invoke a callback.
        self.vehicle_pose_pub = rospy.Publisher('/vehicle_pose', PoseStamped, queue_size=10)
        self.cone_map_pub = rospy.Publisher('/cone_map', ConeArray, queue_size=10)
        # ---------- 订阅/发布 ----------
        self.gps_sub = rospy.Subscriber('/GPS_data', NavSatFix, self.gps_callback, queue_size=10)
        self.imu_sub = rospy.Subscriber('/imu_data', Imu, self.imu_callback, queue_size=50)

        # Select exactly one lidar contract; never mix duplicate frame sources.
        self.cone_input_mode = rospy.get_param('~cone_input_mode', 'quality')
        if self.cone_input_mode == 'quality':
            self.cone_sub = rospy.Subscriber('/lidar/cones', ConeArray, self.quality_cone_callback, queue_size=1)
        elif self.cone_input_mode == 'legacy':
            self.cone_sub = rospy.Subscriber('/clustered_points', PoseArray, self.cone_callback, queue_size=1)
        else:
            raise ValueError('cone_input_mode must be quality or legacy')


        print("\n" + "=" * 60)
        rospy.loginfo("[FSAE SLAM-LiDAR] Node started")
        rospy.loginfo("  雷达坐标系: X 前, Y 左, Z 上 (ROS VLP-16)")
        rospy.loginfo("  车辆坐标系: X 前, Y 右, Z 上")
        rospy.loginfo("  世界坐标系: X 前, Y 右, 原点=初始化位置")
        rospy.loginfo("[雷达外参] X:%.2f Y:%.2f Z:%.2f y_sign:%.1f yaw_offset:%.1f°",
                      self.lidar_offset_x, self.lidar_offset_y, self.lidar_offset_z,
                      self.lidar_y_sign, self.lidar_yaw_offset_deg)
        rospy.loginfo("[建图参数] 合并距离:%.2fm", self.cone_merge_distance)
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
            stamp = msg.header.stamp.to_sec()
            now = rospy.Time.now().to_sec()
            if self.last_gps_stamp is not None and stamp <= self.last_gps_stamp:
                return
            if (not fresh(stamp, now, self.sensor_timeout) or
                    not self.is_gps_valid(msg.latitude, msg.longitude) or
                    not gps_quality_valid(msg.status.status, msg.position_covariance_type,
                        msg.position_covariance, self.gps_max_horizontal_variance,
                        self.gps_allow_unknown_covariance)):
                self.initializer.reset()
                rospy.logwarn_throttle(1.0, 'GPS rejected: stale, no fix or invalid covariance/coordinates')
                return
            imu = self.imu_history.at(stamp, self.alignment_tolerance)
            if imu is None or not fresh(self.last_imu_stamp, now, self.sensor_timeout):
                self.initializer.reset()
                return
            self.last_gps_stamp = stamp
            self.gps_msg_count += 1
            self.current_lat, self.current_lon = msg.latitude, msg.longitude
            if not self._is_initialized:
                origin = self.initializer.add(stamp, msg.latitude, msg.longitude, imu['yaw'])
                if origin is None:
                    return
                self.check_initialization(origin)
            x_geo, y_geo = self.transformer.gps_to_cartesian(
                msg.latitude, msg.longitude, self.origin_lat, self.origin_lon)
            x, y = self.transformer.geographic_to_vehicle_frame(x_geo, y_geo, self.origin_yaw)
            yaw_relative = imu['yaw'] - math.radians(self.origin_yaw)
            # Commit state only after the actual aligned pose passes jump checks.
            if not self.pose_history.add(stamp, x, y, yaw_relative):
                return
            self.current_x, self.current_y = x, y
        self.publish_vehicle_state()

    # ============================================================
    # IMU 回调
    # ============================================================
    def imu_callback(self, msg):
        with self.state_lock:
            stamp = msg.header.stamp.to_sec()
            heading = msg.orientation.z  # INS driver encodes degrees here (not a quaternion).
            if not fresh(stamp, rospy.Time.now().to_sec(), self.sensor_timeout):
                return
            if not self.imu_history.add(stamp, 0.0, 0.0, math.radians(heading)):
                return
            self.last_imu_stamp = stamp
            self.imu_msg_count += 1
            self.current_yaw = msg.orientation.z

            if self.imu_msg_count == 1:
                rospy.loginfo("[IMU] First data received!")


    # ============================================================
    # 雷达锥筒回调：处理 PoseArray
    # ============================================================
    def quality_cone_callback(self, msg):
        if msg.header.frame_id.lstrip('/') != 'velodyne':
            rospy.logwarn_throttle(1.0, 'Lidar cones rejected: unexpected frame')
            return
        converted = PoseArray()
        converted.header = msg.header
        confidences = []
        for cone in msg.cones:
            if (not all(np.isfinite(v) for v in (cone.x, cone.y, cone.confidence)) or
                    not 0.0 < cone.confidence <= 1.0):
                continue
            item = Pose()
            item.position.x, item.position.y = cone.x, cone.y
            converted.poses.append(item)
            confidences.append(float(cone.confidence))
        # Invalid nonempty messages are not evidence of an empty scene.
        if msg.cones and not converted.poses:
            return
        self.cone_callback(converted, confidences)

    def cone_callback(self, msg, confidences=None):
        """
        雷达感知输出的是 PoseArray，每个 pose 是一个锥筒的位置。
        位置在雷达坐标系下（/velodyne），无颜色。
        """
        self.cone_detection_count += 1

        timestamp = msg.header.stamp.to_sec()
        state = self.get_state_snapshot()
        if not state['is_initialized'] or not fresh(timestamp, rospy.Time.now().to_sec(), self.sensor_timeout):
            return
        pose = self.pose_history.at(timestamp, self.alignment_tolerance)
        if pose is None:
            rospy.logwarn_throttle(1.0, 'Lidar frame rejected: no time-aligned pose')
            return
        state['current_x'], state['current_y'] = pose['x'], pose['y']
        state['current_yaw'] = state['origin_yaw'] + math.degrees(pose['yaw'])
        with self.cone_lock:
            if hasattr(self, 'last_cone_stamp') and timestamp <= self.last_cone_stamp:
                return
            self.last_cone_stamp = timestamp
            detections = []
            for index, item in enumerate(msg.poses):
                if not all(not math.isnan(v) and not math.isinf(v) for v in
                           (item.position.x, item.position.y, item.position.z)):
                    continue
                x, y = self.transform_lidar_to_world(
                    item.position.x, item.position.y, item.position.z, state)
                confidence = 1.0 if confidences is None else confidences[index]
                detections.append((x, y, confidence))
            if detections:
                self.update_cone_frame(detections, timestamp)
            # Empty frames clear observations; never refresh old cones as new evidence.
            if not msg.poses:
                self.cone_map = []
            self.cone_map = [c for c in self.cone_map if timestamp - c.last_update < self.cone_timeout]
        self.publish_cone_map(timestamp)

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
    def check_initialization(self, origin):
        self.origin_lat, self.origin_lon = origin[:2]
        self.origin_yaw = math.degrees(origin[2])
        self._is_initialized = True
        self.camera_center_x = self.camera_center_y = 0.0
        rospy.loginfo('[Init] Stable origin: GPS=(%.7f, %.7f), heading=%.2f deg',
                      self.origin_lat, self.origin_lon, self.origin_yaw)

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
    def update_cone_frame(self, detections, timestamp):
        with self.cone_lock:
            # Expired objects cannot be resurrected by association.
            self.cone_map = [c for c in self.cone_map if 0 <= timestamp - c.last_update < self.cone_timeout]
            original = list(self.cone_map)
            candidates = []
            for i, (x, y, confidence) in enumerate(detections):
                for j, cone in enumerate(original):
                    distance = math.hypot(x - cone.x, y - cone.y)
                    if distance < self.cone_merge_distance:
                        candidates.append((distance, -confidence, i, j))
            used_detections, used_objects = set(), set()
            for _, _, i, j in sorted(candidates):
                if i in used_detections or j in used_objects:
                    continue
                x, y, confidence = detections[i]
                if original[j].update(x, y, timestamp, confidence):
                    self.cone_merge_count += 1
                used_detections.add(i)
                used_objects.add(j)
            # Suppress duplicate fragments near already assigned objects; don't spawn ghosts.
            for i in sorted(range(len(detections)), key=lambda k: (-detections[k][2], k)):
                if i in used_detections:
                    continue
                x, y, confidence = detections[i]
                if any(math.hypot(x - c.x, y - c.y) < self.cone_merge_distance for c in self.cone_map):
                    continue
                self.cone_map.append(ConeObject(x, y, timestamp, confidence))
                self.cone_new_count += 1

    # ============================================================
    # 发布车辆位姿
    # ============================================================
    def publish_vehicle_state(self):
        sample = self.pose_history.latest(rospy.Time.now().to_sec(), self.sensor_timeout)
        if not sample or not sample['valid']:
            return
        msg = PoseStamped()
        msg.header.stamp = rospy.Time.from_sec(sample['stamp'])
        msg.header.frame_id = 'map'
        msg.pose.position.x, msg.pose.position.y = sample['x'], sample['y']
        msg.pose.orientation.z = math.sin(sample['yaw'] / 2.0)
        msg.pose.orientation.w = math.cos(sample['yaw'] / 2.0)
        self.vehicle_pose_pub.publish(msg)

    # ============================================================
    # 发布锥筒地图（保持和 zhixian.py 兼容）
    # ============================================================
    def publish_cone_map(self, observation_stamp):
        cone_array_msg = ConeArray()
        cone_array_msg.header = Header()
        cone_array_msg.header.stamp = rospy.Time.from_sec(observation_stamp)
        cone_array_msg.header.frame_id = "map"

        with self.cone_lock:
            for cone_obj in self.cone_map:
                if cone_obj.last_update != observation_stamp:
                    continue
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
        if not self.enable_visualization:
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