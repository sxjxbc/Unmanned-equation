#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
FSAE Formula Student Autonomous Vehicle - Optimized Localization and Mapping Module
修改：输入源改为激光雷达聚类 /clustered_points(PoseArray)，颜色来自视觉ConeDetection
坐标系：激光输出 base_link；输出世界map坐标系
"""

from __future__ import print_function
import rospy
import numpy as np
import math
import cv2
from sensor_msgs.msg import NavSatFix, Imu
from geometry_msgs.msg import PoseStamped, PoseArray
from std_msgs.msg import Header
import threading
import sys
import time
import tf.transformations


if sys.version_info[0] < 3:
    reload(sys)
    sys.setdefaultencoding('utf-8')

from msgs.msg import ConeDetection, ConeArray, Cone


class CoordinateTransformer(object):
    """Coordinate transformation utilities"""
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
        yaw_rad = math.radians(origin_yaw_deg)
        x_vehicle = y_geo * math.cos(yaw_rad) + x_geo * math.sin(yaw_rad)
        y_vehicle = -y_geo * math.sin(yaw_rad) + x_geo * math.cos(yaw_rad)
        return x_vehicle, y_vehicle


class ConeObject(object):
    """优化的锥筒对象类"""
    def __init__(self, x, y, color, timestamp):
        self.x = x
        self.y = y
        self.color = color
        self.observations = [(x, y, timestamp)]
        self.last_update = timestamp
        self.confidence = 1.0
        self.std_dev_x = 0.0
        self.std_dev_y = 0.0
    
    def update(self, x, y, timestamp):
        distance = math.sqrt((x - self.x)**2 + (y - self.y)**2)
        if distance > 2.0:
            rospy.logwarn("[Cone Update] Large deviation detected: %.2fm for %s cone at (%.2f, %.2f)", 
                         distance, self.color, self.x, self.y)
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
    def __init__(self):
        rospy.init_node('high_speed_path_follow_node', anonymous=False)
        
        self.imu_init_wait_time = rospy.get_param('~imu_init_wait_time', 3.0)
        self.gps_lat_min = rospy.get_param('~gps_lat_min', 30.0)
        self.gps_lat_max = rospy.get_param('~gps_lat_max', 40.0)
        self.gps_lon_min = rospy.get_param('~gps_lon_min', 110.0)
        self.gps_lon_max = rospy.get_param('~gps_lon_max', 125.0)

        # 激光输入不需要相机偏移，这里保留参数仅兼容旧launch，不再使用
        self.camera_offset_x = rospy.get_param('~camera_offset_x', 0.5)
        self.camera_offset_y = rospy.get_param('~camera_offset_y', 0.0)
        self.camera_offset_z = rospy.get_param('~camera_offset_z', 0.3)
        
        self.cone_merge_distance_same_color = rospy.get_param('~cone_merge_distance_same_color', 2.5)
        self.cone_merge_distance_diff_color = rospy.get_param('~cone_merge_distance_diff_color', 1.5)
        self.laser_visual_match_thresh = rospy.get_param('~laser_visual_match_thresh',1.2)
        self.cone_timeout = rospy.get_param('~cone_timeout', 99999999999.0)
        self.vis_map_size = rospy.get_param('~vis_map_size', 60)
        
        self.window_width = 1000
        self.window_height = 1000
        self.pixels_per_meter = self.window_width / (2.0 * self.vis_map_size)
        
        self.camera_center_x = 0.0
        self.camera_center_y = 0.0
        self.camera_smoothing = 0.15
        
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

        # 缓存最新视觉检测，用于给激光点匹配颜色
        self.latest_visual_detections = []
        
        self.gps_msg_count = 0
        self.imu_msg_count = 0
        self.laser_cone_count = 0
        self.cone_merge_count = 0
        self.cone_conflict_count = 0
        
        self.vis_shutdown = threading.Event()

        # ========== 修改订阅 ==========
        self.gps_sub = rospy.Subscriber('/GPS_data', NavSatFix, self.gps_callback, queue_size=10)
        self.imu_sub = rospy.Subscriber('/imu_data', Imu, self.imu_callback, queue_size=50)
        # 激光聚类点 PoseArray（与 lidar_nodes/clustering.py 发布话题一致）
        self.laser_sub = rospy.Subscriber('/clustered_points', PoseArray, self.laser_cone_callback, queue_size=20)
        # 保留视觉，只拿颜色，不拿位置
        self.visual_sub = rospy.Subscriber('/perception/cones', ConeDetection, self.visual_color_cache_callback, queue_size=20)

        self.vehicle_pose_pub = rospy.Publisher('/vehicle_pose', PoseStamped, queue_size=10)
        self.cone_map_pub = rospy.Publisher('/cone_map', ConeArray, queue_size=10)
        
        print("\n" + "=" * 60)
        rospy.loginfo("[FSAE SLAM] Node started -- INPUT: LiDAR /clustered_points(PoseArray) + Visual color")
        rospy.loginfo("[坐标系说明]")
        rospy.loginfo("  激光输出坐标系: base_link（车体）")
        rospy.loginfo("  世界输出坐标系: map")
        print("=" * 60)

    def visual_color_cache_callback(self, msg):
        """只缓存视觉检测，用于给激光点匹配颜色，不使用视觉位置"""
        item = {
            "x":msg.x,
            "y":msg.y,
            "z":msg.z,
            "color":msg.color
        }
        self.latest_visual_detections.append(item)
        # 只保留最近20帧，防止内存暴涨
        if len(self.latest_visual_detections) > 20:
            self.latest_visual_detections.pop(0)

    def laser_cone_callback(self, msg):
        """激光聚类锥桶回调，PoseArray，每个pose是base_link下锥桶位置"""
        state = self.get_state_snapshot()
        if not state['is_initialized']:
            return
        if state['current_yaw'] is None:
            return

        self.laser_cone_count +=1
        for pose in msg.poses:
            lidar_base_x = pose.position.x
            lidar_base_y = pose.position.y

            # --------------------------
            # 激光已经是 base_link！！！不要加相机偏移！！！
            # --------------------------
            # 1、base_link车体坐标 → 世界map坐标
            yaw_relative = state['current_yaw'] - state['origin_yaw']
            while yaw_relative > 180:
                yaw_relative -= 360
            while yaw_relative < -180:
                yaw_relative += 360
            yaw_rad = math.radians(yaw_relative)
            cos_yaw = math.cos(yaw_rad)
            sin_yaw = math.sin(yaw_rad)

            world_x_rel = lidar_base_x * cos_yaw - lidar_base_y * sin_yaw
            world_y_rel = lidar_base_x * sin_yaw + lidar_base_y * cos_yaw

            world_x = state['current_x'] + world_x_rel
            world_y = state['current_y'] + world_y_rel

            # 2、就近匹配视觉检测拿颜色
            best_color = "unknown"
            best_dist = float("inf")
            for vis_item in self.latest_visual_detections:
                # vis_item的x,y是相机坐标系，先转到base_link车体
                vis_base_x = vis_item["x"] + self.camera_offset_x
                vis_base_y = vis_item["y"] + self.camera_offset_y
                d = math.hypot(lidar_base_x - vis_base_x, lidar_base_y - vis_base_y)
                if d < self.laser_visual_match_thresh and d < best_dist:
                    best_dist = d
                    best_color = vis_item["color"]

            self.update_cone_map_optimized(world_x, world_y, best_color)

    def update_camera_center(self, vehicle_x, vehicle_y):
        self.camera_center_x += (vehicle_x - self.camera_center_x) * self.camera_smoothing
        self.camera_center_y += (vehicle_y - self.camera_center_y) * self.camera_smoothing

    def world_to_screen(self, world_x, world_y):
        relative_x = world_x - self.camera_center_x
        relative_y = world_y - self.camera_center_y
        screen_x = int(self.window_width / 2 + relative_y * self.pixels_per_meter)
        screen_y = int(self.window_height / 2 - relative_x * self.pixels_per_meter)
        return screen_x, screen_y

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

    def gps_callback(self, msg):
        with self.state_lock:
            self.gps_msg_count += 1
            self.current_lat = msg.latitude
            self.current_lon = msg.longitude
            
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

    def imu_callback(self, msg):
        with self.state_lock:
          self.imu_msg_count += 1

        # ========== 修复：四元数提取yaw（角度制） ==========
        q = [msg.orientation.x,
             msg.orientation.y,
             msg.orientation.z,
             msg.orientation.w]
        roll_rad, pitch_rad, yaw_rad = tf.transformations.euler_from_quaternion(q)
        self.current_yaw = math.degrees(yaw_rad)   # 存角度，和原有代码逻辑匹配
        # =================================================

        if self.imu_msg_count == 1:
            rospy.loginfo("[IMU] First data received! yaw init: %.2f deg", self.current_yaw)

        if not self._is_initialized:
            self.check_initialization()


    def is_gps_valid(self, lat, lon):
        if lat is None or lon is None:
            return False
        return (self.gps_lat_min <= lat <= self.gps_lat_max and
                self.gps_lon_min <= lon <= self.gps_lon_max)

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

    def update_cone_map_optimized(self, x, y, color):
        with self.cone_lock:
            current_time = rospy.Time.now().to_sec()
            best_same_color_cone = None
            best_same_color_distance = float('inf')
            has_diff_color_conflict = False
            conflict_cone = None
            
            for cone in self.cone_map:
                distance = math.sqrt((cone.x - x)**2 + (cone.y - y)**2)
                if cone.color == color:
                    if distance < self.cone_merge_distance_same_color and distance < best_same_color_distance:
                        best_same_color_cone = cone
                        best_same_color_distance = distance
                else:
                    if distance < self.cone_merge_distance_diff_color:
                        has_diff_color_conflict = True
                        conflict_cone = cone
            
            if best_same_color_cone is not None:
                best_same_color_cone.update(x, y, current_time)
                self.cone_merge_count += 1
                
            elif has_diff_color_conflict:
                self.cone_conflict_count += 1
                if self.cone_conflict_count % 10 == 0:
                    rospy.logwarn("[Cone Conflict] Detected %s cone at (%.2f, %.2f) conflicts with existing %s cone at (%.2f, %.2f)", 
                                 color, x, y, conflict_cone.color, conflict_cone.x, conflict_cone.y)
                if conflict_cone.confidence < 0.3:
                    rospy.loginfo("[Cone Conflict] Replacing low-confidence %s cone with new %s cone", 
                                 conflict_cone.color, color)
                    self.cone_map.remove(conflict_cone)
                    new_cone = ConeObject(x, y, color, current_time)
                    self.cone_map.append(new_cone)
            else:
                new_cone = ConeObject(x, y, color, current_time)
                self.cone_map.append(new_cone)
                rospy.loginfo("[Cone] NEW %s cone at (%.2f, %.2f) | Total: %d | Merges: %d | Conflicts: %d",
                            color, x, y, len(self.cone_map), self.cone_merge_count, self.cone_conflict_count)
            
            self.cone_map = [cone for cone in self.cone_map 
                           if (current_time - cone.last_update) < self.cone_timeout]
        
        self.publish_cone_map()

    def publish_vehicle_state(self):
        state = self.get_state_snapshot()
        pose_msg = PoseStamped()
        pose_msg.header = Header()
        pose_msg.header.stamp = rospy.Time.now()
        pose_msg.header.frame_id = "map"
        pose_msg.pose.position.x = state['current_x']
        pose_msg.pose.position.y = state['current_y']
        pose_msg.pose.position.z = 0.0
        
        if state['origin_yaw'] is not None:
            yaw_relative = state['current_yaw'] - state['origin_yaw']
            while yaw_relative > 180:
                yaw_relative -= 360
            while yaw_relative < -180:
                yaw_relative += 360
            yaw_rad = math.radians(yaw_relative)
            pose_msg.pose.orientation.z = math.sin(yaw_rad / 2.0)
            pose_msg.pose.orientation.w = math.cos(yaw_rad / 2.0)
        
        self.vehicle_pose_pub.publish(pose_msg)

    def publish_cone_map(self):
        cone_array_msg = ConeArray()
        cone_array_msg.header = Header()
        cone_array_msg.header.stamp = rospy.Time.now()
        cone_array_msg.header.frame_id = "map"
        with self.cone_lock:
            for cone_obj in self.cone_map:
                cone_msg = Cone()
                cone_msg.color = cone_obj.color
                cone_msg.x = cone_obj.x
                cone_msg.y = cone_obj.y
                cone_msg.confidence = cone_obj.confidence
                cone_array_msg.cones.append(cone_msg)
        self.cone_map_pub.publish(cone_array_msg)

    def draw_init_screen(self):
        img = np.zeros((self.window_height, self.window_width, 3), dtype=np.uint8)
        img[:] = (40, 40, 40)
        state = self.get_state_snapshot()
        cv2.putText(img, "SYSTEM INITIALIZING", (150, 100),
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
        origin_x, origin_y = self.world_to_screen(0, 0)
        if -50 <= origin_x < self.window_width + 50 and -50 <= origin_y < self.window_height + 50:
            x_axis_end_x, x_axis_end_y = self.world_to_screen(5, 0)
            cv2.arrowedLine(img, (origin_x, origin_y), (x_axis_end_x, x_axis_end_y), 
                           (0, 0, 255), 2, tipLength=0.2)
            cv2.putText(img, "X (Forward)", (x_axis_end_x + 5, x_axis_end_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
            y_axis_end_x, y_axis_end_y = self.world_to_screen(0, 5)
            cv2.arrowedLine(img, (origin_x, origin_y), (y_axis_end_x, y_axis_end_y), 
                           (0, 255, 0), 2, tipLength=0.2)
            cv2.putText(img, "Y (Right)", (y_axis_end_x + 5, y_axis_end_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
            cv2.drawMarker(img, (origin_x, origin_y), (255, 255, 255), 
                          cv2.MARKER_CROSS, 20, 2)
            cv2.putText(img, "ORIGIN", (origin_x + 10, origin_y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        color_map = {
            'yellow': (0, 255, 255),
            'blue': (255, 0, 0),
            'orange': (0, 165, 255),
            'red': (0, 0, 255),
            'unknown': (120,120,120)
        }
        with self.cone_lock:
            for cone in self.cone_map:
                cone_color = color_map.get(cone.color.lower(), (128, 128, 128))
                cx, cy = self.world_to_screen(cone.x, cone.y)
                if -20 <= cx < self.window_width + 20 and -20 <= cy < self.window_height + 20:
                    radius = 6
                    if cone.confidence < 0.5:
                        faded_color = tuple(int(c * 0.5) for c in cone_color)
                        cv2.circle(img, (cx, cy), radius, faded_color, -1)
                        cv2.circle(img, (cx, cy), radius, (128, 128, 128), 1)
                    else:
                        cv2.circle(img, (cx, cy), radius, cone_color, -1)
                        cv2.circle(img, (cx, cy), radius, (255, 255, 255), 1)
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
            left_x = state['current_x'] + vehicle_width/2 * math.cos(yaw_rad + math.pi/2)
            left_y = state['current_y'] + vehicle_width/2 * math.sin(yaw_rad + math.pi/2)
            right_x = state['current_x'] + vehicle_width/2 * math.cos(yaw_rad - math.pi/2)
            right_y = state['current_y'] + vehicle_width/2 * math.sin(yaw_rad - math.pi/2)
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
        panel_height = 26
        panel_width = 42
        cv2.rectangle(img, (10, 10), (panel_width, panel_height), (0, 100, 0), -1)
        cv2.rectangle(img, (10, 10), (panel_width, panel_height), (0, 255, 0), 2)
        y_pos = 40
        cv2.putText(img, "SYSTEM ACTIVE", (20, y_pos), 
                   cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 2)
        y_pos += 30
        pos_text = "Vehicle: (%.1f, %.1f)m" % (state['current_x'], state['current_y'])
        cv2.putText(img, pos_text, (20, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        y_pos += 25
        if state['origin_yaw'] is not None and state['current_yaw'] is not None:
            yaw_relative = state['current_yaw'] - state['origin_yaw']
            while yaw_relative > 180:
                yaw_relative -= 360
            while yaw_relative < -180:
                yaw_relative += 360
            yaw_text = "Yaw: %.1f deg (relative)" % yaw_relative
        else:
            yaw_text = "Yaw: N/A"
        cv2.putText(img, yaw_text, (20, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        y_pos += 25
        frame_text = "World Frame: map"
        cv2.putText(img, frame_text, (20, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
        y_pos += 25
        with self.cone_lock:
            cone_text = "Cones: %d" % len(self.cone_map)
        cv2.putText(img, cone_text, (20, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        y_pos += 25
        merge_text = "Merges: %d" % self.cone_merge_count
        cv2.putText(img, merge_text, (20, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        y_pos += 25
        conflict_text = "Conflicts: %d" % self.cone_conflict_count
        cv2.putText(img, conflict_text, (20, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 165, 0), 1)
        cv2.putText(img, "FSAE SLAM -- LiDAR Input", (self.window_width//2 - 180, 30),
                   cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 255, 0), 2)
        scale_length_m = 10
        scale_length_px = int(scale_length_m * self.pixels_per_meter)
        scale_x = self.window_width - 150
        scale_y = self.window_height - 30
        cv2.line(img, (scale_x, scale_y), (scale_x + scale_length_px, scale_y), (255, 255, 255), 2)
        cv2.line(img, (scale_x, scale_y - 5), (scale_x, scale_y + 5), (255, 255, 255), 2)
        cv2.line(img, (scale_x + scale_length_px, scale_y - 5), (scale_x + scale_length_px, scale_y + 5), (255, 255, 255), 2)
        cv2.putText(img, "10m", (scale_x + 30, scale_y - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        return img

    def visualization_loop(self):
        cv2.namedWindow('FSAE SLAM', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('FSAE SLAM', self.window_width, self.window_height)
        rospy.loginfo("[Visualization] OpenCV window created")
        while not rospy.is_shutdown() and not self.vis_shutdown.is_set():
            try:
                state = self.get_state_snapshot()
                if not state['is_initialized']:
                    img = self.draw_init_screen()
                else:
                    img = self.draw_map_screen()
                cv2.imshow('FSAE SLAM', img)
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

