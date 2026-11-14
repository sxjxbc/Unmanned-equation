#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
FSAE无人方程式赛车 - 数据模拟器（完整修复版）
修复：基于实际距离生成锥筒，避免密集和重叠，移除挡路的锥筒
"""

import rospy
import numpy as np
import math
from sensor_msgs.msg import NavSatFix, Imu
from mapping.msg import ConeDetection
from std_msgs.msg import Header

class FSAESimulator:
    """FSAE数据模拟器"""
    def __init__(self):
        rospy.init_node('fsae_simulator', anonymous=False)
        # ========== 发布器 ==========
        self.gps_pub = rospy.Publisher('/GPS_data', NavSatFix, queue_size=10)
        self.imu_pub = rospy.Publisher('/imu_data', Imu, queue_size=10)
        self.cone_pub = rospy.Publisher('/perception/cones', ConeDetection, queue_size=10)
        # ========== 仿真参数 ==========
        self.rate = 30 # Hz
        self.dt = 1.0 / self.rate
        # 起点GPS坐标
        self.origin_lat = 31.839917
        self.origin_lon = 117.226364
        # 车辆参数
        self.vehicle_speed = 5.0 # m/s
        self.lane_width = 3.0 # 车道宽度
        self.cone_spacing = 5.0 # 同侧锥筒间距（米）
        self.path_safety_margin = 1.2  # 路径安全距离（米）
        # 摄像头参数
        self.camera_range = 20.0
        self.camera_fov = 120.0
        self.camera_offset_x = 0.5
        # 噪声参数
        self.gps_noise = 0.1
        self.yaw_noise = 1.0
        self.cone_detection_noise = 0.15
        self.lateral_drift = 0.2
        # 起点停留时间
        self.start_wait_time = 20.0 # 秒
        self.start_wait_frames = int(self.start_wait_time * self.rate)
        # ========== 轨迹生成 ==========
        self.trajectory = self.generate_trajectory()
        self.current_idx = 0
        # ========== 锥筒生成（改进版） ==========
        self.cones = self.generate_cones_by_distance()
        rospy.loginfo("="*60)
        rospy.loginfo("FSAE数据模拟器已启动（完整修复版）")
        rospy.loginfo("起点停留: {} 秒 ({} 帧)".format(self.start_wait_time, self.start_wait_frames))
        rospy.loginfo("轨迹点数: {}".format(len(self.trajectory)))
        red_count = sum(1 for c in self.cones if c['color']=='red')
        blue_count = sum(1 for c in self.cones if c['color']=='blue')
        rospy.loginfo("锥筒总数: {} (红色: {}, 蓝色: {})".format(
            len(self.cones), red_count, blue_count))
        rospy.loginfo("锥筒间距: {:.1f}米".format(self.cone_spacing))
        rospy.loginfo("路径安全距离: {:.1f}米".format(self.path_safety_margin))
        rospy.loginfo("起点GPS: lat={:.6f}, lon={:.6f}".format(
            self.origin_lat, self.origin_lon))
        rospy.loginfo("="*60)
    
    def generate_trajectory(self):
        """生成车辆轨迹"""
        trajectory = []
        x, y, yaw = 0.0, 0.0, 0.0
        
        # ========== 起点停留 ==========
        for i in range(self.start_wait_frames):
            trajectory.append({'x': 0.0, 'y': 0.0, 'yaw': 0.0})
        
        # ========== 第一段：直行50米 ==========
        straight_distance = 50.0
        straight_steps = int(straight_distance / (self.vehicle_speed * self.dt))
        for i in range(straight_steps):
            y_offset = self.lateral_drift * np.sin(2 * np.pi * i / 50)
            trajectory.append({
                'x': self.vehicle_speed * self.dt * i,
                'y': y_offset,
                'yaw': 0.0
            })
        
        x = trajectory[-1]['x']
        y = trajectory[-1]['y']
        
        # ========== 第二段：向右转圈（半径15米） ==========
        circle_radius = 15.0
        circle_center_x = x
        circle_center_y = y - circle_radius
        circumference = 2 * np.pi * circle_radius
        circle_steps = int(circumference / (self.vehicle_speed * self.dt))
        
        for i in range(circle_steps):
            angle = np.pi/2 - 2*np.pi * i / circle_steps
            pos_x = circle_center_x + circle_radius * np.cos(angle)
            pos_y = circle_center_y + circle_radius * np.sin(angle)
            noise_amp = 0.15
            pos_x += noise_amp * np.sin(4 * angle)
            pos_y += noise_amp * np.cos(4 * angle)
            tangent_yaw = np.degrees(angle - np.pi/2)
            trajectory.append({'x': pos_x, 'y': pos_y, 'yaw': tangent_yaw})
        
        x = trajectory[-1]['x']
        y = trajectory[-1]['y']
        rospy.loginfo("右转圈后位置: x={:.2f}, y={:.2f}".format(x, y))
        
        # ========== 第三段：向左转圈（半径15米） ==========
        circle_center_x = x
        circle_center_y = y + circle_radius
        
        for i in range(circle_steps):
            angle = -np.pi/2 + 2*np.pi * i / circle_steps
            pos_x = circle_center_x + circle_radius * np.cos(angle)
            pos_y = circle_center_y + circle_radius * np.sin(angle)
            pos_x += noise_amp * np.sin(4 * angle)
            pos_y += noise_amp * np.cos(4 * angle)
            tangent_yaw = np.degrees(angle + np.pi/2)
            trajectory.append({'x': pos_x, 'y': pos_y, 'yaw': tangent_yaw})
        
        x = trajectory[-1]['x']
        y = trajectory[-1]['y']
        rospy.loginfo("左转圈后位置: x={:.2f}, y={:.2f}".format(x, y))
        
        # ========== 第四段：最后直行50米 ==========
        straight_distance = 50.0
        straight_steps = int(straight_distance / (self.vehicle_speed * self.dt))
        for i in range(straight_steps):
            y_offset = self.lateral_drift * np.sin(2 * np.pi * i / 50)
            trajectory.append({
                'x': x + self.vehicle_speed * self.dt * i,
                'y': y + y_offset,
                'yaw': 0.0
            })
        
        final_x = trajectory[-1]['x']
        final_y = trajectory[-1]['y']
        rospy.loginfo("最终位置: x={:.2f}, y={:.2f}".format(final_x, final_y))
        return trajectory
    
    def is_cone_too_close_to_path(self, cone_x, cone_y):
        """检查锥筒是否距离整条行驶路径太近（小于safety_margin）"""
        # 每隔3帧采样一次，提高检测效率
        for i in range(self.start_wait_frames, len(self.trajectory), 3):
            traj_point = self.trajectory[i]
            traj_x = traj_point['x']
            traj_y = traj_point['y']
            
            # 计算锥筒到轨迹点的距离
            distance = np.sqrt((cone_x - traj_x)**2 + (cone_y - traj_y)**2)
            
            # 如果距离小于安全距离，返回True
            if distance < self.path_safety_margin:
                return True
        
        return False
    
    def generate_cones_by_distance(self):
        """基于实际距离生成锥筒 - 避免密集和重叠，移除挡路的锥筒"""
        left_cones = []  # 红色（左侧）
        right_cones = []  # 蓝色（右侧）
        
        # 跳过起点停留帧，从实际行驶开始
        last_left_pos = None
        last_right_pos = None
        accumulated_distance = 0.0
        
        blocked_count = 0  # 统计被阻挡的锥筒数量
        
        for i in range(self.start_wait_frames, len(self.trajectory)):
            point = self.trajectory[i]
            yaw_rad = np.radians(point['yaw'])
            
            # 计算当前位置的左右锥筒坐标
            left_x = point['x'] - (self.lane_width/2) * np.sin(yaw_rad)
            left_y = point['y'] + (self.lane_width/2) * np.cos(yaw_rad)
            right_x = point['x'] + (self.lane_width/2) * np.sin(yaw_rad)
            right_y = point['y'] - (self.lane_width/2) * np.cos(yaw_rad)
            
            # 第一个锥筒：检查是否挡路后再添加
            if last_left_pos is None:
                # 检查左右锥筒是否会挡路
                left_blocked = self.is_cone_too_close_to_path(left_x, left_y)
                right_blocked = self.is_cone_too_close_to_path(right_x, right_y)
                
                if not left_blocked and not right_blocked:
                    left_cones.append({'x': left_x, 'y': left_y, 'color': 'red'})
                    right_cones.append({'x': right_x, 'y': right_y, 'color': 'blue'})
                    last_left_pos = (left_x, left_y)
                    last_right_pos = (right_x, right_y)
                else:
                    if left_blocked:
                        blocked_count += 1
                    if right_blocked:
                        blocked_count += 1
                continue
            
            # 计算与上一个锥筒的实际距离
            left_dist = np.sqrt((left_x - last_left_pos[0])**2 + 
                               (left_y - last_left_pos[1])**2)
            right_dist = np.sqrt((right_x - last_right_pos[0])**2 + 
                                (right_y - last_right_pos[1])**2)
            
            # 使用较小的距离作为参考
            min_dist = min(left_dist, right_dist)
            accumulated_distance += min_dist
            
            # 当累积距离超过设定间距时，尝试放置新锥筒
            if accumulated_distance >= self.cone_spacing:
                # 检查左右锥筒是否会挡路
                left_blocked = self.is_cone_too_close_to_path(left_x, left_y)
                right_blocked = self.is_cone_too_close_to_path(right_x, right_y)
                
                # 只添加不挡路的锥筒
                if not left_blocked:
                    left_cones.append({'x': left_x, 'y': left_y, 'color': 'red'})
                    last_left_pos = (left_x, left_y)
                else:
                    blocked_count += 1
                
                if not right_blocked:
                    right_cones.append({'x': right_x, 'y': right_y, 'color': 'blue'})
                    last_right_pos = (right_x, right_y)
                else:
                    blocked_count += 1
                
                accumulated_distance = 0.0  # 重置累积距离
        
        rospy.loginfo("生成锥筒完成: 红色={}, 蓝色={}".format(len(left_cones), len(right_cones)))
        rospy.loginfo("因挡路被移除的锥筒: {}个".format(blocked_count))
        
        # 合并左右锥筒
        all_cones = left_cones + right_cones
        
        # 验证锥筒间距
        self.validate_cone_spacing(left_cones, 'red')
        self.validate_cone_spacing(right_cones, 'blue')
        
        return all_cones
    
    def validate_cone_spacing(self, cones, color):
        """验证锥筒间距是否合理"""
        if len(cones) < 2:
            return
        
        distances = []
        for i in range(1, len(cones)):
            dx = cones[i]['x'] - cones[i-1]['x']
            dy = cones[i]['y'] - cones[i-1]['y']
            dist = np.sqrt(dx**2 + dy**2)
            distances.append(dist)
        
        min_dist = min(distances)
        max_dist = max(distances)
        avg_dist = np.mean(distances)
        
        rospy.loginfo("[{}锥筒间距] 最小: {:.2f}m, 最大: {:.2f}m, 平均: {:.2f}m".format(
            color, min_dist, max_dist, avg_dist))
        
        # 检查是否有过近的锥筒
        too_close = [d for d in distances if d < self.cone_spacing * 0.5]
        if too_close:
            rospy.logwarn("[警告] {}锥筒有{}处间距过近 (< {:.1f}m)".format(
                color, len(too_close), self.cone_spacing * 0.5))
    
    def cartesian_to_gps(self, x, y):
        """将笛卡尔坐标转换为GPS坐标"""
        R = 6371000
        origin_lat_rad = np.radians(self.origin_lat)
        lat = self.origin_lat + np.degrees(y / R)
        lon = self.origin_lon + np.degrees(x / (R * np.cos(origin_lat_rad)))
        return lat, lon
    
    def detect_cones(self, vehicle_x, vehicle_y, vehicle_yaw):
        """模拟摄像头检测锥筒"""
        detected = []
        vehicle_yaw_rad = np.radians(vehicle_yaw)
        
        # 摄像头位置
        cam_x = vehicle_x + self.camera_offset_x * np.cos(vehicle_yaw_rad)
        cam_y = vehicle_y + self.camera_offset_x * np.sin(vehicle_yaw_rad)
        
        for cone in self.cones:
            # 计算锥筒相对摄像头的位置
            dx = cone['x'] - cam_x
            dy = cone['y'] - cam_y
            
            # 距离检查
            distance = np.sqrt(dx**2 + dy**2)
            if distance > self.camera_range or distance < 0.5:
                continue
            
            # 视野角度检查
            cone_angle = np.degrees(np.arctan2(dy, dx))
            angle_diff = (cone_angle - vehicle_yaw + 180) % 360 - 180
            if abs(angle_diff) > self.camera_fov / 2:
                continue
            
            # 转换到摄像头坐标系
            cos_yaw = np.cos(-vehicle_yaw_rad)
            sin_yaw = np.sin(-vehicle_yaw_rad)
            cam_rel_x = dx * cos_yaw - dy * sin_yaw
            cam_rel_y = dx * sin_yaw + dy * cos_yaw
            
            # 添加检测噪声
            cam_rel_x += np.random.normal(0, self.cone_detection_noise)
            cam_rel_y += np.random.normal(0, self.cone_detection_noise)
            cam_rel_z = 0.0 + np.random.normal(0, 0.05)
            
            detected.append({
                'color': cone['color'],
                'x': cam_rel_x,
                'y': cam_rel_y,
                'z': cam_rel_z
            })
        
        return detected
    
    def publish_data(self):
        """发布传感器数据"""
        if self.current_idx >= len(self.trajectory):
            rospy.loginfo("轨迹完成，模拟器停止")
            return False
        
        current_point = self.trajectory[self.current_idx]
        
        # 添加噪声
        noisy_x = current_point['x'] + np.random.normal(0, self.gps_noise)
        noisy_y = current_point['y'] + np.random.normal(0, self.gps_noise)
        noisy_yaw = current_point['yaw'] + np.random.normal(0, self.yaw_noise)
        
        # 发布GPS数据
        gps_msg = NavSatFix()
        gps_msg.header = Header()
        gps_msg.header.stamp = rospy.Time.now()
        gps_msg.header.frame_id = "gps"
        lat, lon = self.cartesian_to_gps(noisy_x, noisy_y)
        gps_msg.latitude = lat
        gps_msg.longitude = lon
        gps_msg.altitude = 0.0
        self.gps_pub.publish(gps_msg)
        
        # 发布IMU数据
        imu_msg = Imu()
        imu_msg.header = Header()
        imu_msg.header.stamp = rospy.Time.now()
        imu_msg.header.frame_id = "imu"
        normalized_yaw = noisy_yaw % 360
        imu_msg.orientation.z = normalized_yaw
        imu_msg.orientation.w = 1.0
        self.imu_pub.publish(imu_msg)
        
        # 检测并发布锥筒
        detected_cones = self.detect_cones(
            current_point['x'],
            current_point['y'],
            current_point['yaw']
        )
        
        for cone in detected_cones:
            cone_msg = ConeDetection()
            cone_msg.header = Header()
            cone_msg.header.stamp = rospy.Time.now()
            cone_msg.header.frame_id = "camera"
            cone_msg.color = cone['color']
            cone_msg.x = cone['x']
            cone_msg.y = cone['y']
            cone_msg.z = cone['z']
            self.cone_pub.publish(cone_msg)
        
        # 每30帧打印一次状态
        if self.current_idx % self.rate == 0:
            time_sec = self.current_idx / float(self.rate)
            if self.current_idx < self.start_wait_frames:
                stage = "起点等待"
            else:
                stage = "行驶中"
            rospy.loginfo("[{}] 时间: {:.1f}s | 位置: ({:.1f}, {:.1f})m | 航向: {:.1f}度 | 检测: {}个锥筒".format(
                stage, time_sec, current_point['x'], current_point['y'], 
                current_point['yaw'], len(detected_cones)))
        
        self.current_idx += 1
        return True
    
    def run(self):
        """运行模拟器"""
        rate = rospy.Rate(self.rate)
        rospy.loginfo("开始发布数据...")
        rospy.sleep(1.0)
        
        while not rospy.is_shutdown():
            if not self.publish_data():
                break
            rate.sleep()
        
        rospy.loginfo("模拟完成！")

def main():
    try:
        simulator = FSAESimulator()
        simulator.run()
    except rospy.ROSInterruptException:
        pass
    except Exception as e:
        rospy.logerr("模拟器错误: {}".format(str(e)))
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()