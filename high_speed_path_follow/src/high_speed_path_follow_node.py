#!/usr/bin/env python
# -*- coding: utf-8 -*-
import rospy
from msgs.msg import ConeArray, Cone
import numpy as np
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
import matplotlib.pyplot as plt
from threading import Thread
import time
import threading

# 设置matplotlib
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

class SimplePathPlanner:
    def __init__(self):
        rospy.init_node('high_speed_path_follow_node', anonymous=True)
        
        # 参数设置
        self.publish_rate = 10.0
        self.max_connection_distance = 8.0  # 最大连接距离
        self.min_connection_distance = 2.0  # 最小连接距离
        self.max_path_point_distance = 5.0  # 相邻路径点最大连接距离
        self.max_path_length = 50000.0  # 最大路径总长度

        # 车辆位姿（map全局坐标系）
        self.vehicle_x = None
        self.vehicle_y = None
        self.vehicle_yaw = None
        self.pose_received = False
        
        # 锥筒数据（SLAM输出已经是map全局坐标系）
        self.global_left_cones = []
        self.global_right_cones = []
        self.valid_connections = []
        self.global_path_points = []
        self.ordered_path_points = []
        
        # 线程锁
        self.data_lock = threading.Lock()
        self.data_received = False
        
        # ROS路径对象
        self.path = Path()
        self.path.header.frame_id = "map"
        self.path_ready = False
        
        # matplotlib初始化
        self.fig = None
        self.ax = None
        self.plot_initialized = False
        
        # ==========订阅修改============
        self.path_pub = rospy.Publisher("/planned_path", Path, queue_size=10)
        self.timer = rospy.Timer(rospy.Duration(1.0/self.publish_rate), self.publish_path_timer)
        # 订阅SLAM输出的小车位姿 /vehicle_pose
        self.pose_sub = rospy.Subscriber("/vehicle_pose", PoseStamped, self.pose_callback)
        # 订阅SLAM建图输出锥桶地图 /cone_map
        self.track_sub = rospy.Subscriber("/cone_map", ConeArray, self.cone_map_callback)

        # 启动可视化线程
        self.viz_thread = Thread(target=self.visualization_worker)
        self.viz_thread.daemon = True
        self.viz_thread.start()
        
        rospy.loginfo("Path planner initialized: use SLAM /cone_map map coordinate")

    def pose_callback(self, msg):
        """接收slam输出车辆全局位姿 PoseStamped(map)"""
        with self.data_lock:
            self.vehicle_x = msg.pose.position.x
            self.vehicle_y = msg.pose.position.y
            # 四元数转yaw
            q = msg.pose.orientation
            import tf.transformations
            _, _, yaw = tf.transformations.euler_from_quaternion([q.x, q.y, q.z, q.w])
            self.vehicle_yaw = yaw
            self.pose_received = True
            rospy.loginfo_throttle(5, "Vehicle pose updated: (%.2f, %.2f, yaw=%.2f)", 
                                self.vehicle_x, self.vehicle_y, self.vehicle_yaw)

    def cone_map_callback(self, data):
        """接收SLAM输出的全局锥桶地图 ConeArray，已经是map坐标系"""
        rospy.loginfo("Received slam cone map total: %d cones", len(data.cones))
        
        with self.data_lock:
            if not self.pose_received:
                rospy.logwarn("No vehicle pose available, skip planning")
                return

            new_left_cones = []
            new_right_cones = []
            # 根据颜色区分左右锥桶，根据你的实际颜色字符串匹配
            for cone in data.cones:
                cx = cone.x
                cy = cone.y
                dist_car = np.hypot(cx - self.vehicle_x, cy - self.vehicle_y)
                if dist_car > 200:
                    continue
                # 黄色/橙色=左；蓝色=右，根据你的slam输出颜色修改
                col = cone.color.lower()
                if col in ("yellow", "orange"):
                    new_left_cones.append((cx, cy))
                elif col == "blue":
                    new_right_cones.append((cx, cy))

            # 合并锥筒，去重
            self.merge_cones(new_left_cones, new_right_cones)
            self.data_received = True
            
            rospy.loginfo("Updated cones: Left %d, Right %d", 
                         len(self.global_left_cones), len(self.global_right_cones))
            # 执行路径规划
            self.plan_path()

    def merge_cones(self, new_left, new_right):
        """合并新的锥筒数据，避免重复"""
        merge_threshold = 1.0  # 1米内的锥筒认为是重复的
        
        for new_cone in new_left:
            is_duplicate = False
            for existing_cone in self.global_left_cones:
                if np.sqrt((new_cone[0] - existing_cone[0])**2 + 
                          (new_cone[1] - existing_cone[1])**2) < merge_threshold:
                    is_duplicate = True
                    break
            if not is_duplicate:
                self.global_left_cones.append(new_cone)
        
        for new_cone in new_right:
            is_duplicate = False
            for existing_cone in self.global_right_cones:
                if np.sqrt((new_cone[0] - existing_cone[0])**2 + 
                          (new_cone[1] - existing_cone[1])**2) < merge_threshold:
                    is_duplicate = True
                    break
            if not is_duplicate:
                self.global_right_cones.append(new_cone)

    def find_valid_connections(self):
        """找到左右锥筒之间的有效连接"""
        connections = []
        
        for i, left_cone in enumerate(self.global_left_cones):
            for j, right_cone in enumerate(self.global_right_cones):
                distance = np.sqrt((left_cone[0] - right_cone[0])**2 + 
                                (left_cone[1] - right_cone[1])**2)
                
                if self.min_connection_distance <= distance <= self.max_connection_distance:
                    mid_x = (left_cone[0] + right_cone[0]) / 2.0
                    mid_y = (left_cone[1] + right_cone[1]) / 2.0
                    
                    if self.pose_received:
                        dx = mid_x - self.vehicle_x
                        dy = mid_y - self.vehicle_y
                        relative_angle = np.arctan2(dy, dx)
                        
                        angle_diff = np.abs(relative_angle - self.vehicle_yaw)
                        if angle_diff > np.pi:
                            angle_diff = 2 * np.pi - angle_diff
                        
                        if angle_diff > np.pi / 2:
                            continue
                    
                    connections.append({
                        'left_cone': left_cone,
                        'right_cone': right_cone,
                        'midpoint': (mid_x, mid_y),
                        'distance': distance,
                        'left_idx': i,
                        'right_idx': j
                    })
        return connections

    def order_path_points(self, path_points):
        """将路径点按照合理的顺序连接，形成连续路径"""
        if len(path_points) <= 1:
            return path_points
        
        ordered_points = []
        remaining_points = list(path_points)
        
        if self.pose_received:
            start_point = min(remaining_points, 
                            key=lambda p: np.sqrt((p[0] - self.vehicle_x)**2 + 
                                                (p[1] - self.vehicle_y)**2))
        else:
            start_point = remaining_points[0]
        
        ordered_points.append(start_point)
        remaining_points.remove(start_point)
        
        total_path_length = 0.0
        
        while remaining_points:
            current_point = ordered_points[-1]
            
            next_candidates = []
            for point in remaining_points:
                dist = np.sqrt((point[0] - current_point[0])**2 + 
                            (point[1] - current_point[1])**2)
                if dist <= self.max_path_point_distance:
                    next_candidates.append((point, dist))
            
            if next_candidates:
                next_candidates.sort(key=lambda x: x[1])
                next_point = next_candidates[0][0]
                
                last_dist = np.sqrt((next_point[0] - current_point[0])**2 + 
                                (next_point[1] - current_point[1])**2)
                total_path_length += last_dist
                
                if total_path_length <= self.max_path_length:
                    ordered_points.append(next_point)
                    remaining_points.remove(next_point)
                else:
                    break
            else:
                break
        
        if self.pose_received and len(ordered_points) > 1:
            vehicle_dx = np.cos(self.vehicle_yaw)
            vehicle_dy = np.sin(self.vehicle_yaw)
            vehicle_direction = np.array([vehicle_dx, vehicle_dy])
            
            path_direction = np.array([ordered_points[1][0] - ordered_points[0][0], 
                                    ordered_points[1][1] - ordered_points[0][1]])
            path_direction = path_direction / np.linalg.norm(path_direction)
            
            if np.dot(vehicle_direction, path_direction) < 0:
                ordered_points.reverse()
        
        return ordered_points

    def plan_path(self):
        """在map全局坐标系中规划路径"""
        if len(self.global_left_cones) == 0 or len(self.global_right_cones) == 0:
            rospy.logwarn("Insufficient cones for path planning: Left %d, Right %d", 
                        len(self.global_left_cones), len(self.global_right_cones))
            self.valid_connections = []
            self.global_path_points = []
            self.ordered_path_points = []
            return
        
        self.valid_connections = self.find_valid_connections()
        
        if not self.valid_connections:
            rospy.logwarn("No valid connections found between left and right cones")
            self.global_path_points = []
            self.ordered_path_points = []
            return
        
        path_points = [conn['midpoint'] for conn in self.valid_connections]
        self.global_path_points = path_points
        self.ordered_path_points = self.order_path_points(path_points)
        self.generate_ros_path()
        
        rospy.loginfo("Generated path: %d connections -> %d path points -> %d ordered points", 
                    len(self.valid_connections), len(self.global_path_points), 
                    len(self.ordered_path_points))

    def generate_ros_path(self):
        """生成ROS路径消息"""
        self.path.poses = []
        
        if not self.ordered_path_points:
            self.path_ready = False
            return
        
        current_time = rospy.Time.now()
        
        if self.pose_received and len(self.ordered_path_points) > 1:
            start_point = self.ordered_path_points[0]
            end_point = self.ordered_path_points[-1]
            
            vehicle_dx = np.cos(self.vehicle_yaw)
            vehicle_dy = np.sin(self.vehicle_yaw)
            vehicle_direction = np.array([vehicle_dx, vehicle_dy])
            
            path_dx = end_point[0] - start_point[0]
            path_dy = end_point[1] - start_point[1]
            path_direction = np.array([path_dx, path_dy])
            path_direction = path_direction / np.linalg.norm(path_direction)
            
            if np.dot(vehicle_direction, path_direction) < 0:
                self.ordered_path_points.reverse()
                rospy.loginfo("Reversed path direction to align with vehicle heading")
        
        for i, (x, y) in enumerate(self.ordered_path_points):
            pose = PoseStamped()
            pose.header.stamp = current_time
            pose.header.frame_id = "map"
            pose.pose.position.x = x
            pose.pose.position.y = y
            pose.pose.position.z = 0.0
            
            if i < len(self.ordered_path_points) - 1:
                next_x, next_y = self.ordered_path_points[i + 1]
                yaw = np.arctan2(next_y - y, next_x - x)
            else:
                yaw = self.vehicle_yaw if self.vehicle_yaw is not None else 0.0
            
            pose.pose.orientation.z = np.sin(yaw / 2.0)
            pose.pose.orientation.w = np.cos(yaw / 2.0)
            
            self.path.poses.append(pose)
        
        self.path.header.stamp = current_time
        self.path_ready = True

    def publish_path_timer(self, event):
        """定时发布路径"""
        if self.path_ready:
            self.path.header.stamp = rospy.Time.now()
            self.path_pub.publish(self.path)

    def visualization_worker(self):
        """可视化线程"""
        rospy.loginfo("Starting visualization worker")
        time.sleep(1.0)
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                self.init_matplotlib()
                break
            except Exception as e:
                rospy.logwarn("Matplotlib init attempt %d failed: %s", attempt + 1, str(e))
                time.sleep(1.0)
        
        if not self.plot_initialized:
            rospy.logerr("Failed to initialize matplotlib after %d attempts", max_retries)
            return
        
        self.update_plot()
        
        while not rospy.is_shutdown():
            try:
                self.update_plot()
                time.sleep(0.5)
            except Exception as e:
                rospy.logwarn("Visualization update error: %s", str(e))
                time.sleep(1.0)

    def init_matplotlib(self):
        try:
            import matplotlib
            matplotlib.use('TkAgg')
            plt.ion()
            
            self.fig, self.ax = plt.subplots(figsize=(12, 8))
            self.ax.set_title('Path Planning Visualization')
            self.ax.set_xlabel('X (m)')
            self.ax.set_ylabel('Y (m)')
            self.ax.grid(True, alpha=0.3)
            self.ax.set_aspect('equal')
            
            self.ax.set_xlim(-10, 10)
            self.ax.set_ylim(-10, 10)
            
            plt.show(block=False)
            plt.pause(0.1)
            
            self.plot_initialized = True
            rospy.loginfo("Visualization initialized successfully")
            
        except Exception as e:
            rospy.logerr("Visualization init failed: %s", str(e))
            raise

    def update_plot(self):
        if not self.plot_initialized:
            return
            
        try:
            with self.data_lock:
                self.ax.clear()
                self.ax.set_title('Path Planning - Global Coordinate System(SLAM map)')
                self.ax.set_xlabel('X (m)')
                self.ax.set_ylabel('Y (m)')
                self.ax.grid(True, alpha=0.3)
                self.ax.set_aspect('equal')
                
                if self.global_left_cones:
                    left_x = [c[0] for c in self.global_left_cones]
                    left_y = [c[1] for c in self.global_left_cones]
                    self.ax.scatter(left_x, left_y, c='orange', s=100, marker='^', 
                                label='Left Cones(yellow/orange)', alpha=0.8)
                
                if self.global_right_cones:
                    right_x = [c[0] for c in self.global_right_cones]
                    right_y = [c[1] for c in self.global_right_cones]
                    self.ax.scatter(right_x, right_y, c='blue', s=100, marker='^', 
                                label='Right Cones(blue)', alpha=0.8)
                
                if self.pose_received and self.vehicle_x is not None and self.vehicle_yaw is not None:
                    self.ax.scatter(self.vehicle_x, self.vehicle_y, c='green', s=200, 
                                marker='o', label='Vehicle', alpha=0.9)
                    
                    dx = 2.0 * np.cos(self.vehicle_yaw)
                    dy = 2.0 * np.sin(self.vehicle_yaw)
                    self.ax.arrow(self.vehicle_x, self.vehicle_y, dx, dy,
                                head_width=0.5, head_length=0.3, fc='green', ec='green')
                
                if self.ordered_path_points:
                    ordered_x = [p[0] for p in self.ordered_path_points]
                    ordered_y = [p[1] for p in self.ordered_path_points]
                    self.ax.scatter(ordered_x, ordered_y, c='lime', s=80, marker='s',
                                label='Ordered Path', alpha=0.9)
                    
                    if len(ordered_x) > 1:
                        self.ax.plot(ordered_x, ordered_y, 'green', linewidth=2, 
                                alpha=0.8, label='Final Path')
                
                self.ax.legend()
                self.fig.canvas.draw()
                self.fig.canvas.flush_events()
                
        except Exception as e:
            rospy.logwarn("Plot update error: %s", str(e))

    def cleanup(self):
        rospy.loginfo("Cleaning up resources")
        if self.fig:
            plt.close(self.fig)

if __name__ == '__main__':
    planner = None
    try:
        planner = SimplePathPlanner()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
    except KeyboardInterrupt:
        rospy.loginfo("Interrupted by user")
    finally:
        if planner:
            planner.cleanup()
        plt.close('all')
