#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
FSAE Formula Student - Real-time Straight Line Planner v2.0 - Stable Two-Pair

功能：
1. 持续订阅 /cone_map 和 /vehicle_pose
2. 按配置选择附近锥桶，在固定赛道坐标系拟合中心线
3. 不依赖锥桶颜色，只使用几何关系：车道宽约3m、同排锥桶纵向位置接近
4. 按当前车辆位置生成约30m前视的局部 /planned_path
5. 每个周期持续更新路径，而不是原来的“10秒接收后锁死一次”
6. 启动阶段必须至少识别到1组左右锥桶才开始发布路径；启动后短时丢失时保持上一条中心线

坐标约定：
- /vehicle_pose 和 /cone_map 均使用 jiantu 发布的 map 坐标系
- 车辆局部坐标：X 前，Y 右
"""

from __future__ import print_function
import math
import threading
import time
import json
from track_frame import TrackFrame, bootstrap_frame, paired_centers, median

import numpy as np
import rospy
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Header, String
import tf.transformations as tf_trans
from mapping.msg import ConeArray


class RealtimeStraightLinePlanner(object):
    def __init__(self):
        rospy.init_node('straight_line_planner_realtime', anonymous=False)

        # ==================== 赛道参数 ====================
        self.track_width = rospy.get_param('~track_width', 3.15)

        # 局部规划窗口：由于雷达最多稳定提供4个前方锥桶，没必要一次规划175m
        self.local_path_length = rospy.get_param('~local_path_length', 40.0)
        self.path_point_spacing = rospy.get_param('~path_point_spacing', 0.5)
        self.blend_distance = rospy.get_param('~blend_distance', 6.0)

        # ==================== 锥桶筛选 ====================
        self.min_cone_confidence = rospy.get_param('~min_cone_confidence', 0.25)
        # 保留更多候选，真正的选择在 pair_cones() 中完成。
        self.max_local_cones = rospy.get_param('~max_local_cones', 20)
        self.max_pair_count = rospy.get_param('~max_pair_count', 2)
        self.pair_row_min_spacing = rospy.get_param('~pair_row_min_spacing', 1.5)
        self.track_center_y = rospy.get_param('~track_center_y', 0.0)
        self.track_forward_min = rospy.get_param('~track_forward_min', 0.5)
        self.track_forward_max = rospy.get_param('~track_forward_max', 25.0)


        # 左右配对：实际赛道宽2.7~3.6m，欧氏距离门限含测量和错位余量。
        self.pair_distance_min = rospy.get_param('~pair_distance_min', 2.5)
        self.pair_distance_max = rospy.get_param('~pair_distance_max', 3.8)
        self.pair_max_longitudinal_diff = rospy.get_param('~pair_max_longitudinal_diff', 1.5)

        # 直线赛道稳定保护：
        # 1) 新拟合的中心线斜率不能过大；
        # 2) 相邻周期斜率不能突然跳变；
        # 3) 只有一个左右锥桶对时，保持上一周期的斜率，只更新横向位置。
        self.straight_slope_limit = rospy.get_param('~straight_slope_limit', 0.10)
        self.max_slope_change = rospy.get_param('~max_slope_change', 0.02)
        self.slope_alpha = rospy.get_param('~slope_alpha', 0.12)

        # ==================== 稳定性 ====================
        self.last_center_b = None
        self.centerline_timeout = rospy.get_param('~centerline_timeout', 1.75)
        self.max_center_offset = rospy.get_param('~max_center_offset', 1.5)
        self.publish_hz = rospy.get_param('~publish_hz', 10.0)
        self.yaw_track_world = None   # 赛道正前方的绝对世界朝向

        # ==================== 初始航向偏差参数 ====================
        self.yaw_bias = 0.0             # 初始化航向偏差
        self.yaw_bias_initialized = False  # 是否已完成初始化
        self.yaw_bias_history = []      # 记录历史偏差，用于求平均
        self.max_bias_samples = 5       # 最多用前5组求平均

        # ==================== 路径滤波参数 ====================
        # 新路径权重（越小越能抑制抖动）
        self.centerline_alpha = rospy.get_param('~new_path_weight', 0.15)
        # 路径点滑动平均滤波窗口大小（建议奇数）
        self.path_smooth_window = rospy.get_param('~path_smooth_window', 3)

        # ==================== 单侧修正参数 ====================
        # 当只识别到单侧锥桶时，向另一侧修正的额外偏移量(m)
        self.side_correction_offset = rospy.get_param('~side_correction_offset', 0.6)
        # 判定单侧锥桶的阈值（局部坐标系下，y > 0.5 视为左侧，y < -0.5 视为右侧）
        self.side_threshold = rospy.get_param('~side_threshold', 0.5)

        # ==================== 状态 ====================
        self.lock = threading.RLock()
        self.cones = []
        self.track_frame = None
        self.last_fit_observation_time = None
        self.last_cone_time = None

        self.pose_received = False
        self.vehicle_x = 0.0
        self.vehicle_y = 0.0
        self.vehicle_yaw = 0.0

        self.centerline_valid = False
        self.centerline_k = 0.0
        # A single row can start the car, but its normal is only a provisional
        # direction. Anchor direction checks to the first accepted two-row fit
        # without rotating the fixed track frame or moving this anchor later.
        self.centerline_reference_k = None
        self.centerline_b = 0.0
        self.centerline_pairs = 0
        self.centerline_last_valid_time = None
        self.planning_started = False

        self.path_seq = 0
        self.last_publish_time = None

        # ==================== ROS ====================
        self.cone_sub = rospy.Subscriber(
            '/cone_map', ConeArray, self.cone_map_callback, queue_size=5)
        self.pose_sub = rospy.Subscriber(
            '/vehicle_pose', PoseStamped, self.vehicle_pose_callback, queue_size=10)
        self.path_pub = rospy.Publisher('/planned_path', Path, queue_size=2)
        self.road_status_pub = rospy.Publisher('/planning/legacy_road_status', String, queue_size=5)

        self.timer = rospy.Timer(
            rospy.Duration(1.0 / self.publish_hz), self.planning_timer)
        self.status_timer = rospy.Timer(
            rospy.Duration(2.0), self.status_timer_callback)

        rospy.loginfo('=' * 80)
        rospy.loginfo('[实时直线规划器] v1.2 - TWO-PAIR')
        rospy.loginfo('车道宽度=%.2fm, 局部规划长度=%.1fm, 发布频率=%.1fHz',
                      self.track_width, self.local_path_length, self.publish_hz)
        rospy.loginfo('前向候选锥桶上限=%d，目标完整锥桶组数=%d',
                      self.max_local_cones, self.max_pair_count)
        rospy.loginfo('核心变化：先按地图坐标寻找完整左右锥桶组，再取最近两组')
        rospy.loginfo('=' * 80)

    # ------------------------------------------------------------------
    # 数据回调
    # ------------------------------------------------------------------
    def cone_map_callback(self, msg):
        stamp = msg.header.stamp.to_sec()
        age = rospy.Time.now().to_sec()-stamp
        if stamp <= 0 or age < -0.1:
            return
        now = time.time()-max(0.0, age)
        cones = []
        for cone in msg.cones:
            if float(cone.confidence) < self.min_cone_confidence:
                continue
            cones.append((float(cone.x), float(cone.y), float(cone.confidence)))

        with self.lock:
            self.cones = cones
            self.last_cone_time = now

    def vehicle_pose_callback(self, msg):
        q = [msg.pose.orientation.x,
             msg.pose.orientation.y,
             msg.pose.orientation.z,
             msg.pose.orientation.w]
        try:
            _, _, yaw = tf_trans.euler_from_quaternion(q)
        except Exception:
            return

        with self.lock:
            self.vehicle_x = float(msg.pose.position.x)
            self.vehicle_y = float(msg.pose.position.y)
            self.vehicle_yaw = float(yaw)
            self.pose_received = True

    # ------------------------------------------------------------------
    # 坐标变换
    # ------------------------------------------------------------------
    def world_to_vehicle(self, wx, wy):
        dx = wx - self.vehicle_x
        dy = wy - self.vehicle_y
        c = math.cos(self.vehicle_yaw)
        s = math.sin(self.vehicle_yaw)
        # 车辆坐标：X前，Y右
        lx = dx * c + dy * s
        ly = -dx * s + dy * c
        return lx, ly

    def vehicle_to_world(self, lx, ly):
        c = math.cos(self.vehicle_yaw)
        s = math.sin(self.vehicle_yaw)
        wx = self.vehicle_x + lx * c - ly * s
        wy = self.vehicle_y + lx * s + ly * c
        return wx, wy

    # ------------------------------------------------------------------
    # 锥桶配对与中心线估计
    # 配对、拟合、历史滤波使用固定赛道坐标，而非当前车辆坐标。
    # ------------------------------------------------------------------
    def select_local_cones(self):
        """Candidate rows use the fixed track frame, never moving body axes."""
        if self.last_cone_time is None or time.time()-self.last_cone_time > self.centerline_timeout:
            return []
        return list(self.cones)

    def pair_cones(self, cones):
        vehicle = (self.vehicle_x, self.vehicle_y, self.vehicle_yaw)
        if self.track_frame is None:
            self.track_frame = bootstrap_frame(cones, vehicle, self)
            if self.track_frame is None:
                return []
            self.yaw_track_world = self.track_frame.yaw
        return paired_centers(cones, self.track_frame, vehicle, self)

    def estimate_centerline(self):
        """Fit d=k*s+b entirely in a fixed track frame (metres, radians)."""
        rows = self.pair_cones(self.select_local_cones())
        if not rows:
            return None
        # Do not integrate/filter the same observation repeatedly at timer rate.
        if self.last_fit_observation_time == self.last_cone_time:
            return None
        self.last_fit_observation_time = self.last_cone_time
        centers = [(p[0], p[1]) for p in rows]
        slope_from_pairs = len(centers) >= 2 and centers[-1][0]-centers[0][0] > .8
        old_k = self.centerline_k if self.centerline_valid else 0.0
        if slope_from_pairs:
            mx = sum(p[0] for p in centers)/len(centers)
            my = sum(p[1] for p in centers)/len(centers)
            raw_k = sum((x-mx)*(y-my) for x,y in centers)/sum((x-mx)**2 for x,y in centers)
            reference_k = getattr(self, 'centerline_reference_k', None)
            comparison_k = 0.0 if reference_k is None else reference_k
            direction_change = abs(math.atan(raw_k)-math.atan(comparison_k))
            if direction_change > math.atan(self.straight_slope_limit):
                # A fixed-frame direction violation rejects the full observation;
                # do not turn rejected heading into a false lateral intercept.
                rospy.logwarn_throttle(1.0, 'Fixed-track slope rejected: raw=%.4f ref=%.4f',
                                       raw_k, comparison_k)
                if not self.centerline_valid:
                    # A rejected startup hypothesis must not permanently lock
                    # a bad reference before the first usable line exists.
                    self.track_frame = None
                    self.yaw_track_world = None
                return None
            if reference_k is None:
                self.centerline_reference_k = raw_k
            change = max(-self.max_slope_change, min(self.max_slope_change, raw_k-old_k))
            k = old_k+self.slope_alpha*change if self.centerline_valid else raw_k
        else:
            k = old_k
        b = median([y-k*x for x,y in centers])
        if self.last_center_b is not None:
            limit = rospy.get_param('~max_center_b_step', .35)
            b = self.last_center_b+max(-limit, min(limit, b-self.last_center_b))
        self.last_center_b = b
        return k, b, len(rows), slope_from_pairs

    def update_centerline(self):
        result = self.estimate_centerline()
        if result is None:
             # 已经启动后，如果长时间没数据，保持上一条有效中心线
            if self.centerline_valid and self.centerline_last_valid_time is not None:
                if time.time() - self.centerline_last_valid_time > 3.0:  # 超过3秒没数据，强制停车或报警
                    rospy.logerr_throttle(2.0, "【严重警告】超过3秒未识别到锥桶，请人工接管或停车！")
                    # 这里可以发布一个空的路径，或者让控制器减速
            return False

        k, b, pairs, slope_from_pairs = result
        a = self.centerline_alpha

        if not self.centerline_valid:
            self.centerline_k = k
            self.centerline_b = b
        else:
            self.centerline_k = (1.0 - a) * self.centerline_k + a * k
            self.centerline_b = (1.0 - a) * self.centerline_b + a * b

        self.centerline_pairs = pairs
        # Preserve supporting measurement age even when fitting the same map again.
        self.centerline_last_valid_time = self.last_cone_time

        if pairs < 2:
            rospy.logwarn_throttle(
                2.0,
                '[实时规划] 当前只有%d组完整左右锥桶；保持/保护中心线 k=%.4f, b=%.3f',
                pairs, self.centerline_k, self.centerline_b)
        elif pairs == 2:
            rospy.loginfo_throttle(
                2.0,
                '[实时规划] 已锁定2组完整左右锥桶，中心线 k=%.4f, b=%.3f',
                self.centerline_k, self.centerline_b)

        first_start = not self.planning_started
        self.planning_started = True
        self.centerline_valid = True

        if first_start:
            rospy.loginfo(
                '✅ [实时规划] 已识别到第一组左右锥桶，开始发布局部中心线'
            )

        return True

    # ------------------------------------------------------------------
    # 路径生成
    # ------------------------------------------------------------------
    def generate_local_path(self):
        """Generate a smooth return path in fixed track axes, then publish map points."""
        updated = self.update_centerline()
        if not self.planning_started:
            return None, False, False
        vs, vd = self.track_frame.local(self.vehicle_x, self.vehicle_y)
        blend = 12.0 if self.centerline_pairs < 2 else self.blend_distance
        points = []
        x = 0.0
        while x <= self.local_path_length+1e-6:
            # k,b are fixed-frame line parameters. Subtract CURRENT vehicle d.
            offset = self.centerline_k*(vs+x)+self.centerline_b-vd
            offset = max(-self.max_center_offset, min(self.max_center_offset, offset))
            alpha = .5-.5*math.cos(math.pi*x/blend) if blend > 1e-6 and x < blend else 1.0
            points.append(self.track_frame.world(vs+x, vd+alpha*offset))
            x += self.path_point_spacing
        window = self.path_smooth_window
        if window >= 3 and len(points) > window:
            half = window//2
            # Use actual slice length, including even configured windows.
            points = [p if i < half or i >= len(points)-half else
                      tuple(sum(q[j] for q in points[i-half:i+half+1])/(2*half+1) for j in (0,1))
                      for i,p in enumerate(points)]
        return points, True, updated

    def publish_path(self, path_points):
        if len(path_points) < 2:
            return

        msg = Path()
        msg.header = Header()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = 'map'

        for i, (x, y) in enumerate(path_points):
            pose = PoseStamped()
            pose.header = msg.header
            pose.pose.position.x = x
            pose.pose.position.y = y
            pose.pose.position.z = 0.0

            if i < len(path_points) - 1:
                dx = path_points[i + 1][0] - x
                dy = path_points[i + 1][1] - y
            else:
                dx = x - path_points[i - 1][0]
                dy = y - path_points[i - 1][1]

            yaw = math.atan2(dy, dx)
            pose.pose.orientation.z = math.sin(yaw / 2.0)
            pose.pose.orientation.w = math.cos(yaw / 2.0)
            msg.poses.append(pose)

        self.path_pub.publish(msg)
        self.path_seq += 1
        self.last_publish_time = time.time()

    # ------------------------------------------------------------------
    # 定时器
    # ------------------------------------------------------------------
    def planning_timer(self, _event):
        with self.lock:
            self._planning_timer(_event)

    def _planning_timer(self, _event):
        with self.lock:
            pose_ok = self.pose_received

        if not pose_ok:
            return

        path_points, path_ready, updated = self.generate_local_path()
        age = (time.time()-self.centerline_last_valid_time
               if self.centerline_last_valid_time is not None else None)
        self.road_status_pub.publish(String(json.dumps(dict(
            stamp=rospy.Time.now().to_sec(),
            valid=bool(self.centerline_valid and age is not None),
            observation_age_s=age, updated=updated, pairs=self.centerline_pairs,
            frame_origin=([self.track_frame.x, self.track_frame.y] if self.track_frame else None),
            frame_yaw=(self.track_frame.yaw if self.track_frame else None),
            direction_reference_k=getattr(self, 'centerline_reference_k', None),
            k=self.centerline_k, b=self.centerline_b))))

        if path_ready and path_points is not None:
            self.publish_path(path_points)
            rospy.loginfo_throttle(
                2.0,
                '[实时规划] 中心线%s: pairs=%d, k=%.4f, b=%.3f, 发布点=%d',
                '已更新' if updated else '保持',
                self.centerline_pairs, self.centerline_k, self.centerline_b, len(path_points))
        else:
            rospy.logwarn_throttle(
                2.0,
                '[实时规划] 等待第一组左右锥桶，暂不发布 /planned_path')

    def status_timer_callback(self, _event):
        with self.lock:
            n_cones = len(self.cones)
        rospy.loginfo(
            '[实时规划状态] pose=%s cones=%d started=%s local=%s pairs=%d k=%.4f b=%.3f path_seq=%d',
            self.pose_received, n_cones, self.planning_started, self.centerline_valid,
            self.centerline_pairs, self.centerline_k, self.centerline_b, self.path_seq)

    def run(self):
        rospy.spin()


if __name__ == '__main__':
    try:
        RealtimeStraightLinePlanner().run()
    except rospy.ROSInterruptException:
        pass
