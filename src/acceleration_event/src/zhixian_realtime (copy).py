#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
FSAE Formula Student - Real-time Straight Line Planner v2.0 - Stable Two-Pair

功能：
1. 持续订阅 /cone_map 和 /vehicle_pose
2. 使用车辆前方最多4个锥桶，实时估计局部赛道中心线
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

import numpy as np
import rospy
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Header
import tf.transformations as tf_trans
from mapping.msg import ConeArray


class RealtimeStraightLinePlanner(object):
    def __init__(self):
        rospy.init_node('straight_line_planner_realtime', anonymous=False)

        # ==================== 赛道参数 ====================
        self.track_width = rospy.get_param('~track_width', 3.0)
        self.total_length = rospy.get_param('~total_length', 175.0)

        # 局部规划窗口：由于雷达最多稳定提供4个前方锥桶，没必要一次规划175m
        self.local_path_length = rospy.get_param('~local_path_length', 30.0)
        self.path_point_spacing = rospy.get_param('~path_point_spacing', 0.5)
        self.blend_distance = rospy.get_param('~blend_distance', 6.0)

        # ==================== 锥桶筛选 ====================
        self.min_cone_confidence = rospy.get_param('~min_cone_confidence', 0.25)
        self.local_min_x = rospy.get_param('~local_min_x', 0.5)
        self.local_max_x = rospy.get_param('~local_max_x', 24.0)
        self.local_max_abs_y = rospy.get_param('~local_max_abs_y', 4.0)
        # 保留更多候选，真正的选择在 pair_cones() 中完成。
        self.max_local_cones = rospy.get_param('~max_local_cones', 20)
        self.max_pair_count = rospy.get_param('~max_pair_count', 2)
        self.pair_row_min_spacing = rospy.get_param('~pair_row_min_spacing', 1.5)
        self.track_center_y = rospy.get_param('~track_center_y', 0.0)
        self.track_forward_min = rospy.get_param('~track_forward_min', 0.5)
        self.track_forward_max = rospy.get_param('~track_forward_max', 25.0)


        # 左右配对：理想横向距离约3m；同排锥桶的纵向位置应接近
        self.pair_distance_min = rospy.get_param('~pair_distance_min', 2.3)
        self.pair_distance_max = rospy.get_param('~pair_distance_max', 3.7)
        self.pair_max_longitudinal_diff = rospy.get_param('~pair_max_longitudinal_diff', 1.5)
        self.max_centerline_slope = rospy.get_param('~max_centerline_slope', 0.25)

        # 直线赛道稳定保护：
        # 1) 新拟合的中心线斜率不能过大；
        # 2) 相邻周期斜率不能突然跳变；
        # 3) 只有一个左右锥桶对时，保持上一周期的斜率，只更新横向位置。
        self.straight_slope_limit = rospy.get_param('~straight_slope_limit', 0.06)
        self.max_slope_change = rospy.get_param('~max_slope_change', 0.02)
        self.slope_alpha = rospy.get_param('~slope_alpha', 0.12)

        # ==================== 稳定性 ====================
        self.centerline_alpha = rospy.get_param('~centerline_alpha', 0.15)
        self.last_center_b = None
        self.centerline_timeout = rospy.get_param('~centerline_timeout', 0.50)
        self.max_center_offset = rospy.get_param('~max_center_offset', 1.5)
        self.publish_hz = rospy.get_param('~publish_hz', 10.0)

        # ==================== 路径滤波参数 ====================
        # 旧路径权重（越大越平滑，响应越慢）
        self.old_path_weight = rospy.get_param('~old_path_weight', 0.85)
        # 新路径权重（越小越能抑制抖动）
        self.new_path_weight = rospy.get_param('~new_path_weight', 0.15)
        
        # 确保两者之和为1
        self.centerline_alpha = self.new_path_weight  # 替换原有的 0.15
        # 路径点滑动平均滤波窗口大小（建议奇数）
        self.path_smooth_window = rospy.get_param('~path_smooth_window', 3)

        # ==================== 单侧修正参数 ====================
        # 当只识别到单侧锥桶时，向另一侧修正的额外偏移量(m)
        self.side_correction_offset = rospy.get_param('~side_correction_offset', 0.6)
        # 判定单侧锥桶的阈值（局部坐标系下，y > 0.5 视为左侧，y < -0.5 视为右侧）
        self.side_threshold = rospy.get_param('~side_threshold', 0.5)

        # ==================== 状态 ====================
        self.lock = threading.Lock()
        self.cones = []
        self.last_cone_time = None

        self.pose_received = False
        self.vehicle_x = 0.0
        self.vehicle_y = 0.0
        self.vehicle_yaw = 0.0

        self.centerline_valid = False
        self.centerline_k = 0.0
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
        now = msg.header.stamp.to_sec() if not msg.header.stamp.is_zero() else time.time()
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
    # 注意：左右配对基于 map 坐标的赛道几何，而不是当前车辆 yaw。
    # ------------------------------------------------------------------
    def select_local_cones(self):
        with self.lock:
            cones = list(self.cones)
            last_time = self.last_cone_time
            vehicle_x = self.vehicle_x
            vehicle_y = self.vehicle_y
            vehicle_yaw = self.vehicle_yaw

        if last_time is None:
            return []
        if time.time() - last_time > self.centerline_timeout:
            return []

        # 重要变化：
        # 配对不再依赖“当前车辆坐标系下 y 一正一负”。
        # 车辆一旦 yaw 很大，同一组真实左右锥桶可能在车辆坐标里落到同一侧，
        # 这样原算法就会把完整的一组锥桶拆散。
        #
        # 直线赛道的左右关系在 map 坐标中是稳定的，因此先按 map X/Y
        # 找车辆前方的赛道锥桶，再转换到车辆坐标供中心线拟合。
        local = []

        world_min_x = vehicle_x + self.track_forward_min
        world_max_x = vehicle_x + self.track_forward_max

        for wx, wy, conf in cones:
            if wx < world_min_x or wx > world_max_x:
                continue

            # 直线赛道中心附近的历史锥桶才是候选。
            if abs(wy - self.track_center_y) > (self.track_width + 2.0):
                continue

            lx, ly = self.world_to_vehicle(wx, wy)

            local.append((
                lx, ly,
                wx, wy,
                conf
            ))

        # 地图坐标的赛道纵向顺序优先，保证最近两组完整锥桶先被考虑。
        local.sort(key=lambda p: (p[2], abs(p[3] - self.track_center_y)))
        return local[:self.max_local_cones]

    def pair_cones(self, local):
        candidates = []
        n = len(local)

        # 1. 正常左右配对逻辑
        for i in range(n):
            lx1, ly1, wx1, wy1, _ = local[i]
            for j in range(i + 1, n):
                lx2, ly2, wx2, wy2, _ = local[j]

                side1 = wy1 - self.track_center_y
                side2 = wy2 - self.track_center_y

                if side1 * side2 >= 0.0:
                    # 调试日志：看看是不是因为同侧被拒
                    rospy.loginfo_throttle(2.0, "【配对失败】同侧: side1=%.2f, side2=%.2f", side1, side2)
                    continue

                world_dx = abs(wx1 - wx2)
                if world_dx > self.pair_max_longitudinal_diff:
                    # 调试日志：纵向差太大
                    rospy.loginfo_throttle(2.0, "【配对失败】纵向差太大: %.2f > %.2f", world_dx, self.pair_max_longitudinal_diff)
                    continue

                world_d = math.hypot(wx1 - wx2, wy1 - wy2)
                if (world_d < self.pair_distance_min or world_d > self.pair_distance_max):
                    # 调试日志：横向距离不合格
                    rospy.logwarn_throttle(2.0, "【配对失败】横向距离不合格: %.2f (需要 %.2f ~ %.2f)", world_d, self.pair_distance_min, self.pair_distance_max)
                    continue

                # 计算中心点（同时保留世界坐标和车辆局部坐标）
                center_wx = 0.5 * (wx1 + wx2)
                center_wy = 0.5 * (wy1 + wy2)
                center_lx = 0.5 * (lx1 + lx2)
                center_ly = 0.5 * (ly1 + ly2)

                score = (
                    abs(world_d - self.track_width)
                    + 0.35 * world_dx
                    + 0.01 * max(0.0, center_wx - self.vehicle_x)
                )

                candidates.append((
                    score,
                    center_wx,
                    center_wy,
                    center_lx,
                    center_ly,
                    i,
                    j,
                    world_d
                ))

        # 2. 对候选对进行排序和去重选择
        candidates.sort(key=lambda item: (item[0], item[1]))

        selected = []
        used = set()

        for item in candidates:
            _, center_wx, center_wy, center_lx, center_ly, i, j, width = item

            if i in used or j in used:
                continue

            duplicate_row = False
            for chosen in selected:
                if abs(center_wx - chosen[1]) < self.pair_row_min_spacing:
                    duplicate_row = True
                    break

            if duplicate_row:
                continue

            selected.append(item)
            used.add(i)
            used.add(j)

            if len(selected) >= self.max_pair_count:
                break

        # 3. 单侧锥桶修正兜底逻辑（只有在完全配对不到时才启用）
        if not selected:
            # 按车辆局部坐标 X 轴（前向）排序，找最近的有效锥桶
            local_sorted = sorted(local, key=lambda p: p[0])
            for lx, ly, wx, wy, conf in local_sorted:
                if lx < self.track_forward_min:
                    continue  # 太近的不要

                # 判断属于左侧还是右侧（使用局部坐标 Y 轴）
                if ly > self.side_threshold:
                    # 只有左侧锥桶 -> 车辆向右修正
                    virtual_center_ly = ly - (self.track_width / 2.0) - self.side_correction_offset
                    virtual_center_wx, virtual_center_wy = self.vehicle_to_world(lx, virtual_center_ly)
                    
                    selected.append((0.0, virtual_center_wx, virtual_center_wy, lx, virtual_center_ly, 0, 0, self.track_width))
                    rospy.logwarn_throttle(1.0, "【单侧修正】仅识别到左侧锥桶，规划中心向右偏移 %.2f m", self.side_correction_offset)
                    break
                    
                elif ly < -self.side_threshold:
                    # 只有右侧锥桶 -> 车辆向左修正
                    virtual_center_ly = ly + (self.track_width / 2.0) + self.side_correction_offset
                    virtual_center_wx, virtual_center_wy = self.vehicle_to_world(lx, virtual_center_ly)
                    
                    selected.append((0.0, virtual_center_wx, virtual_center_wy, lx, virtual_center_ly, 0, 0, self.track_width))
                    rospy.logwarn_throttle(1.0, "【单侧修正】仅识别到右侧锥桶，规划中心向左偏移 %.2f m", self.side_correction_offset)
                    break

        # 最终按赛道前向顺序排列
        selected.sort(key=lambda item: item[1])
        return selected

    def estimate_centerline(self):
        local = self.select_local_cones()
        if len(local) < 2:
            return None

        pairs = self.pair_cones(local)
        if not pairs:
            return None

        centers = []
        for _, _, _, center_lx, center_ly, _, _, width in pairs:
            centers.append((
                float(center_lx),
                float(center_ly),
                float(width),
            ))

        centers.sort(key=lambda p: p[0])

        # 只有一组左右锥桶：
        # 不重新估计道路方向，只根据当前中心点更新横向位置。
        if len(centers) < 2 or abs(centers[1][0] - centers[0][0]) <= 0.8:
            if self.centerline_valid:
                k = self.centerline_k
            else:
                k = 0.0

            x0, y0, _ = centers[0]
            b = y0 - k * x0
            return float(k), float(b), len(centers), False

        # 至少两组左右锥桶：先正常拟合。
        xs = np.asarray([p[0] for p in centers], dtype=float)
        ys = np.asarray([p[1] for p in centers], dtype=float)
        try:
            raw_k, _ = np.polyfit(xs, ys, 1)
        except Exception:
            raw_k = self.centerline_k if self.centerline_valid else 0.0

        raw_k = float(raw_k)

        # 直线赛道不允许中心线方向突然出现明显变化。
        # 若当前没有有效中心线，则第一次也只接受很小的斜率。
        if abs(raw_k) > self.straight_slope_limit:
            if self.centerline_valid:
                k = self.centerline_k
            else:
                k = max(-self.straight_slope_limit,
                        min(self.straight_slope_limit, raw_k))

            rospy.logwarn_throttle(
                1.0,
                '[实时规划保护] 拒绝异常中心线斜率 raw_k=%.4f，使用 k=%.4f',
                raw_k, k)
        elif self.centerline_valid and abs(raw_k - self.centerline_k) > self.max_slope_change:
            # 不允许相邻周期的道路方向突然跳变。
            direction = 1.0 if raw_k >= self.centerline_k else -1.0
            k = self.centerline_k + direction * self.max_slope_change
            rospy.logwarn_throttle(
                1.0,
                '[实时规划保护] 限制中心线斜率跳变 raw_k=%.4f old_k=%.4f new_k=%.4f',
                raw_k, self.centerline_k, k)
        else:
            if self.centerline_valid:
                k = ((1.0 - self.slope_alpha) * self.centerline_k
                     + self.slope_alpha * raw_k)
            else:
                k = raw_k

        # 用最终接受的斜率重新求截距，确保中心线真正经过当前测量中心。
        b_values = [p[1] - k * p[0] for p in centers]
        b = float(np.median(np.asarray(b_values, dtype=float)))

        # v2稳定保护：两组锥桶视距下，中心线横向偏移不能瞬间跳变
        if hasattr(self, 'last_center_b'):
            max_b_step = rospy.get_param('~max_center_b_step', 0.35)
            db = b - self.last_center_b
            if abs(db) > max_b_step:
                b = self.last_center_b + math.copysign(max_b_step, db)
        self.last_center_b = b

        return float(k), float(b), len(centers), True

    def update_centerline(self):
        result = self.estimate_centerline()
        if result is None:
            # 启动前：没有第一组左右锥桶，不能发布路径。
            # 启动后：保留上一条有效中心线，避免短时丢雷达导致路径跳回 y=0。
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
        self.centerline_last_valid_time = time.time()

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
        """在当前车辆坐标系中生成局部中心线，再转换回map坐标。"""
        updated = self.update_centerline()

        # 第一次看到有效锥桶对之前，严禁生成/发布 y=0 保底路径。
        if not self.planning_started:
            return None, False, False

        # 已经启动后，即使当前周期暂时没更新到锥桶，也沿用上一条有效中心线。
        k = self.centerline_k
        b = self.centerline_b

        path_points = []
        x = 0.0
        while x <= self.local_path_length + 1e-6:
            center_y = k * x + b
            if abs(center_y) > self.max_center_offset:
                center_y = max(-self.max_center_offset,
                               min(self.max_center_offset, center_y))

            if self.blend_distance > 1e-6 and x < self.blend_distance:
                alpha = 0.5 - 0.5 * math.cos(math.pi * x / self.blend_distance)
                y = alpha * center_y
            else:
                y = center_y

            wx, wy = self.vehicle_to_world(x, y)
            path_points.append((wx, wy))
            x += self.path_point_spacing

        # ==================== 新增：路径点滑动平均滤波 ====================
        if self.path_smooth_window >= 3 and len(path_points) > self.path_smooth_window:
            smoothed_points = []
            half_win = self.path_smooth_window // 2
            for i in range(len(path_points)):
                # 边缘点不滤波，保持原始值
                if i < half_win or i >= len(path_points) - half_win:
                    smoothed_points.append(path_points[i])
                else:
                    # 对窗口内的点求均值
                    win_x, win_y = 0.0, 0.0
                    for j in range(i - half_win, i + half_win + 1):
                        win_x += path_points[j][0]
                        win_y += path_points[j][1]
                    smoothed_points.append((
                        win_x / float(self.path_smooth_window),
                        win_y / float(self.path_smooth_window)
                    ))
            path_points = smoothed_points
        # ==================================================================

        return path_points, True, updated

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
            pose_ok = self.pose_received

        if not pose_ok:
            return

        path_points, path_ready, updated = self.generate_local_path()

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
