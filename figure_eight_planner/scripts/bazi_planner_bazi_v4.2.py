#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
八字绕环规划器 v4.0

核心策略：
  1. FIND_START：寻找起止线附近代表启动的两个平行锥桶
  2. START：仿真自动启动；真实车可关闭 auto_start 后接 RES Go 接口
  3. RUN：右圆 2 圈 -> 左圆 2 圈；行驶中用最近平行锥桶实时修正中心线
  4. EXIT：第四圈后沿进入时同向从交叉点驶出，并在 25m 范围内停车
  5. FINISH：发布 race_status=1，由 tracking 触发 2000 E-STOP

坐标系：X=前，Y=右，yaw=数学正角（右转为正）。
"""

from __future__ import print_function
import rospy
import math
import numpy as np
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped, PoseArray
from std_msgs.msg import Float64, UInt8
from visualization_msgs.msg import Marker, MarkerArray
import tf.transformations as tf_trans



# ============================================================
# 赛道规格
# ============================================================
R_INNER = 7.625
R_OUTER = 10.625
R_RACE = (R_INNER + R_OUTER) / 2.0

S_WAIT_GATE = 0
S_WAIT_GO = 1
S_RUN = 2
S_DONE = 3

STAGE_IDLE = 0
STAGE_RIGHT_LAP1 = 1
STAGE_RIGHT_LAP2 = 2
STAGE_LEFT_LAP3 = 3
STAGE_LEFT_LAP4 = 4
STAGE_EXIT = 5
STAGE_FINISH = 6


def wrap_pi(rad):
    while rad > math.pi:
        rad -= 2.0 * math.pi
    while rad < -math.pi:
        rad += 2.0 * math.pi
    return rad


class BaziPlannerV4(object):
    def __init__(self):
        rospy.init_node('bazi_planner', anonymous=False)

        # ---------- 基础 ----------
        self.rate = rospy.get_param('~rate', 20.0)
        self.dt = 1.0 / self.rate
        self.path_spacing = rospy.get_param('~path_spacing', 0.4)

        # ---------- 双锥桶启动 ----------
        self.trigger_min_range = rospy.get_param('~trigger_min_range', 4.0)
        self.trigger_max_range = rospy.get_param('~trigger_max_range', 18.0)
        self.trigger_lateral_limit = rospy.get_param('~trigger_lateral_limit', 6.0)
        self.gate_width_min = rospy.get_param('~gate_width_min', 2.4)
        self.gate_width_max = rospy.get_param('~gate_width_max', 3.6)
        self.gate_x_diff_max = rospy.get_param('~gate_x_diff_max', 1.5)
        self.gate_confirm_frames = rospy.get_param('~gate_confirm_frames', 4)
        self.gate_center_stability = rospy.get_param('~gate_center_stability', 0.8)

        # ---------- 局部锥桶 ----------
        self.cone_min_confidence = rospy.get_param('~cone_min_confidence', 0.15)
        self.local_min_x = rospy.get_param('~local_min_x', 1.0)
        self.local_max_x = rospy.get_param('~local_max_x', 18)
        self.local_max_abs_y = rospy.get_param('~local_max_abs_y', 6.0)
        self.lidar_region_enabled = rospy.get_param('~lidar_region_enabled', False)
        self.lidar_loss_grace = float(rospy.get_param('~lidar_loss_grace', 0.5))
        self.lidar_loss_speed = float(rospy.get_param('~lidar_loss_speed', 1.0))
        if (math.isnan(self.lidar_loss_grace) or math.isinf(self.lidar_loss_grace) or
                math.isnan(self.lidar_loss_speed) or math.isinf(self.lidar_loss_speed) or
                self.lidar_loss_grace < 0 or self.lidar_loss_speed < 0):
            raise ValueError('Invalid lidar loss limits')
        self.lidar_missing_since = None
        self.lidar_region_range = float(rospy.get_param('~lidar_region_range', 5.0))
        self.lidar_region_half_width = float(rospy.get_param('~lidar_region_half_width', 4.0))
        self.lidar_offset_x = float(rospy.get_param('~lidar_offset_x', 2.11552))
        self.lidar_offset_y = float(rospy.get_param('~lidar_offset_y', -0.02212))
        self.lidar_yaw_offset_deg = float(rospy.get_param('~lidar_yaw_offset_deg', 0.5))
        if (not all((not math.isnan(v) and not math.isinf(v)) for v in (self.lidar_region_range, self.lidar_region_half_width,
                                              self.lidar_offset_x, self.lidar_offset_y, self.lidar_yaw_offset_deg)) or
                min(self.lidar_region_range, self.lidar_region_half_width) <= 0):
            raise ValueError('Invalid local lidar region')
        self.local_pair_width_min = rospy.get_param('~local_pair_width_min', 2.4)
        self.local_pair_width_max = rospy.get_param('~local_pair_width_max', 3.6)
        self.local_pair_xdiff_max = rospy.get_param('~local_pair_xdiff_max', 1.8)
        self.local_max_pairs = rospy.get_param('~local_max_pairs', 2)
        self.cone_blend_weight_one_pair = rospy.get_param('~cone_blend_weight_one_pair', 0.08)
        self.cone_blend_weight_two_pairs = rospy.get_param('~cone_blend_weight_two_pairs', 0.25)
        self.max_center_correction = rospy.get_param('~max_center_correction', 1.0)
        # 雷达仅作为局部修正，不改变全局预设八字路径
        self.radar_correction_gain = rospy.get_param('~radar_correction_gain', 0.25)
        self.radar_single_side_gain = rospy.get_param('~radar_single_side_gain', 0.08)
        self.max_yaw_correction = rospy.get_param('~max_yaw_correction', 0.25)

        # ---------- 路径 ----------
        self.front_overhang = rospy.get_param('~front_overhang',0.7)
        self.entry_length = rospy.get_param('~entry_length', 15.0 + self.front_overhang)
        self.exit_length = rospy.get_param('~exit_length', 25.0)
        self.finish_stop_distance = rospy.get_param('~finish_stop_distance', 23.0)
        self.auto_start = rospy.get_param('~auto_start', True)

        # ---------- WAIT_GATE 前进段 ----------
        self.approach_speed = rospy.get_param('~approach_speed', 1.0)
        self.approach_max_length = rospy.get_param('~approach_max_length', 8.0)
        self.wait_gate_start_x = None
        self.wait_gate_start_y = None

        # ---------- 速度 ----------
        self.max_speed = rospy.get_param('~max_speed', 3.5)
        self.min_speed = rospy.get_param('~min_speed', 0.5)
        self.curve_speed = rospy.get_param('~curve_speed', 3.0)
        self.entry_speed = rospy.get_param('~entry_speed', 3.0)
        self.a_max = rospy.get_param('~a_max', 1.5)
        self.a_min = rospy.get_param('~a_min', 2.5)
        self.end_decel_dist = rospy.get_param('~end_decel_dist', 15.0)

        # ---------- 状态 ----------
        self.state = S_WAIT_GATE
        self.stage = STAGE_IDLE
        self.finish_s = None
        self.start_gate = None
        self.vx = 0.0
        self.vy = 0.0
        self.vyaw = 0.0
        self.veh_speed = 0.0
        self.pose_received = False
        self.cones = []

        self.preset_path = None
        self.arc = None
        self.total_length = 0.0
        self.progress_s = 0.0

        # ---------- 双锥桶门确认 ----------
        self.gate_confirm_count = 0
        self.last_gate_center = None
        self.last_gate_pair = None

        # ---------- ROS ----------
        rospy.Subscriber('/vehicle_pose', PoseStamped, self._pose_cb, queue_size=10)
        rospy.Subscriber('/vehicle_speed', Float64, self._speed_cb, queue_size=10)
        rospy.Subscriber('/cone_map', PoseArray, self._cone_cb, queue_size=10)

        self.path_pub = rospy.Publisher('/planned_path', Path, queue_size=10)
        self.spd_pub = rospy.Publisher('/target_speed', Float64, queue_size=10)
        self.stat_pub = rospy.Publisher('/race_status', UInt8, queue_size=10)
        self.marker_pub = rospy.Publisher('/bazi_debug', MarkerArray, queue_size=10)

        rospy.Timer(rospy.Duration(self.dt), self._update)
        rospy.loginfo("Bazi Planner v4.0 started")

    # ============================================================
    # 回调
    # ============================================================
    def _pose_cb(self, msg):
        self.vx = msg.pose.position.x
        self.vy = msg.pose.position.y
        q = msg.pose.orientation
        n = math.sqrt(q.x*q.x + q.y*q.y + q.z*q.z + q.w*q.w)
        if n > 1e-6:
            qx, qy, qz, qw = q.x/n, q.y/n, q.z/n, q.w/n
        else:
            qx, qy, qz, qw = q.x, q.y, q.z, q.w
        _, _, self.vyaw = tf_trans.euler_from_quaternion([qx, qy, qz, qw])

        if not self.pose_received:
            # 第一帧记录 WAIT_GATE 起点
            self.wait_gate_start_x = self.vx
            self.wait_gate_start_y = self.vy

        self.pose_received = True

    def _speed_cb(self, msg):
        self.veh_speed = msg.data


    def _cone_cb(self, msg):
        self.cones = [
            (p.position.x, p.position.y, p.position.z)
            for p in msg.poses
            if p.position.z >= self.cone_min_confidence
        ]

    # ============================================================
    # 坐标转换
    # ============================================================
    def _map_to_vehicle(self, wx, wy):
        dx = wx - self.vx
        dy = wy - self.vy
        c = math.cos(self.vyaw)
        s = math.sin(self.vyaw)
        lx = dx * c + dy * s
        ly = -dx * s + dy * c
        return lx, ly

    def _vehicle_to_map(self, lx, ly):
        c = math.cos(self.vyaw)
        s = math.sin(self.vyaw)
        return (
            self.vx + lx * c - ly * s,
            self.vy + lx * s + ly * c
        )

    def _in_lidar_planning_region(self, vehicle_x, vehicle_y):
        # Inverse of mapper installation rotation, relative to nose-mounted lidar.
        dx, dy = vehicle_x-self.lidar_offset_x, vehicle_y-self.lidar_offset_y
        angle = math.radians(self.lidar_yaw_offset_deg)
        x = dx*math.cos(angle) + dy*math.sin(angle)
        y = -dx*math.sin(angle) + dy*math.cos(angle)
        return (x > 0.0 and abs(y) <= self.lidar_region_half_width and
                x*x+y*y <= self.lidar_region_range*self.lidar_region_range)

    def _local_cones(self):
        out = []
        for idx, (wx, wy, conf) in enumerate(self.cones):
            lx, ly = self._map_to_vehicle(wx, wy)
            if self.lidar_region_enabled and not self._in_lidar_planning_region(lx, ly):
                continue
            if self.local_min_x <= lx <= self.local_max_x and abs(ly) <= self.local_max_abs_y:
                out.append({'idx': idx, 'x': lx, 'y': ly, 'conf': conf,
                            'wx': wx, 'wy': wy})
        return out

    def _limit_speed_for_lidar(self, speed, now):
        # A confirmed local pair may come from fresh detections or decaying history.
        if not self.lidar_region_enabled:
            return speed
        if self._find_local_center_pairs():
            self.lidar_missing_since = None
            return speed
        if self.lidar_missing_since is None:
            self.lidar_missing_since = now
        if now - self.lidar_missing_since < self.lidar_loss_grace:
            return speed
        return min(speed, self.lidar_loss_speed)

    # ============================================================
    # 双锥桶门识别
    # ============================================================
    def _find_gate_candidates(self):
        cones = self._local_cones()
        candidates = []
        # 用位置计算自 WAIT_GATE 起点以来前进的距离
        if self.wait_gate_start_x is not None:
            traveled = math.hypot(self.vx - self.wait_gate_start_x,
                                  self.vy - self.wait_gate_start_y)
        else:
            traveled = 0.0

         #门在车辆前方的期望距离 =  入口长度 - 已走距离
        expected_lx = self.entry_length - traveled
        
        for i in range(len(cones)):
            for j in range(i + 1, len(cones)):
                a = cones[i]
                b = cones[j]
                dx = a['x'] - b['x']
                dy = a['y'] - b['y']
                width = math.hypot(dx, dy)
                x_diff = abs(a['x'] - b['x'])
                center_x = 0.5 * (a['x'] + b['x'])
                center_y = 0.5 * (a['y'] + b['y'])

                if not (self.gate_width_min <= width <= self.gate_width_max):
                    continue
                if x_diff > self.gate_x_diff_max:
                    continue
                if not (self.trigger_min_range <= center_x <= self.trigger_max_range):
                    continue
                if abs(center_y) > self.trigger_lateral_limit:
                    continue
                # 门中心必须接近"期望距离"
                if abs(center_x - expected_lx) > 3.0:
                    continue

                width_error = abs(width - (R_OUTER - R_INNER))
                # 在等待发车阶段，优先远端且几何上最像起止门的一对。
                score = (abs(center_x - expected_lx) * 1.0 + 
                         width_error * 3.0 + 
                         abs(x_diff) * 0.5 + 
                         abs(center_y) * 0.2)
                         
                candidates.append({
                    'i': i,
                    'j': j,
                    'center': (center_x, center_y),
                    'width': width,
                    'score': score
                })

        candidates.sort(key=lambda x: x['score'])
        return candidates

    def _confirm_start_gate(self):
        candidates = self._find_gate_candidates()
        if not candidates:
            self.gate_confirm_count = 0
            self.last_gate_center = None
            self.last_gate_pair = None
            return None

        best = candidates[0]
        cx, cy = best['center']

        if self.last_gate_center is not None:
            d = math.hypot(cx - self.last_gate_center[0], cy - self.last_gate_center[1])
            if d <= self.gate_center_stability:
                self.gate_confirm_count += 1
            else:
                self.gate_confirm_count = 1
        else:
            self.gate_confirm_count = 1

        self.last_gate_center = best['center']
        self.last_gate_pair = best

        rospy.loginfo_throttle(
            1.0,
            "[GATE] pair width=%.2f center=(%.2f, %.2f) confirm=%d/%d",
            best['width'], cx, cy,
            self.gate_confirm_count, self.gate_confirm_frames)

        if self.gate_confirm_count >= self.gate_confirm_frames:
            return best
        return None

    def _gate_map_geometry(self, gate):
        local_cones = self._local_cones()
        if gate['i'] >= len(local_cones) or gate['j'] >= len(local_cones):
            return None

        a = local_cones[gate['i']]
        b = local_cones[gate['j']]
        center_lx = 0.5 * (a['x'] + b['x'])
        center_ly = 0.5 * (a['y'] + b['y'])
        center_wx, center_wy = self._vehicle_to_map(center_lx, center_ly)

        # 两锥桶连线是横向方向，取其垂线作为赛道前进方向。
        vx = b['x'] - a['x']
        vy = b['y'] - a['y']
        # 一种垂向量：(-vy, vx)
        dir_x = -vy
        dir_y = vx
        norm = math.hypot(dir_x, dir_y)
        if norm < 1e-6:
            return None
        dir_x /= norm
        dir_y /= norm

        # 朝向当前车辆前方
        if dir_x < 0.0:
            dir_x = -dir_x
            dir_y = -dir_y

        gate_yaw = math.atan2(dir_y, dir_x)
        return center_wx, center_wy, gate_yaw

    # ============================================================
    # 预设八字路径
    # ============================================================
    def _build_local_path(self):
        """X前/Y右。两圆分别完整绕 2 圈，且所有路径点保持约 path_spacing。"""
        path = []
        sp = self.path_spacing
        R = R_RACE
        circle_arc = 4.0 * math.pi * R
        n_circle = max(20, int(round(circle_arc / sp)))

        # 入口：从起点线后方进入起点线中心
        n_entry = max(2, int(round(self.entry_length / sp)))
        for i in range(n_entry):
            path.append((-self.entry_length + i * self.entry_length / float(n_entry - 1), 0.0))

        # 右侧圆（物理右侧 = +Y）：cx, cy = 0.0, +R，顺时针 = 角度递增
        cx, cy = 0.0, R
        for i in range(1, n_circle + 1):
            theta = -math.pi / 2.0 + 4.0 * math.pi * i / float(n_circle)
            path.append((cx + R * math.cos(theta), cy + R * math.sin(theta)))

        # 左侧圆（物理左侧 = -Y）：cx, cy = 0.0, -R，逆时针 = 角度递减
        cx, cy = 0.0, -R
        for i in range(1, n_circle + 1):
            theta = math.pi / 2.0 - 4.0 * math.pi * i / float(n_circle)
            path.append((cx + R * math.cos(theta), cy + R * math.sin(theta)))

        # 出口
        n_exit = max(2, int(round(self.exit_length / sp)))
        for i in range(1, n_exit + 1):
            x = i * self.exit_length / float(n_exit)
            path.append((x, 0.0))

        return path

    def _generate_preset_path(self, anchor_wx, anchor_wy, anchor_yaw):
        local = self._build_local_path()
        c = math.cos(anchor_yaw)
        s = math.sin(anchor_yaw)

        self.preset_path = []
        for xl, yl in local:
            mx = anchor_wx + xl * c - yl * s
            my = anchor_wy + xl * s + yl * c
            self.preset_path.append((mx, my))

        self.arc = [0.0]
        for i in range(1, len(self.preset_path)):
            dx = self.preset_path[i][0] - self.preset_path[i - 1][0]
            dy = self.preset_path[i][1] - self.preset_path[i - 1][1]
            self.arc.append(self.arc[-1] + math.hypot(dx, dy))
        self.total_length = self.arc[-1]

        # 规则：第四圈完成后从交叉点沿进入时同向驶出，并在计时线后 25m 内停车。
        # 这里保留 2m 工程余量，默认在出口段 23m 处发出完成状态。
        two_laps = 2.0 * math.pi * R_RACE
        self.finish_s = min(
            self.total_length,
            self.entry_length + 4.0 * two_laps + self.finish_stop_distance
        )

        best_s = 0.0
        best_d = float('inf')
        for i, (x, y) in enumerate(self.preset_path):
            d = math.hypot(x - self.vx, y - self.vy)
            if d < best_d:
                best_d = d
                best_s = self.arc[i]
        self.progress_s = best_s

        rospy.loginfo("[PATH] preset generated: points=%d, length=%.2fm, s0=%.2fm",
                      len(self.preset_path), self.total_length, self.progress_s)

    # ============================================================
    # 路径工具
    # ============================================================
    def _point_at_s(self, s):
        s = max(0.0, min(s, self.total_length))
        i = np.searchsorted(self.arc, s)
        i = min(i, len(self.preset_path) - 1)
        if i == 0:
            return self.preset_path[0]
        sa, sb = self.arc[i - 1], self.arc[i]
        if sb - sa < 1e-9:
            return self.preset_path[i]
        f = (s - sa) / (sb - sa)
        x = self.preset_path[i - 1][0] + (self.preset_path[i][0] - self.preset_path[i - 1][0]) * f
        y = self.preset_path[i - 1][1] + (self.preset_path[i][1] - self.preset_path[i - 1][1]) * f
        return x, y

    def _segment(self, s0, length):
        seg = []
        s_end = min(s0 + length, self.total_length)
        s = s0
        while s <= s_end + 1e-6:
            seg.append(self._point_at_s(s))
            s += self.path_spacing
        if not seg or self.arc[-1] - s_end > 1e-6:
            seg.append(self._point_at_s(s_end))
        return seg

    def _update_progress(self):
        s_lo = max(0.0, self.progress_s - 2.0)
        s_hi = min(self.total_length, self.progress_s + 15.0)
        i0 = max(0, np.searchsorted(self.arc, s_lo) - 1)
        i1 = min(len(self.preset_path) - 1, np.searchsorted(self.arc, s_hi) + 1)

        best_s = self.progress_s
        best_d = float('inf')
        for i in range(i0, i1 + 1):
            d = math.hypot(self.preset_path[i][0] - self.vx,
                           self.preset_path[i][1] - self.vy)
            if d < best_d:
                best_d = d
                best_s = self.arc[i]

        if best_s >= self.progress_s - 0.5:
            self.progress_s = max(self.progress_s, best_s)
        return best_d

    # ============================================================
    # 局部锥桶中心线
    # ============================================================
    def _find_local_center_pairs(self):
        cones = self._local_cones()
        candidates = []

        for i in range(len(cones)):
            for j in range(i + 1, len(cones)):
                a = cones[i]
                b = cones[j]
                width = math.hypot(a['x'] - b['x'], a['y'] - b['y'])
                xdiff = abs(a['x'] - b['x'])
                if not (self.local_pair_width_min <= width <= self.local_pair_width_max):
                    continue
                if xdiff > self.local_pair_xdiff_max:
                    continue

                cx = 0.5 * (a['x'] + b['x'])
                cy = 0.5 * (a['y'] + b['y'])
                candidates.append({
                    'i': i,
                    'j': j,
                    'x': cx,
                    'y': cy,
                    'width': width,
                    'score': abs(width - 3.0) + 0.25 * abs(xdiff)
                })

        candidates.sort(key=lambda p: (p['x'], p['score']))

        selected = []
        used = set()
        for p in candidates:
            if p['i'] in used or p['j'] in used:
                continue
            if selected and abs(p['x'] - selected[-1]['x']) < 1.5:
                continue
            selected.append(p)
            used.add(p['i'])
            used.add(p['j'])
            if len(selected) >= self.local_max_pairs:
                break
        return selected

    def _build_fused_segment(self, s0, lookahead):
        """预设八字路径为主，雷达只提供小幅横向误差修正。
        适应真实雷达：最多两组锥桶，并允许漏检。
        """
        base_seg = self._segment(s0, lookahead)
        if len(base_seg) < 2:
            return base_seg, 0.0

        pairs = self._find_local_center_pairs()
        if not pairs:
            return base_seg, 0.0

        # 计算车辆坐标系下道路中心误差
        if len(pairs) >= 2:
            y_err = sum([p['y'] for p in pairs]) / float(len(pairs))
            weight = self.radar_correction_gain
        else:
            y_err = pairs[0]['y']
            weight = self.radar_single_side_gain

        # 限制修正，避免锥桶跳变造成方向盘打满
        y_err = max(-self.max_center_correction,
                    min(self.max_center_correction, y_err))
        correction = -y_err * weight

        fused=[]
        for wx, wy in base_seg:
            lx, ly = self._map_to_vehicle(wx, wy)
            # 越远的点修正越小，保持全局形状
            alpha = max(0.0, min(1.0, 1.0-lx/max(self.local_max_x,1.0)))
            fused_ly = ly + correction * alpha
            mx, my = self._vehicle_to_map(lx, fused_ly)
            fused.append((mx,my))

        return fused, weight

    # ============================================================
    # 速度
    # ============================================================
    def _curvature_at(self, s):
        entry_end = self.entry_length
        circle_arc = 4.0 * math.pi * R_RACE
        c1_start = entry_end
        c1_end = c1_start + circle_arc
        c2_start = c1_end
        c2_end = c2_start + circle_arc
        if c1_start < s < c1_end or c2_start < s < c2_end:
            return 1.0 / R_RACE
        return 0.0

    def _plan_speed(self, s, v_now):
        look = 8.0
        kmax = 0.0
        ss = s
        while ss <= min(s + look, self.total_length):
            kmax = max(kmax, self._curvature_at(ss))
            ss += 0.5

        v_limit = self.max_speed if kmax < 0.01 else min(self.max_speed, self.curve_speed)
        d_end = self.total_length - s
        if d_end < self.end_decel_dist:
            v_limit = min(v_limit,
                          self.min_speed + (d_end / self.end_decel_dist) *
                          (self.max_speed - self.min_speed))

        if v_limit > v_now:
            v_new = min(v_limit, v_now + self.a_max * self.dt)
        else:
            v_new = max(v_limit, v_now - self.a_min * self.dt)
        return max(self.min_speed, min(self.max_speed, v_new))

    def _update_stage(self):
        if self.finish_s is not None and self.progress_s >= self.finish_s:
            new_stage = STAGE_FINISH
        else:
            lap = 2.0 * math.pi * R_RACE
            s0 = self.entry_length
            if self.progress_s < s0:
                new_stage = STAGE_IDLE
            elif self.progress_s < s0 + lap:
                new_stage = STAGE_RIGHT_LAP1
            elif self.progress_s < s0 + 2.0 * lap:
                new_stage = STAGE_RIGHT_LAP2
            elif self.progress_s < s0 + 3.0 * lap:
                new_stage = STAGE_LEFT_LAP3
            elif self.progress_s < s0 + 4.0 * lap:
                new_stage = STAGE_LEFT_LAP4
            else:
                new_stage = STAGE_EXIT

        if new_stage != self.stage:
            self.stage = new_stage
            names = {
                STAGE_IDLE: 'ENTRY',
                STAGE_RIGHT_LAP1: 'RIGHT LAP 1',
                STAGE_RIGHT_LAP2: 'RIGHT LAP 2',
                STAGE_LEFT_LAP3: 'LEFT LAP 3',
                STAGE_LEFT_LAP4: 'LEFT LAP 4',
                STAGE_EXIT: 'EXIT',
                STAGE_FINISH: 'FINISH'
            }
            rospy.loginfo('[BAZI] stage -> %s, s=%.2f/%.2f',
                          names.get(new_stage, 'UNKNOWN'),
                          self.progress_s, self.total_length)

    # ============================================================
    # 主循环
    # ============================================================
    def _straight_segment(self, length=8.0):
        seg = []
        c = math.cos(self.vyaw)
        s = math.sin(self.vyaw)
        n = max(2, int(round(length / self.path_spacing)))
        for i in range(n):
            d = i * length / float(n - 1)
            seg.append((self.vx + d * c, self.vy + d * s))
        return seg

    def _update(self, _):
        if not self.pose_received:
            return

        # ---------------- WAIT_GATE ----------------
        if self.state == S_WAIT_GATE:
            # 计算自第一帧以来前进的距离
            if self.wait_gate_start_x is not None:
                traveled = math.hypot(self.vx - self.wait_gate_start_x,
                                      self.vy - self.wait_gate_start_y)
            else:
                traveled = 0.0

            # 保护：走太远还没找到门，停车
            if traveled > self.approach_max_length:
                rospy.logwarn_throttle(
                    1.0,
                    "[GATE] traveled %.2fm without gate, stopping (max=%.2f)",
                    traveled, self.approach_max_length)
                self._publish_path(self._straight_segment(3.0))
                self.spd_pub.publish(Float64(0.0))
                self.stat_pub.publish(UInt8(0))
                return

            gate = self._confirm_start_gate()
            if gate is None:
                # 还没找到门：以 approach_speed 直行，继续找
                self._publish_path(self._straight_segment(5.0))
                self.spd_pub.publish(Float64(self.approach_speed))
                self.stat_pub.publish(UInt8(0))
                rospy.loginfo_throttle(1.0,
                    "[GATE] approaching... traveled=%.2fm/%.2fm v=%.2f",
                    traveled, self.approach_max_length, self.approach_speed)
                return

            # 找到门，后续不变
            geom = self._gate_map_geometry(gate)
            if geom is None:
                return

            scx, scy, gate_yaw = geom
            self.start_gate = gate
            self._generate_preset_path(scx, scy, gate_yaw)
            self.state = S_WAIT_GO if not self.auto_start else S_RUN
            self.stage = STAGE_IDLE
            self.last_gate_center = None
            self.gate_confirm_count = 0

            rospy.loginfo(
                '[START] gate confirmed center=(%.2f, %.2f) yaw=%.1fdeg auto_start=%s',
                scx, scy, math.degrees(gate_yaw), self.auto_start)

            if self.state == S_WAIT_GO:
                self.spd_pub.publish(Float64(0.0))
            return

        # ---------------- WAIT_GO ----------------
        if self.state == S_WAIT_GO:
            # 真实 RES Go 尚未接入时，保持 0 速度。仿真建议 auto_start=true。
            if self.preset_path is not None:
                seg = self._segment(self.progress_s, 3.0)
                self._publish_path(seg)
            self.spd_pub.publish(Float64(0.0))
            self.stat_pub.publish(UInt8(0))
            return

        # ---------------- RUN ----------------
        if self.state == S_RUN:
            self._update_progress()
            self._update_stage()

            lookahead = max(4.0, min(12.0, 2.2 * max(self.veh_speed, 0.5)))
            seg, blend_weight = self._build_fused_segment(self.progress_s, lookahead)

            if len(seg) >= 2:
                self._publish_path(seg)

            if self.finish_s is not None and self.progress_s >= self.finish_s:
                self.state = S_DONE
                self.stage = STAGE_FINISH
                self.spd_pub.publish(Float64(0.0))
                self.stat_pub.publish(UInt8(1))
                rospy.loginfo('[FINISH] exit completed; stop command latched via race_status=1')
                self._publish_markers()
                return

            v = self._plan_speed(self.progress_s, self.veh_speed)
            v = self._limit_speed_for_lidar(v, rospy.Time.now().to_sec())
            self.spd_pub.publish(Float64(v))
            self.stat_pub.publish(UInt8(0))

            rospy.loginfo_throttle(
                1.0,
                '[RUN] stage=%d s=%.1f/%.1f v=%.2f seg=%d cone_blend=%.2f',
                self.stage, self.progress_s, self.total_length, v,
                len(seg), blend_weight)

            self._publish_markers()
            return

        # ---------------- DONE ----------------
        self.spd_pub.publish(Float64(0.0))
        self.stat_pub.publish(UInt8(1))
    # ============================================================
    # 发布
    # ============================================================
    def _publish_path(self, seg):
        msg = Path()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = 'map'
        for x, y in seg:
            p = PoseStamped()
            p.header = msg.header
            p.pose.position.x = x
            p.pose.position.y = y
            p.pose.orientation.w = 1.0
            msg.poses.append(p)
        self.path_pub.publish(msg)

    def _publish_markers(self):
        ma = MarkerArray()

        m = Marker()
        m.header.frame_id = 'map'
        m.header.stamp = rospy.Time.now()
        m.ns = 'veh'
        m.id = 0
        m.type = Marker.ARROW
        m.action = Marker.ADD
        m.pose.position.x = self.vx
        m.pose.position.y = self.vy
        m.pose.position.z = 0.1
        q = tf_trans.quaternion_from_euler(0, 0, self.vyaw)
        m.pose.orientation.x, m.pose.orientation.y, m.pose.orientation.z, m.pose.orientation.w = q
        m.scale.x, m.scale.y, m.scale.z = 2.5, 0.5, 0.5
        m.color.a = 1.0
        m.color.r, m.color.g, m.color.b = 1.0, 0.0, 0.0
        m.lifetime = rospy.Duration(0.3)
        ma.markers.append(m)

        self.marker_pub.publish(ma)


if __name__ == '__main__':
    try:
        BaziPlannerV4()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
