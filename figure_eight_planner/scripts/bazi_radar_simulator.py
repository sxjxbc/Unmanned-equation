#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
八字绕环雷达模拟器
-----------------
模拟 clustering.py 的 ROS 输出接口，并与虚拟 VCU 真值车辆对齐。

发布：
  /clustered_points        geometry_msgs/PoseArray

订阅（默认）：
  /vehicle_pose_gt         geometry_msgs/PoseStamped
  /vehicle_speed           std_msgs/Float64

仿真闭环：
  tracking -> virtual_vcu -> /vehicle_pose_gt -> RadarSim -> /clustered_points

/clustered_points 的坐标系：
  frame_id = "/velodyne"
  X 前，Y 左，Z 上
  不带颜色，不带 confidence，和当前 clustering.py 的输出接口一致。

默认：
  1. 车辆固定在 (0, 0)，车头朝 +X；
  2. 前方放置一个约 3m 宽的起点门；
  3. 自动生成八字左右两个环的内外侧锥桶；
  4. 只发布车辆前方 FOV 内最近的 2~4 个锥桶；
  5. 可以打开 auto_move，让虚拟车辆沿八字中心线运动。

说明：
  这个程序不修改 clustering.py。
  它直接“冒充 clustering.py”向 /clustered_points 发 PoseArray。
"""

from __future__ import print_function

import math
import random
import rospy

from geometry_msgs.msg import PoseArray, Pose, PoseStamped
from std_msgs.msg import Float64
import tf.transformations as tf_trans


# ---------------- 赛道规格（沿用当前八字规划器） ----------------
R_INNER = 7.625
R_OUTER = 10.625
R_RACE = (R_INNER + R_OUTER) / 2.0
TRACK_WIDTH = R_OUTER - R_INNER


class BaziRadarSimulator(object):

    def __init__(self):
        rospy.init_node("bazi_radar_simulator", anonymous=False)

        # 发布参数
        self.rate_hz = rospy.get_param("~rate", 10.0)
        self.frame_id = rospy.get_param("~frame_id", "/velodyne")

        # 车辆状态：默认由 Virtual VCU 提供真值；保留 auto_move 作为脱离 VCU 的备用模式。
        self.use_external_pose = rospy.get_param("~use_external_pose", True)
        self.pose_topic = rospy.get_param("~pose_topic", "/vehicle_pose_gt")
        self.speed_topic = rospy.get_param("~speed_topic", "/vehicle_speed")
        self.auto_move = rospy.get_param("~auto_move", False)
        self.vehicle_speed = rospy.get_param("~vehicle_speed", 2.0)
        self.external_pose_received = False
        self.external_speed_received = False

        # 雷达视场/距离
        self.min_range = rospy.get_param("~min_range", 1.0)
        self.max_range = rospy.get_param("~max_range", 25.0)
        self.fov_deg = rospy.get_param("~fov_deg", 100.0)
        self.max_cones = rospy.get_param("~max_cones", 6)

        # 模拟噪声和漏检
        self.noise_std = rospy.get_param("~noise_std", 0.03)
        self.z_noise_std = rospy.get_param("~z_noise_std", 0.01)
        self.dropout_probability = rospy.get_param(
            "~dropout_probability", 0.05
        )

        # 起止线/起点门：八字两个圆心连线对应 X=0 的横向起止线。
        self.start_finish_x = rospy.get_param("~start_finish_x", 0.0)
        self.gate_x = rospy.get_param("~gate_x", self.start_finish_x)
        self.prioritize_gate = rospy.get_param("~prioritize_gate", True)
        self.front_overhang = rospy.get_param('~front_overhang', 0.7)

        # 锥桶布置
        self.cone_spacing = rospy.get_param("~cone_spacing", 3.0)

        # 初始车辆位姿（map）
        self.vehicle_x = rospy.get_param("~start_x", -15.0 - 0.7)
        self.vehicle_y = rospy.get_param("~start_y", 0.0)
        self.vehicle_yaw = math.radians(
            rospy.get_param("~start_yaw_deg", 0.0)
        )

        #雷达外参
        self.lidar_offset_x = rospy.get_param('~lidar_offset_x', 2.11552)
        self.lidar_offset_y = rospy.get_param('~lidar_offset_y', -0.02212)

        # 虚拟车辆轨迹
        self.motion_s = 0.0
        self.motion_path = self._build_motion_path()
        self.motion_arc = self._build_arc(self.motion_path)
        self.motion_total = self.motion_arc[-1] if self.motion_arc else 0.0

        # 固定世界锥桶地图
        self.cones_world = self._build_cones_world()

        self.pose_pub = rospy.Publisher(
            "/clustered_points",
            PoseArray,
            queue_size=1
        )

        self.sim_pose_pub = rospy.Publisher(
            "/sim_vehicle_pose",
            PoseStamped,
            queue_size=1
        )

        self.sim_speed_pub = rospy.Publisher(
            "/sim_vehicle_speed",
            Float64,
            queue_size=1
        )

        if self.use_external_pose:
            self.gt_pose_sub = rospy.Subscriber(
                self.pose_topic, PoseStamped, self._gt_pose_cb, queue_size=1
            )
            self.gt_speed_sub = rospy.Subscriber(
                self.speed_topic, Float64, self._gt_speed_cb, queue_size=1
            )

        self.last_time = rospy.Time.now()

        rospy.loginfo("=" * 60)
        rospy.loginfo("八字绕环雷达模拟器启动")
        rospy.loginfo(" /clustered_points: PoseArray")
        rospy.loginfo(" frame: %s", self.frame_id)
        rospy.loginfo(" use_external_pose: %s (%s)", self.use_external_pose, self.pose_topic)
        rospy.loginfo(" auto_move fallback: %s", self.auto_move)
        rospy.loginfo(" visible cones: %d max", self.max_cones)
        rospy.loginfo(" cone spacing: %.2f m", self.cone_spacing)
        rospy.loginfo(" gate width: %.2f m", TRACK_WIDTH)
        rospy.loginfo("=" * 60)

        self.timer = rospy.Timer(
            rospy.Duration(1.0 / self.rate_hz),
            self._timer_cb
        )

    # =========================================================
    # 外部车辆真值
    # =========================================================

    def _gt_pose_cb(self, msg):
        self.vehicle_x = msg.pose.position.x
        self.vehicle_y = msg.pose.position.y
        q = msg.pose.orientation
        qnorm = math.sqrt(q.x*q.x + q.y*q.y + q.z*q.z + q.w*q.w)
        if qnorm > 1e-6:
            qx, qy, qz, qw = q.x/qnorm, q.y/qnorm, q.z/qnorm, q.w/qnorm
            self.vehicle_yaw = math.atan2(2.0*(qw*qz + qx*qy),
                                          1.0 - 2.0*(qy*qy + qz*qz))
        self.external_pose_received = True

    def _gt_speed_cb(self, msg):
        self.vehicle_speed = msg.data
        self.external_speed_received = True

    # =========================================================
    # 生成八字中心线
    # =========================================================

    def _build_motion_path(self):
        """
        与当前八字规划器的几何思路一致：
        入口直线 -> 下侧圆 2圈 -> 上侧圆2圈 -> 出口直线
        坐标系：X前，Y左。
        """
        path = []

        # 入口：仅作为 auto_move 备用模式使用。
        entry_start = self.start_finish_x - 15.0 - self.front_overhang
        entry_end = self.start_finish_x
        x = entry_start
        while x <= entry_end:
            path.append((x, 0.0))
            x += 0.25

        # 下侧圆：从交叉点(0,0)开始，顺时针2圈
        n_circle = int((4.0 * math.pi * R_RACE) / 0.25)
        cx, cy = 0.0, -R_RACE

        for i in range(1, n_circle + 1):
            theta = math.pi / 2.0 - 4.0 * math.pi * i / float(n_circle)
            path.append((
                cx + R_RACE * math.cos(theta),
                cy + R_RACE * math.sin(theta)
            ))

        # 上侧圆：从交叉点(0,0)开始，逆时针2圈
        cx, cy = 0.0, R_RACE

        for i in range(1, n_circle + 1):
            theta = -math.pi / 2.0 + 4.0 * math.pi * i / float(n_circle)
            path.append((
                cx + R_RACE * math.cos(theta),
                cy + R_RACE * math.sin(theta)
            ))

        # 出口
        x = self.start_finish_x + 0.25
        while x <= self.start_finish_x + 25.0:
            path.append((x, 0.0))
            x += 0.25

        return path

    def _build_arc(self, path):
        arc = [0.0]
        for i in range(1, len(path)):
            dx = path[i][0] - path[i - 1][0]
            dy = path[i][1] - path[i - 1][1]
            arc.append(arc[-1] + math.hypot(dx, dy))
        return arc

    def _point_on_path(self, s):
        if not self.motion_path:
            return 0.0, 0.0, 0.0

        s = max(0.0, min(s, self.motion_total))

        # 简单线性搜索；仿真点数不大
        for i in range(1, len(self.motion_arc)):
            if self.motion_arc[i] >= s:
                s0 = self.motion_arc[i - 1]
                s1 = self.motion_arc[i]

                if s1 - s0 < 1e-9:
                    x, y = self.motion_path[i]
                    return x, y, 0.0

                f = (s - s0) / (s1 - s0)

                x0, y0 = self.motion_path[i - 1]
                x1, y1 = self.motion_path[i]

                x = x0 + (x1 - x0) * f
                y = y0 + (y1 - y0) * f

                yaw = math.atan2(y1 - y0, x1 - x0)

                return x, y, yaw

        x0, y0 = self.motion_path[-2]
        x1, y1 = self.motion_path[-1]
        return x1, y1, math.atan2(y1 - y0, x1 - x0)

    # =========================================================
    # 生成固定世界锥桶
    # =========================================================

    def _build_cones_world(self):
        cones = []

        # 入口段
        x = self.start_finish_x - 18.0
        while x <= self.start_finish_x + 25.0:
            cones.append((x, +TRACK_WIDTH / 2.0, 0.25))
            cones.append((x, -TRACK_WIDTH / 2.0, 0.25))
            x += self.cone_spacing

        # 圆上的锥桶：跳过切点附近 ±10°
        def add_circle(radius, n, cy_sign):
            R_circle = cy_sign * R_RACE
            for i in range(n):
                theta = 2.0 * math.pi * i / float(n)
                # 跳过切点附近（theta 接近 ±π/2）
                if cy_sign < 0:
                    # 下侧圆，切点在 theta = π/2
                    if abs(theta - math.pi/2) < math.radians(10):
                        continue
                else:
                    # 上侧圆，切点在 theta = -π/2
                    if abs(theta - (-math.pi/2)) < math.radians(10):
                        continue
                    if abs(theta - (3*math.pi/2)) < math.radians(10):
                        continue
                x = radius * math.cos(theta)
                y = R_circle + radius * math.sin(theta)
                cones.append((x, y, 0.25))

        # 下侧圆
        add_circle(R_INNER, 17, -1)
        add_circle(R_OUTER, 13, -1)
        # 上侧圆
        add_circle(R_INNER, 17, +1)
        add_circle(R_OUTER, 13, +1)

        return cones

    # =========================================================
    # 世界坐标 -> 虚拟 LiDAR 坐标
    # =========================================================

    def _world_to_vehicle(self, wx, wy, wz):
        dx = wx - self.vehicle_x
        dy = wy - self.vehicle_y
        c = math.cos(self.vehicle_yaw)
        s = math.sin(self.vehicle_yaw)

        # 世界 -> 车体中心（X前 Y右）
        vx = c * dx + s * dy
        body_y = -s * dx + c * dy

        # 车体中心 -> 雷达安装位置（减去雷达安装偏移）
        rel_x = vx - self.lidar_offset_x
        rel_y = body_y - self.lidar_offset_y

        # 车体 Y右 -> velodyne Y左
        radar_x = rel_x
        radar_y = -rel_y

        # 噪声
        radar_x += random.gauss(0.0, self.noise_std)
        radar_y += random.gauss(0.0, self.noise_std)
        vz = wz + random.gauss(0.0, self.z_noise_std)
        return radar_x, radar_y, vz

    # =========================================================
    # 找前方可见锥桶
    # =========================================================

    def _visible_cones(self):
        visible = []

        half_fov = math.radians(self.fov_deg / 2.0)

        for idx, (wx, wy, wz) in enumerate(self.cones_world):

            vx, vy, vz = self._world_to_vehicle(
                wx, wy, wz
            )

            distance = math.hypot(vx, vy)

            if distance < self.min_range:
                continue

            if distance > self.max_range:
                continue

            angle = math.atan2(vy, vx)

            if abs(angle) > half_fov:
                continue

            # 模拟单帧漏检
            if random.random() < self.dropout_probability:
                continue

            visible.append((
                distance,
                idx,
                vx,
                vy,
                vz
            ))

        visible.sort(key=lambda item: item[0])

        if self.prioritize_gate:
            gate_items = []
            other_items = []
            for item in visible:
                _, idx, vx, vy, vz = item
                wx, wy, _ = self.cones_world[idx]
                is_gate = (abs(wx - self.gate_x) < 1e-6 and
                           abs(abs(wy) - TRACK_WIDTH / 2.0) < 1e-6)
                (gate_items if is_gate else other_items).append(item)
            if len(gate_items) >= 2:
                keep_n = max(0, self.max_cones - len(gate_items))
                visible = gate_items + other_items[:keep_n]
                return visible

        return visible[:self.max_cones]

    # =========================================================
    # 发布 PoseArray
    # =========================================================

    def _publish_clustered_points(self):
        msg = PoseArray()

        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = self.frame_id

        visible = self._visible_cones()

        for distance, idx, x, y, z in visible:

            pose = Pose()

            pose.position.x = x
            pose.position.y = y
            pose.position.z = z

            # clustering.py 本身不会使用 orientation
            pose.orientation.w = 1.0

            msg.poses.append(pose)

        self.pose_pub.publish(msg)

        rospy.loginfo_throttle(
            1.0,
            "[RadarSim] visible=%d  vehicle=(%.2f, %.2f) yaw=%.1fdeg",
            len(visible),
            self.vehicle_x,
            self.vehicle_y,
            math.degrees(self.vehicle_yaw)
        )

    # =========================================================
    # 发布虚拟车位姿
    # =========================================================

    def _publish_sim_pose(self):
        msg = PoseStamped()

        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = "map"

        msg.pose.position.x = self.vehicle_x
        msg.pose.position.y = self.vehicle_y
        msg.pose.position.z = 0.0

        q = tf_trans.quaternion_from_euler(
            0.0,
            0.0,
            self.vehicle_yaw
        )

        msg.pose.orientation.x = q[0]
        msg.pose.orientation.y = q[1]
        msg.pose.orientation.z = q[2]
        msg.pose.orientation.w = q[3]

        self.sim_pose_pub.publish(msg)
        self.sim_speed_pub.publish(
            Float64(self.vehicle_speed if self.auto_move else 0.0)
        )

    # =========================================================
    # 自动运动
    # =========================================================

    def _update_motion(self, dt):
        if not self.auto_move:
            return

        self.motion_s += self.vehicle_speed * dt

        # 到终点后循环
        if self.motion_total > 0.0:
            if self.motion_s > self.motion_total:
                self.motion_s = 0.0

        self.vehicle_x, self.vehicle_y, self.vehicle_yaw = \
            self._point_on_path(self.motion_s)

    # =========================================================
    # 定时器
    # =========================================================

    def _timer_cb(self, _event):
        now = rospy.Time.now()
        dt = (now - self.last_time).to_sec()
        self.last_time = now

        if dt <= 0.0:
            dt = 1.0 / self.rate_hz

        if self.use_external_pose:
            if not self.external_pose_received:
                rospy.logwarn_throttle(2.0,
                    "[RadarSim] waiting for %s", self.pose_topic)
        elif self.auto_move:
            self._update_motion(dt)

        self._publish_clustered_points()


def main():
    BaziRadarSimulator()
    rospy.spin()


if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass
