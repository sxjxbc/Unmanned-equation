#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
FSAE Formula Student - Figure-8 Track Path Planning Module
八字赛道路径规划模块 v3.3
修改：输入源改为 激光雷达聚类 /clustered_points(PoseArray) + 视觉 /visual_cone_array(ConeArray)时空融合
"""

from __future__ import print_function
import rospy
import numpy as np
import math
import threading
import sys
from collections import deque

if sys.version_info[0] < 3:
    reload(sys)
    sys.setdefaultencoding('utf-8')

from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped, PoseArray, Pose
from std_msgs.msg import Header
from msgs.msg import ConeArray, Cone


class FusedCone(object):
    """融合后锥筒：激光位置 + 视觉颜色、置信度"""
    def __init__(self, x, y, z, color, confidence):
        self.x = x
        self.y = y
        self.z = z
        self.color = color
        self.confidence = confidence


class ConeTemplate(object):
    """预设锥筒模板"""
    def __init__(self, x, y, color, cone_id, is_tall=False):
        self.x = x
        self.y = y
        self.color = color
        self.cone_id = cone_id
        self.is_tall = is_tall
        self.matched = False
        self.actual_x = None
        self.actual_y = None
        
        self.observation_history = deque(maxlen=50)
        self.locked_x = None
        self.locked_y = None
        self.is_locked = False
        self.match_count = 0


class Figure8PathPlanner(object):
    """八字赛道路径规划器 - 激光聚类+视觉融合版本"""
    
    def __init__(self):
        rospy.init_node('figure_eight_planner_node', anonymous=False)

        self.map_receive_duration = 25.0
        self.receive_start_time = None
        self.reception_done = False

        self.circle_radius_inner = 7.625
        self.circle_radius_outer = 10.625
        self.circle_spacing = 18.25
        self.lane_width = 3.0
        
        self.match_distance_threshold = 3.5
        self.min_cones_for_match = 2
        self.cone_confidence_threshold = 0.15
        # 激光点与视觉锥筒匹配空间阈值(m)
        self.laser_visual_match_thresh = 1.2
        
        self.yellow_pair_distance_min = 2.3
        self.yellow_pair_ideal_distance = 3.0
        self.yellow_pair_distance_max = 3.7
        
        self.min_observations_to_lock = 20
        self.lock_stability_threshold = 0.06
        self.min_locked_cones_for_path = 8
        
        self.matching_phase = "SEARCHING"
        self.initial_match_quality_threshold = 0.3
        self.stable_match_quality_threshold = 0.6
        self.match_attempts = 0
        self.successful_updates = 0
        
        self.path_locked = False
        self.locked_path = None
        
        # 精确控制路径点数量
        self.start_straight_points = 31
        self.bottom_circle_points = 360
        self.top_circle_points = 360
        self.end_straight_points = 30
        self.total_path_points = self.start_straight_points + self.bottom_circle_points + self.top_circle_points + self.end_straight_points
        
        self.template_cones = []
        self.template_path_points = []
        self._create_template_map_from_coordinates()
        
        self.current_cone_map = []  # 存放FusedCone融合结果
        self.raw_cone_count = 0
        self.filtered_cone_count = 0
        self.matched_cones = []
        self.actual_path = None
        self.is_map_matched = False
        self.map_lock = threading.Lock()
        
        self.transform_center_x = 0.0
        self.transform_center_y = 0.0
        self.transform_rotation = 0.0
        self.match_quality = 0.0

        # 缓存最新消息
        self.latest_laser_msg = None
        self.latest_visual_msg = None

        # 订阅：激光聚类点 PoseArray（与 lidar_nodes/clustering.py 发布话题一致）；视觉ConeArray
        self.laser_sub = rospy.Subscriber("/clustered_points", PoseArray, self.laser_callback, queue_size=10)
        self.visual_sub = rospy.Subscriber("/visual_cone_array", ConeArray, self.visual_callback, queue_size=10)

        self.path_pub = rospy.Publisher('/planned_path', Path, queue_size=10)
        
        rospy.loginfo("=" * 80)
        rospy.loginfo("[八字赛道规划器] v3.3 激光聚类+视觉融合版本")
        rospy.loginfo("[输入] 激光：/clustered_points(PoseArray) | 视觉：/visual_cone_array(ConeArray)")
        rospy.loginfo("[赛道] 内圈直径=%.2fm，外圈直径=%.2fm，圆心距=%.2fm",
                     self.circle_radius_inner * 2, self.circle_radius_outer * 2,
                     self.circle_spacing)
        rospy.loginfo("[路径] 精确点数量: 起点直线=%d + 下方2圈=%d + 上方2圈=%d + 终点直线=%d = 总计%d点",
                     self.start_straight_points, self.bottom_circle_points,
                     self.top_circle_points, self.end_straight_points, self.total_path_points)
        rospy.loginfo("[匹配] 激光-视觉空间匹配阈值 %.2fm", self.laser_visual_match_thresh)
        rospy.loginfo("[匹配] 置信度阈值: %.2f", self.cone_confidence_threshold)
        rospy.loginfo("[优化] 持续优化时间: %.1f秒", self.map_receive_duration)
        rospy.loginfo("[锁定] 最少锁定锥筒数: %d", self.min_locked_cones_for_path)
        rospy.loginfo("=" * 80)

        self._start_reception()

    def laser_callback(self, msg: PoseArray):
        self.latest_laser_msg = msg

    def visual_callback(self, msg: ConeArray):
        self.latest_visual_msg = msg
        # 两边都收到数据，执行融合
        if self.latest_laser_msg is not None and self.latest_visual_msg is not None:
            self.fusion_step()

    def fusion_step(self):
        """激光PoseArray聚类点 + 视觉锥筒做空间匹配，输出FusedCone列表"""
        if self.reception_done:
            return
        elapsed_time = (rospy.Time.now() - self.receive_start_time).to_sec()
        if elapsed_time > self.map_receive_duration:
            self._stop_reception()
            return

        laser_poses = self.latest_laser_msg.poses
        visual_cones = self.latest_visual_msg.cones

        fused_list = []
        self.raw_cone_count = len(laser_poses)

        # 遍历激光聚类中心点，找最近视觉锥筒拿颜色置信
        for pose in laser_poses:
            lx = pose.position.x
            ly = pose.position.y
            lz = pose.position.z

            best_vis = None
            best_dist = float("inf")
            for vis_cone in visual_cones:
                dx = vis_cone.x - lx
                dy = vis_cone.y - ly
                dist = math.hypot(dx, dy)
                if dist < self.laser_visual_match_thresh and dist < best_dist:
                    best_dist = dist
                    best_vis = vis_cone

            if best_vis is not None:
                fused = FusedCone(
                    x=lx,
                    y=ly,
                    z=lz,
                    color=best_vis.color,
                    confidence=best_vis.confidence
                )
                fused_list.append(fused)

        # 置信过滤
        filtered = [f for f in fused_list if f.confidence >= self.cone_confidence_threshold]
        self.filtered_cone_count = len(filtered)

        with self.map_lock:
            self.current_cone_map = filtered
            if self.match_attempts % 10 == 0:
                rospy.loginfo("[融合]激光原始点:%d 融合后:%d (置信>%.2f)",
                              self.raw_cone_count, self.filtered_cone_count, self.cone_confidence_threshold)

            if len(self.current_cone_map) < self.min_cones_for_match:
                return

            if self.path_locked:
                self._publish_path()
                return

            self.match_attempts += 1

            if self.matching_phase == "SEARCHING":
                success = self._match_cones_from_closest_yellow_pair(self.current_cone_map)
                if success:
                    self.is_map_matched = True
                    self.matching_phase = "INITIAL"
                    rospy.loginfo("[阶段1] ✓ 初始匹配成功! 质量=%.1f%%",
                                 self.match_quality * 100)

            elif self.matching_phase == "INITIAL":
                self._refine_matching(self.current_cone_map)
                self._update_cone_observations(self.current_cone_map)

                if self.match_quality >= self.stable_match_quality_threshold:
                    self.matching_phase = "REFINING"
                    rospy.loginfo("[阶段2] ✓ 达到稳定匹配质量! 质量=%.1f%%",
                                 self.match_quality * 100)

            elif self.matching_phase == "REFINING":
                self._refine_matching(self.current_cone_map)
                self._update_cone_observations(self.current_cone_map)
                self._check_and_lock_path()

                if self.match_attempts % 5 == 0:
                    locked_count = sum(1 for cone in self.matched_cones if cone.is_locked)
                    rospy.loginfo("[优化] 质量=%.1f%%, 锁定=%d/%d, 时间=%.1fs",
                                 self.match_quality * 100,
                                 locked_count, self.min_locked_cones_for_path,
                                 elapsed_time)

            if not self.path_locked:
                self._compute_actual_path()
            self._publish_path()

    def _create_template_map_from_coordinates(self):
        """根据新坐标创建预设地图"""
        pixel_coords = [
            # 1-2: 黄色锥筒
            (1287, 1103), (1287, 1031),
            # 3: 黄色高锥筒
            (1276, 747),
            # 4-8: 红色锥筒
            (1048, 960), (1141, 941), (1217, 890), (1270, 812), (1287, 722),
            # 9-11: 蓝色锥筒
            (1048, 1066), (1180, 1038), (1276, 980),
            # 12-15: 红色锥筒
            (955, 941), (879, 890), (826, 812), (809, 722),
            # 16-19: 蓝色锥筒
            (916, 1038), (728, 852), (803, 963), (700, 722),
            # 20: 黄色高锥筒
            (1276, 697),
            # 21-24: 红色锥筒
            (1048, 484), (1141, 503), (1217, 554), (1270, 632),
            # 25-27: 蓝色锥筒
            (1048, 378), (1728, 503), (1804, 554),
            # 28-30: 红色锥筒
            (826, 632), (1407, 464), (1767, 406),
            # 31-33: 蓝色锥筒
            (728, 592), (803, 481), (1180, 406),
            # 34-37: 黄色锥筒
            (1287, 341), (1287, 413), (1396, 1103), (1396, 1031),
            # 38: 黄色高锥筒
            (1407, 747),
            # 39-43: 蓝色锥筒
            (1635, 960), (1542, 941), (1466, 890), (1413, 812), (1396, 722),
            # 44-46: 红色锥筒
            (1635, 1066), (1503, 1038), (1407, 980),
            # 47-50: 蓝色锥筒
            (1728, 941), (1804, 890), (1857, 812), (1874, 722),
            # 51-54: 红色锥筒
            (1767, 1038), (1955, 852), (1880, 963), (1983, 722),
            # 55: 黄色高锥筒
            (1407, 697),
            # 56-59: 蓝色锥筒
            (1635, 484), (1542, 503), (1466, 554), (1413, 632),
            # 60-62: 红色锥筒
            (1635, 378), (955, 503), (879, 554),
            # 63-65: 蓝色锥筒
            (1857, 632), (1276, 464), (916, 406),
            # 66-68: 红色锥筒
            (1955, 592), (1880, 481), (1503, 406),
            # 69-70: 黄色锥筒
            (1396, 341), (1396, 413)
        ]
        colors = []
        is_tall_flags = []
        colors.extend(['yellow', 'yellow'])
        is_tall_flags.extend([False, False])
        colors.append('yellow')
        is_tall_flags.append(True)
        colors.extend(['red'] * 5)
        is_tall_flags.extend([False] * 5)
        colors.extend(['blue'] * 3)
        is_tall_flags.extend([False] * 3)
        colors.extend(['red'] * 4)
        is_tall_flags.extend([False] * 4)
        colors.extend(['blue'] * 4)
        is_tall_flags.extend([False] * 4)
        colors.append('yellow')
        is_tall_flags.append(True)
        colors.extend(['red'] * 4)
        is_tall_flags.extend([False] * 4)
        colors.extend(['blue'] * 3)
        is_tall_flags.extend([False] * 3)
        colors.extend(['red'] * 3)
        is_tall_flags.extend([False] * 3)
        colors.extend(['blue'] * 3)
        is_tall_flags.extend([False] * 3)
        colors.extend(['yellow'] * 4)
        is_tall_flags.extend([False] * 4)
        colors.append('yellow')
        is_tall_flags.append(True)
        colors.extend(['blue'] * 5)
        is_tall_flags.extend([False] * 5)
        colors.extend(['red'] * 3)
        is_tall_flags.extend([False] * 3)
        colors.extend(['blue'] * 4)
        is_tall_flags.extend([False] * 4)
        colors.extend(['red'] * 4)
        is_tall_flags.extend([False] * 4)
        colors.append('yellow')
        is_tall_flags.append(True)
        colors.extend(['blue'] * 4)
        is_tall_flags.extend([False] * 4)
        colors.extend(['red'] * 3)
        is_tall_flags.extend([False] * 3)
        colors.extend(['blue'] * 3)
        is_tall_flags.extend([False] * 3)
        colors.extend(['red'] * 3)
        is_tall_flags.extend([False] * 3)
        colors.extend(['yellow'] * 2)
        is_tall_flags.extend([False] * 2)

        pixels_per_meter = 106.0 / 3.0
        all_x = [p[0] for p in pixel_coords]
        all_y = [p[1] for p in pixel_coords]
        img_center_x = sum(all_x) / len(all_x)
        img_center_y = sum(all_y) / len(all_y)
        rospy.loginfo("[坐标转换] 图像中心: (%.1f, %.1f) 像素", img_center_x, img_center_y)
        rospy.loginfo("[坐标转换] 转换比例: %.2f 像素/米", pixels_per_meter)
        world_coords = []
        for px, py in pixel_coords:
            wx = (px - img_center_x) / pixels_per_meter
            wy = (img_center_y - py) / pixels_per_meter
            world_coords.append((wx, wy))
        cone_id = 0
        for i, (wx, wy) in enumerate(world_coords):
            self.template_cones.append(ConeTemplate(wx, wy, colors[i], cone_id, is_tall_flags[i]))
            cone_id += 1

        yellow_normal_cones = [c for c in self.template_cones if c.color == 'yellow' and not c.is_tall]
        yellow_normal_cones.sort(key=lambda c: math.sqrt(c.x**2 + c.y**2))
        if len(yellow_normal_cones) >= 2:
            self.template_start_left = yellow_normal_cones[0]
            self.template_start_right = yellow_normal_cones[1]
            if self.template_start_left.x > self.template_start_right.x:
                self.template_start_left, self.template_start_right = self.template_start_right, self.template_start_left
            self.template_start_center_x = (self.template_start_left.x + self.template_start_right.x) / 2.0
            self.template_start_center_y = (self.template_start_left.y + self.template_start_right.y) / 2.0
            rospy.loginfo("[地图] 模板起点线中心: (%.2f, %.2f)", self.template_start_center_x, self.template_start_center_y)

        all_template_x = [c.x for c in self.template_cones]
        all_template_y = [c.y for c in self.template_cones]
        self.template_center_x = sum(all_template_x) / len(all_template_x)
        self.template_center_y = sum(all_template_y) / len(all_template_y)
        rospy.loginfo("[地图] 模板地图中心: (%.2f, %.2f)", self.template_center_x, self.template_center_y)
        red_count = len([c for c in self.template_cones if c.color == 'red'])
        blue_count = len([c for c in self.template_cones if c.color == 'blue'])
        yellow_count = len([c for c in self.template_cones if c.color == 'yellow'])
        tall_count = len([c for c in self.template_cones if c.is_tall])
        rospy.loginfo("[地图] 创建了 %d 个模板锥筒 (红=%d, 蓝=%d, 黄=%d, 高锥筒=%d)",
                      len(self.template_cones), red_count, blue_count, yellow_count, tall_count)
        self._create_template_path()

    def _create_template_path(self):
        path_points = []
        center_x = self.template_center_x
        center_y = self.template_center_y
        start_x = center_x - 15.0
        end_x = center_x
        x_step = (end_x - start_x) / (self.start_straight_points - 1)
        for i in range(self.start_straight_points):
            x = start_x + i * x_step
            path_points.append((x, center_y))
        radius = (self.circle_radius_inner + self.circle_radius_outer) / 2.0
        cx_down = center_x
        cy_down = center_y - radius
        angles_1 = np.linspace(90, -270, self.bottom_circle_points // 2, endpoint=False)
        for angle in angles_1:
            rad = math.radians(angle)
            x = cx_down + radius * math.cos(rad)
            y = cy_down + radius * math.sin(rad)
            path_points.append((x, y))
        angles_2 = np.linspace(90, -270, self.bottom_circle_points // 2, endpoint=False)
        for angle in angles_2:
            rad = math.radians(angle)
            x = cx_down + radius * math.cos(rad)
            y = cy_down + radius * math.sin(rad)
            path_points.append((x, y))
        cx_up = center_x
        cy_up = center_y + radius
        angles_3 = np.linspace(-90, 270, self.top_circle_points // 2, endpoint=False)
        for angle in angles_3:
            rad = math.radians(angle)
            x = cx_up + radius * math.cos(rad)
            y = cy_up + radius * math.sin(rad)
            path_points.append((x, y))
        angles_4 = np.linspace(-90, 270, self.top_circle_points // 2, endpoint=False)
        for angle in angles_4:
            rad = math.radians(angle)
            x = cx_up + radius * math.cos(rad)
            y = cy_up + radius * math.sin(rad)
            path_points.append((x, y))
        start_end_x = center_x
        end_end_x = center_x + 15.0
        x_step_end = (end_end_x - start_end_x) / (self.end_straight_points - 1)
        for i in range(self.end_straight_points):
            x = start_end_x + i * x_step_end
            path_points.append((x, center_y))
        self.template_path_points = path_points
        actual_points = len(path_points)
        expected_points = self.total_path_points
        if actual_points != expected_points:
            rospy.logwarn("[路径] 警告: 实际路径点数量(%d)与预期(%d)不符!", actual_points, expected_points)
        else:
            rospy.loginfo("[路径] ✓ 精确生成路径点: 总计%d点", self.total_path_points)

    def _start_reception(self):
        self.receive_start_time = rospy.Time.now()
        rospy.loginfo("[接收] 开始接收激光聚类+视觉锥筒，优化时间=%.1f秒", self.map_receive_duration)

    def _stop_reception(self):
        if self.laser_sub is not None:
            self.laser_sub.unregister()
        if self.visual_sub is not None:
            self.visual_sub.unregister()
        self.reception_done = True
        if not self.path_locked:
            self._force_lock_path()
        rospy.loginfo("[接收] 已停止，尝试锁定路径")

    def _force_lock_path(self):
        if self.actual_path and len(self.actual_path) > 0:
            self.locked_path = list(self.actual_path)
            self.path_locked = True
            self.matching_phase = "LOCKED"
            locked_count = sum(1 for cone in self.matched_cones if cone.is_locked)
            rospy.loginfo("[强制锁定] 基于 %d 个锁定锥筒，%d 个匹配锥筒", locked_count, len(self.matched_cones))

    def _normalize_color(self, color):
        color_lower = color.lower()
        if 'orange' in color_lower or 'red' in color_lower:
            return 'red'
        elif 'blue' in color_lower:
            return 'blue'
        elif 'yellow' in color_lower:
            return 'yellow'
        return color_lower

    def _match_cones_from_closest_yellow_pair(self, actual_cones):
        yellow_cones = [c for c in actual_cones if self._normalize_color(c.color) == 'yellow']
        if len(yellow_cones) < 2:
            if self.match_attempts % 10 == 0:
                rospy.logwarn("[匹配] 检测到的黄色锥筒不足2个（当前: %d）", len(yellow_cones))
            return False
        yellow_cones_sorted = sorted(yellow_cones, key=lambda c: math.sqrt(c.x**2 + c.y**2))
        closest_yellow1 = yellow_cones_sorted[0]
        closest_yellow2 = yellow_cones_sorted[1]
        pair_distance = math.sqrt((closest_yellow1.x - closest_yellow2.x)**2 + (closest_yellow1.y - closest_yellow2.y)**2)
        if (pair_distance < self.yellow_pair_distance_min or pair_distance > self.yellow_pair_distance_max):
            if self.match_attempts % 10 == 0:
                rospy.logwarn("[匹配] 起点线锥筒对间距异常: %.2fm (期望: %.2f-%.2fm)", pair_distance, self.yellow_pair_distance_min, self.yellow_pair_distance_max)
        if closest_yellow1.x < closest_yellow2.x:
            actual_left = closest_yellow1
            actual_right = closest_yellow2
        else:
            actual_left = closest_yellow2
            actual_right = closest_yellow1
        actual_start_center_x = (actual_left.x + actual_right.x) / 2.0
        actual_start_center_y = (actual_left.y + actual_right.y) / 2.0
        actual_dx = actual_right.x - actual_left.x
        actual_dy = actual_right.y - actual_left.y
        actual_line_angle = math.atan2(actual_dy, actual_dx)
        template_dx = self.template_start_right.x - self.template_start_left.x
        template_dy = self.template_start_right.y - self.template_start_left.y
        template_line_angle = math.atan2(template_dy, template_dx)
        rotation_angle = actual_line_angle - template_line_angle
        while rotation_angle > math.pi:
            rotation_angle -= 2 * math.pi
        while rotation_angle < -math.pi:
            rotation_angle += 2 * math.pi
        candidate_transforms = []
        for angle_offset in [0, math.pi]:
            test_rotation = rotation_angle + angle_offset
            while test_rotation > math.pi:
                test_rotation -= 2 * math.pi
            while test_rotation < -math.pi:
                test_rotation += 2 * math.pi
            cos_r = math.cos(test_rotation)
            sin_r = math.sin(test_rotation)
            rotated_template_center_x = (self.template_start_center_x * cos_r - self.template_start_center_y * sin_r)
            rotated_template_center_y = (self.template_start_center_x * sin_r + self.template_start_center_y * cos_r)
            translation_x = actual_start_center_x - rotated_template_center_x
            translation_y = actual_start_center_y - rotated_template_center_y
            match_score = self._evaluate_transform_quality(actual_cones, translation_x, translation_y, test_rotation)
            candidate_transforms.append({
                'rotation': test_rotation,
                'translation_x': translation_x,
                'translation_y': translation_y,
                'score': match_score,
                'angle_offset': angle_offset
            })
            rospy.loginfo("[方向评估] %s: 匹配质量=%.1f%%", "原方向" if angle_offset == 0 else "翻转180°", match_score * 100)
        best_transform = max(candidate_transforms, key=lambda t: t['score'])
        if best_transform['angle_offset'] != 0:
            rospy.loginfo("[方向选择] ✓ 自动选择翻转方向（质量提升 %.1f%%）", (best_transform['score'] - candidate_transforms[0]['score']) * 100)
        rotation_angle = best_transform['rotation']
        translation_x = best_transform['translation_x']
        translation_y = best_transform['translation_y']
        cos_r = math.cos(rotation_angle)
        sin_r = math.sin(rotation_angle)
        transformed_templates = []
        for template_cone in self.template_cones:
            x_rot = template_cone.x * cos_r - template_cone.y * sin_r
            y_rot = template_cone.x * sin_r + template_cone.y * cos_r
            x_final = x_rot + translation_x
            y_final = y_rot + translation_y
            transformed_templates.append({'original': template_cone, 'x': x_final, 'y': y_final, 'color': template_cone.color})
        matched_count = 0
        match_details = {'red': {'matched':0,'total':0},'blue':{'matched':0,'total':0},'yellow':{'matched':0,'total':0}}
        for actual_cone in actual_cones:
            color = self._normalize_color(actual_cone.color)
            if color in match_details:
                match_details[color]['total'] +=1
        for actual_cone in actual_cones:
            actual_color = self._normalize_color(actual_cone.color)
            best_match = None
            best_distance = float('inf')
            for template_data in transformed_templates:
                template_color = self._normalize_color(template_data['color'])
                if template_color != actual_color:
                    continue
                if template_data['original'].matched:
                    continue
                distance = math.sqrt((template_data['x'] - actual_cone.x)**2 + (template_data['y'] - actual_cone.y)**2)
                if distance < self.match_distance_threshold and distance < best_distance:
                    best_match = template_data
                    best_distance = distance
            if best_match:
                template_cone = best_match['original']
                template_cone.matched = True
                template_cone.actual_x = actual_cone.x
                template_cone.actual_y = actual_cone.y
                template_cone.observation_history.append((actual_cone.x, actual_cone.y))
                template_cone.match_count +=1
                self.matched_cones.append(template_cone)
                matched_count +=1
                color = self._normalize_color(actual_cone.color)
                if color in match_details:
                    match_details[color]['matched'] +=1
        if len(actual_cones) >0:
            self.match_quality = float(matched_count)/len(actual_cones)
        else:
            self.match_quality =0.0
        rospy.loginfo("[初始匹配] 结果: %d/%d (%.1f%%) | R:%d/%d B:%d/%d Y:%d/%d",
                      matched_count, len(actual_cones), self.match_quality*100,
                      match_details['red']['matched'],match_details['red']['total'],
                      match_details['blue']['matched'],match_details['blue']['total'],
                      match_details['yellow']['matched'],match_details['yellow']['total'])
        if matched_count < self.min_cones_for_match:
            for template_cone in self.template_cones:
                template_cone.matched=False
                template_cone.actual_x=None
                template_cone.actual_y=None
                template_cone.observation_history.clear()
            self.matched_cones=[]
            return False
        self.transform_center_x = translation_x
        self.transform_center_y = translation_y
        self.transform_rotation = rotation_angle
        return True

    def _evaluate_transform_quality(self, actual_cones, translation_x, translation_y, rotation_angle):
        cos_r = math.cos(rotation_angle)
        sin_r = math.sin(rotation_angle)
        transformed_templates = []
        for template_cone in self.template_cones:
            x_rot = template_cone.x * cos_r - template_cone.y * sin_r
            y_rot = template_cone.x * sin_r + template_cone.y * cos_r
            x_final = x_rot + translation_x
            y_final = y_rot + translation_y
            transformed_templates.append({'x':x_final,'y':y_final,'color':template_cone.color})
        matched_count=0
        forward_cone_count=0
        for actual_cone in actual_cones:
            actual_color = self._normalize_color(actual_cone.color)
            if actual_cone.x>0:
                forward_cone_count +=1
            best_distance = float('inf')
            for template_data in transformed_templates:
                template_color = self._normalize_color(template_data['color'])
                if template_color != actual_color:
                    continue
                distance = math.sqrt((template_data['x']-actual_cone.x)**2 + (template_data['y']-actual_cone.y)**2)
                if distance < best_distance:
                    best_distance = distance
            if best_distance < self.match_distance_threshold:
                matched_count +=1
        match_rate = float(matched_count)/max(len(actual_cones),1)
        forward_ratio = float(forward_cone_count)/max(len(actual_cones),1)
        quality_score = 0.7*match_rate +0.3*forward_ratio
        return quality_score

    def _refine_matching(self, actual_cones):
        cos_r = math.cos(self.transform_rotation)
        sin_r = math.sin(self.transform_rotation)
        unmatched_templates = [c for c in self.template_cones if not c.matched]
        newly_matched = 0
        for template_cone in unmatched_templates:
            x_rot = template_cone.x * cos_r - template_cone.y * sin_r
            y_rot = template_cone.x * sin_r + template_cone.y * cos_r
            x_pred = x_rot + self.transform_center_x
            y_pred = y_rot + self.transform_center_y
            best_match = None
            best_distance = float('inf')
            template_color = self._normalize_color(template_cone.color)
            for actual_cone in actual_cones:
                actual_color = self._normalize_color(actual_cone.color)
                if actual_color != template_color:
                    continue
                already_matched=False
                for matched in self.matched_cones:
                    if matched.actual_x and matched.actual_y:
                        dist_to_matched = math.sqrt((matched.actual_x - actual_cone.x)**2 + (matched.actual_y - actual_cone.y)**2)
                        if dist_to_matched <0.5:
                            already_matched=True
                            break
                if already_matched:
                    continue
                distance = math.sqrt((x_pred - actual_cone.x)**2 + (y_pred - actual_cone.y)**2)
                if distance < self.match_distance_threshold and distance < best_distance:
                    best_match = actual_cone
                    best_distance = distance
            if best_match:
                template_cone.matched = True
                template_cone.actual_x = best_match.x
                template_cone.actual_y = best_match.y
                template_cone.observation_history.append((best_match.x, best_match.y))
                template_cone.match_count = 1
                self.matched_cones.append(template_cone)
                newly_matched +=1
        if newly_matched>0:
            self.successful_updates +=1
            rospy.loginfo("[精细化] 新增匹配 %d 个锥筒, 总计 %d/%d", newly_matched, len(self.matched_cones), len(self.template_cones))
        if len(actual_cones)>0:
            self.match_quality = float(len(self.matched_cones)) / len(self.template_cones)

    def _transform_point(self, x, y, origin_x, origin_y, angle):
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
        x_rot = x * cos_a - y * sin_a
        y_rot = x * sin_a + y * cos_a
        return x_rot + origin_x, y_rot + origin_y

    def _update_cone_observations(self, actual_cones):
        for actual_cone in actual_cones:
            best_match = None
            best_distance = float('inf')
            actual_color = self._normalize_color(actual_cone.color)
            for matched_cone in self.matched_cones:
                if matched_cone.is_locked:
                    continue
                template_color = self._normalize_color(matched_cone.color)
                if template_color != actual_color:
                    continue
                ref_x = matched_cone.actual_x if matched_cone.actual_x else matched_cone.x
                ref_y = matched_cone.actual_y if matched_cone.actual_y else matched_cone.y
                dist = math.sqrt((ref_x - actual_cone.x)**2 + (ref_y - actual_cone.y)**2)
                if dist < self.match_distance_threshold and dist < best_distance:
                    best_match = matched_cone
                    best_distance = dist
            if best_match and not best_match.is_locked:
                best_match.observation_history.append((actual_cone.x, actual_cone.y))
                best_match.match_count +=1
                if len(best_match.observation_history)>=3:
                    observations = list(best_match.observation_history)
                    x_vals = [o[0] for o in observations]
                    y_vals = [o[1] for o in observations]
                    best_match.actual_x = np.median(x_vals)
                    best_match.actual_y = np.median(y_vals)
                if len(best_match.observation_history)>= self.min_observations_to_lock:
                    observations = list(best_match.observation_history)
                    x_vals = [o[0] for o in observations]
                    y_vals = [o[1] for o in observations]
                    std_x = np.std(x_vals)
                    std_y = np.std(y_vals)
                    position_std = math.sqrt(std_x**2 + std_y**2)
                    if position_std < self.lock_stability_threshold:
                        median_x = np.median(x_vals)
                        median_y = np.median(y_vals)
                        best_match.is_locked = True
                        best_match.locked_x = median_x
                        best_match.locked_y = median_y
                        best_match.actual_x = median_x
                        best_match.actual_y = median_y
                        rospy.loginfo("[锁定] 锥筒 %s #%d @ (%.2f, %.2f), 稳定度=%.3fm",
                                      best_match.color, best_match.cone_id, median_x, median_y, position_std)

    def _check_and_lock_path(self):
        if self.path_locked:
            return
        locked_count = sum(1 for cone in self.matched_cones if cone.is_locked)
        if locked_count >= self.min_locked_cones_for_path:
            self._compute_actual_path()
            self.locked_path = list(self.actual_path)
            self.path_locked = True
            self.matching_phase = "LOCKED"
            rospy.loginfo("="*80)
            rospy.loginfo("[路径锁定] ✓ 已锁定！基于 %d 个稳定锥筒 (共%d个匹配)", locked_count, len(self.matched_cones))
            rospy.loginfo("[路径锁定] 质量: %.1f%%, 优化次数: %d", self.match_quality*100, self.successful_updates)
            rospy.loginfo("[路径锁定] 路径点数: %d", len(self.locked_path))
            rospy.loginfo("="*80)

    def _compute_actual_path(self):
        if len(self.matched_cones) <2:
            return
        if not self.template_path_points:
            return
        actual_path_points = []
        for template_x, template_y in self.template_path_points:
            actual_x, actual_y = self._transform_point(template_x, template_y,
                                                       self.transform_center_x, self.transform_center_y,
                                                       self.transform_rotation)
            actual_path_points.append((actual_x, actual_y))
        self.actual_path = actual_path_points

    def _publish_path(self):
        path_to_publish = self.locked_path if self.path_locked else self.actual_path
        if not path_to_publish:
            return
        path_msg = Path()
        path_msg.header = Header()
        path_msg.header.stamp = rospy.Time.now()
        path_msg.header.frame_id = "map"
        for x,y in path_to_publish:
            pose = PoseStamped()
            pose.header = path_msg.header
            pose.pose.position.x = x
            pose.pose.position.y = y
            pose.pose.position.z = 0.0
            pose.pose.orientation.w = 1.0
            path_msg.poses.append(pose)
        self.path_pub.publish(path_msg)

    def _print_debug_info(self):
        rospy.loginfo("="*80)
        rospy.loginfo("调试信息:")
        rospy.loginfo("-"*80)
        with self.map_lock:
            rospy.loginfo("激光原始聚类点: %d", self.raw_cone_count)
            rospy.loginfo("融合过滤后锥筒: %d (置信度阈值=%.2f)", self.filtered_cone_count, self.cone_confidence_threshold)
            if len(self.current_cone_map)>0:
                color_counts={}
                for cone in self.current_cone_map:
                    c = self._normalize_color(cone.color)
                    color_counts[c]=color_counts.get(c,0)+1
                rospy.loginfo("  颜色分布: %s", color_counts)
        rospy.loginfo("-"*80)
        rospy.loginfo("匹配阶段: %s", self.matching_phase)
        rospy.loginfo("匹配状态: %s", "已匹配 ✓" if self.is_map_matched else "未匹配")
        if self.is_map_matched:
            rospy.loginfo("匹配质量: %.1f%%", self.match_quality*100)
        rospy.loginfo("路径状态: %s", "已锁定 ✓" if self.path_locked else "未锁定")
        rospy.loginfo("匹配锥筒: %d / %d", len(self.matched_cones), len(self.template_cones))
        if len(self.matched_cones)>0:
            matched_red = len([c for c in self.matched_cones if c.color == 'red'])
            matched_blue = len([c for c in self.matched_cones if c.color == 'blue'])
            matched_yellow = len([c for c in self.matched_cones if c.color == 'yellow'])
            rospy.loginfo("  匹配分布: 红=%d, 蓝=%d, 黄=%d", matched_red, matched_blue, matched_yellow)
        locked_count = sum(1 for cone in self.matched_cones if cone.is_locked)
        rospy.loginfo("锁定锥筒: %d / %d 需要", locked_count, self.min_locked_cones_for_path)
        rospy.loginfo("-"*80)
        rospy.loginfo("优化统计:")
        rospy.loginfo("  尝试次数: %d", self.match_attempts)
        rospy.loginfo("  成功更新: %d", self.successful_updates)
        elapsed = (rospy.Time.now() - self.receive_start_time).to_sec() if self.receive_start_time else 0
        rospy.loginfo("  已用时间: %.1fs / %.1fs", elapsed, self.map_receive_duration)
        if self.is_map_matched:
            rospy.loginfo("-"*80)
            rospy.loginfo("变换参数:")
            rospy.loginfo("  平移: (%.2f, %.2f)", self.transform_center_x, self.transform_center_y)
            rospy.loginfo("  旋转: %.1f°", math.degrees(self.transform_rotation))
        if self.actual_path:
            rospy.loginfo("-"*80)
            rospy.loginfo("路径信息:")
            rospy.loginfo("  路径点数: %d", len(self.actual_path))
            rospy.loginfo("  起点: (%.3f, %.3f)", self.actual_path[0][0], self.actual_path[0][1])
            rospy.loginfo("  终点: (%.3f, %.3f)", self.actual_path[-1][0], self.actual_path[-1][1])
        rospy.loginfo("="*80)

    def run(self):
        rospy.loginfo("[系统] 启动路径规划器（激光聚类+视觉融合）...")
        rospy.loginfo("[系统] 渐进式匹配: 快速初始化 -> 持续优化 -> 高质量锁定")
        rospy.loginfo("[系统] 路径点数量: 总计%d点", self.total_path_points)
        rospy.loginfo("[系统] 使用 Ctrl+C 退出")
        rospy.loginfo("="*80)
        rate = rospy.Rate(0.2)
        debug_counter =0
        try:
            while not rospy.is_shutdown():
                rate.sleep()
                debug_counter +=1
                if debug_counter %6 ==0:
                    self._print_debug_info()
        except rospy.ROSInterruptException:
            pass
        except KeyboardInterrupt:
            rospy.loginfo("[系统] 键盘中断，正在关闭...")
        finally:
            rospy.loginfo("[系统] 路径规划器已关闭")

def main():
    try:
        planner = Figure8PathPlanner()
        planner.run()
    except rospy.ROSInterruptException:
        pass
    except KeyboardInterrupt:
        rospy.loginfo("[系统] 关闭中...")
    except Exception as e:
        rospy.logerr("[错误] %s", str(e))
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()
