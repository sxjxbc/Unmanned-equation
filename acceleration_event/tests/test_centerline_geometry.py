#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Geometry regressions against actual node methods; no ROS or vehicle output.

This file also runs directly under the original Python 2 + NumPy environment.
"""
from __future__ import print_function
import ast
import json
import math
import os
import sys
import threading
import time
import unittest
from itertools import combinations

import numpy as np

SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, SRC)
from data_quality import (fresh, finite, map_line_to_local, local_line_to_map,
                          pair_quality, centerline_from_status, LongitudinalProgress)


class Object(object):
    def __init__(self, **values):
        self.__dict__.update(values)


class Stamp(Object):
    def __sub__(self, other):
        return Object(to_sec=lambda: self.to_sec() - other.to_sec())


def node_methods(filename, names, namespace):
    with open(os.path.join(SRC, filename), 'rb') as stream:
        tree = ast.parse(stream.read(), filename)
    body = [method for cls in tree.body if isinstance(cls, ast.ClassDef)
            for method in cls.body
            if isinstance(method, ast.FunctionDef) and method.name in names]
    assert len(body) == len(names)
    module = ast.Module(body=body)
    if 'type_ignores' in ast.Module._fields:
        module.type_ignores = []
    eval(compile(ast.fix_missing_locations(module), filename, 'exec'), namespace)
    return type('Subject', (object,), dict((name, namespace[name]) for name in names))()


class GeometryTests(unittest.TestCase):
    def setUp(self):
        self.now = 10.0
        self.receive_now = 100.0
        self.warnings = []
        self.ros = Object(
            Time=Object(now=lambda: Stamp(to_sec=lambda value=self.now: value)),
            get_param=lambda name, default: default,
            logwarn_throttle=lambda *args: self.warnings.append(args),
            loginfo_throttle=lambda *args: None,
            loginfo=lambda *args: None)

    def namespace(self):
        return dict(math=math, np=np, rospy=self.ros, fresh=fresh,
                    finite=finite, pair_quality=pair_quality,
                    centerline_from_status=centerline_from_status,
                    monotonic=lambda: self.receive_now, combinations=combinations,
                    json=json, time=time, Header=Object, String=Object,
                    Path=lambda: Object(header=None, poses=[]),
                    PoseStamped=lambda: Object(header=None, pose=Object(
                        position=Object(x=0, y=0, z=0), orientation=Object(z=0, w=1))),
                    map_line_to_local=map_line_to_local,
                    local_line_to_map=local_line_to_map)

    def planner(self):
        names = ['world_to_vehicle', 'vehicle_to_world', 'select_local_cones',
                 'pair_cones', 'estimate_centerline', 'update_centerline',
                 'generate_local_path', 'publish_path']
        o = node_methods('zhixian_realtime.py', names, self.namespace())
        o.lock = threading.RLock()
        o.cones = []
        o.last_cone_time = 10.0
        o.vehicle_x = o.vehicle_y = o.vehicle_yaw = 0.0
        o.track_width = 3.0
        o.track_center_y = 0.0
        o.track_forward_min = .5
        o.track_forward_max = 25.0
        o.max_local_cones = 20
        o.centerline_timeout = .5
        o.pair_distance_min = 2.3
        o.pair_distance_max = 3.7
        o.pair_max_longitudinal_diff = 1.5
        o.pair_row_min_spacing = 1.5
        o.cone_row_spacing = 5.0
        o.cone_row_spacing_tolerance = 1.5
        o.max_pair_count = 2
        o.side_threshold = .5
        o.side_correction_offset = .6
        o.centerline_valid = o.planning_started = False
        o.centerline_k = o.centerline_b = 0.0
        o.track_reference_k = None
        o.centerline_pairs = 0
        o.centerline_source = 'none'
        o.centerline_confidence = o.pending_centerline_confidence = 0.0
        o.last_valid_observation_stamp = None
        o.centerline_last_valid_time = None
        o.straight_slope_limit = .06
        o.max_centerline_slope = .25
        o.max_slope_change = .02
        o.slope_alpha = .12
        o.centerline_alpha = .15
        o.last_center_b = None
        o.observation_hold_timeout = .5
        o.max_center_offset = 1.5
        o.local_path_length = 30.0
        o.path_point_spacing = .5
        o.blend_distance = 6.0
        o.path_smooth_window = 3
        o.pose_stamp = 10.0
        o.path_seq = 0
        o.status_messages = []
        o.paths = []
        o.status_pub = Object(publish=o.status_messages.append)
        o.path_pub = Object(publish=o.paths.append)
        return o

    def controller(self):
        names = ['world_to_vehicle', 'vehicle_to_world', 'path_is_fresh',
                 'estimate_local_centerline', 'local_centerline_result',
                 'get_local_centerline_target', 'local_blend_weight',
                 '_planning_status_callback', '_control_loop', 'calculate_errors_from_xy',
                 'update_finish_distance', 'finish_status']
        o = node_methods('genzong_realtime.py', names, self.namespace())
        o.vehicle_x = o.vehicle_y = o.vehicle_yaw = 0.0
        o.path_timeout = o.path_observation_timeout = .5
        o.local_blend_two_pairs = .65
        o.local_blend_one_pair = .45
        o.local_blend_single_side = .2
        o.local_centerline_k = o.local_centerline_b = 0.0
        o.local_centerline_map_k = o.local_centerline_map_b = 0.0
        o.local_centerline_pairs = 0
        o.local_centerline_active = False
        o.local_centerline_observation_stamp = None
        o.local_centerline_confidence = 0.0
        o.finish_progress = LongitudinalProgress()
        o.finish_start_mode = 'first_valid_pose'
        o.pose_stamp = 10.0
        o.finish_triggered = False
        o.total_distance = 0.0
        self.set_controller_line(o)
        return o

    def set_controller_line(self, o, k=0, b=0, pairs=2, confidence=1,
                            source='two_pairs', observation_stamp=None):
        stamp = self.now if observation_stamp is None else observation_stamp
        o.path_received = True
        o.current_path = Object(header=Object(stamp=Object(to_sec=lambda value=self.now: value)))
        o.path_receive_time = o.status_receive_time = self.receive_now
        o.planning_status = dict(valid=True, path_stamp=self.now, observation_stamp=stamp,
            pairs=pairs, centerline=dict(map_k=k, map_b=b, pairs=pairs, source=source,
                                        confidence=confidence, observation_stamp=stamp))

    def transfer_plan(self, planner, controller):
        points, ready, _ = planner.generate_local_path()
        self.assertTrue(ready)
        planner.publish_path(points)
        controller.current_path = planner.paths[-1]
        controller.path_received = True
        controller.path_receive_time = self.receive_now
        controller._planning_status_callback(planner.status_messages[-1])

    def cones(self, k=0.0, b=0.0):
        return [(x, k*x+b+side*1.5, 1.0)
                for x in (5.0, 10.0) for side in (-1, 1)]

    def assertOnLine(self, point, k, b):
        self.assertAlmostEqual(point[1], k*point[0]+b, places=8)

    def test_map_local_round_trip_with_translation_and_rotation(self):
        for yaw in (-.7, -.2, 0, .3, .7):
            local = map_line_to_local(.03, .4, 12.0, -.2, yaw)
            result = local_line_to_map(local[0], local[1], 12.0, -.2, yaw)
            self.assertAlmostEqual(result[0], .03)
            self.assertAlmostEqual(result[1], .4)

    def test_perpendicular_lines_are_rejected(self):
        self.assertIsNone(map_line_to_local(0, 0, 0, 0, math.pi/2))
        self.assertIsNone(local_line_to_map(0, 0, 0, 0, math.pi/2))

    def test_planner_normal_pairs_take_priority(self):
        o = self.planner()
        o.cones = self.cones()
        self.assertTrue(o.update_centerline())
        self.assertEqual(o.centerline_pairs, 2)
        self.assertEqual(o.centerline_source, 'two_pairs')
        self.assertFalse(any('单侧' in str(args) for args in self.warnings))

    def test_planner_yaw_does_not_change_fixed_road(self):
        for degrees in (-30, -10, 0, 10, 30):
            o = self.planner()
            o.vehicle_yaw = math.radians(degrees)
            o.cones = self.cones()
            points, ready, updated = o.generate_local_path()
            self.assertTrue(ready)
            self.assertTrue(updated)
            self.assertAlmostEqual(o.centerline_k, 0.0)
            self.assertAlmostEqual(o.centerline_b, 0.0)
            # Beyond the intentional entry blend, points must lie on map Y=0.
            for point in points[16:]:
                self.assertOnLine(point, 0, 0)

    def test_planner_single_side_cannot_start(self):
        o = self.planner()
        o.cones = [(5, 1.5, 1)]
        self.assertIsNone(o.estimate_centerline())
        self.assertEqual(o.generate_local_path(), (None, False, False))

    def test_planner_single_cone_fallback_after_start_both_sides(self):
        for side in (-1, 1):
            o = self.planner()
            o.cones = self.cones()
            o.update_centerline()
            o.last_cone_time = self.now = 10.1
            o.vehicle_yaw = math.radians(20)
            o.cones = [(5, side*1.5, 1)]
            result = o.estimate_centerline()
            self.assertIsNotNone(result)
            self.assertAlmostEqual(result[1], -side*.6)
            self.assertEqual(result[2], 0)
            self.assertTrue(o.update_centerline())
            self.assertEqual(o.centerline_source, 'single_side')
            self.assertEqual(o.centerline_pairs, 0)

    def test_planner_mixed_unpaired_cones_do_not_become_single_side(self):
        o = self.planner()
        o.cones = self.cones()
        o.update_centerline()
        o.cones = [(5, -1.5, 1), (10, 1.5, 1)]
        self.assertIsNone(o.estimate_centerline())

    def test_planner_hold_remains_in_map_after_pose_changes(self):
        o = self.planner()
        o.cones = self.cones(.02, .1)
        o.update_centerline()
        self.now = 10.1
        o.vehicle_x, o.vehicle_y, o.vehicle_yaw = 1.0, .3, -.2
        o.cones = []
        o.last_cone_time = 10.1
        points, ready, updated = o.generate_local_path()
        self.assertTrue(ready)
        self.assertFalse(updated)
        for point in points[16:]:
            self.assertOnLine(point, .02, .1)
        self.assertEqual(o.last_valid_observation_stamp, 10.0)
        self.now = 10.6
        self.assertEqual(o.generate_local_path(), (None, False, False))

    def test_planner_repeated_frame_does_not_refilter(self):
        o = self.planner()
        o.cones = self.cones()
        o.update_centerline()
        o.last_cone_time = self.now = 10.1
        o.cones = self.cones(0, .3)
        self.assertTrue(o.update_centerline())
        b = o.centerline_b
        o.vehicle_y, o.vehicle_yaw = .2, .15
        self.assertFalse(o.update_centerline())
        self.assertEqual(o.centerline_b, b)

    def test_planner_rejects_bad_map_slope_without_stitching_intercept(self):
        o = self.planner()
        o.cones = self.cones()
        o.update_centerline()
        o.last_cone_time = self.now = 10.1
        o.cones = self.cones(.1, .2)
        self.assertFalse(o.update_centerline())
        self.assertAlmostEqual(o.centerline_k, 0)
        self.assertAlmostEqual(o.centerline_b, 0)
        self.assertEqual(o.last_valid_observation_stamp, 10.0)

    def test_planner_lateral_envelope_inhibits_instead_of_bending_line(self):
        o = self.planner()
        o.cones = self.cones()
        o.update_centerline()
        o.vehicle_y = 2.0
        points, ready, _ = o.generate_local_path()
        self.assertIsNone(points)
        self.assertFalse(ready)

    def test_planner_no_forward_representation_returns_no_path(self):
        o = self.planner()
        o.cones = self.cones()
        o.update_centerline()
        for yaw in (math.pi/2, math.pi):
            o.vehicle_yaw = yaw
            points, ready, _ = o.generate_local_path()
            self.assertIsNone(points)
            self.assertFalse(ready)

    def test_controller_hold_target_remains_on_map_line(self):
        o = self.controller()
        for yaw in (-.35, .35):
            o.vehicle_x, o.vehicle_y, o.vehicle_yaw = 1.0, .3, yaw
            target = o.get_local_centerline_target(4)
            self.assertIsNotNone(target)
            self.assertOnLine((target['x'], target['y']), 0, 0)
        self.now = 10.6
        self.assertIsNone(o.get_local_centerline_target(4))

    def test_controller_new_measurement_after_pose_change_keeps_fixed_line(self):
        o = self.controller()
        self.set_controller_line(o, .02, .1)
        o.get_local_centerline_target(4)
        o.vehicle_x, o.vehicle_y, o.vehicle_yaw = .5, .2, .15
        self.now = 10.1
        self.set_controller_line(o, .02, .1)
        target = o.get_local_centerline_target(4)
        self.assertOnLine((target['x'], target['y']), .02, .1)

    def test_controller_repeated_frame_does_not_refilter_or_refresh_hold(self):
        o = self.controller()
        self.set_controller_line(o, 0, .2)
        o.get_local_centerline_target(4)
        self.now = 10.1
        self.receive_now = 100.1
        o.vehicle_yaw = .1
        target = o.get_local_centerline_target(4)
        self.assertEqual(o.local_centerline_map_b, .2)
        self.assertEqual(target['observation_stamp'], 10.0)
        self.assertAlmostEqual(target['confidence'], .8)

    def test_controller_one_pair_preserves_map_direction(self):
        p, o = self.planner(), self.controller()
        p.cones = self.cones(.02, .1)
        self.transfer_plan(p, o)
        self.now = p.last_cone_time = 10.1
        p.cones = p.cones[:2]
        p.vehicle_yaw = o.vehicle_yaw = .25
        self.transfer_plan(p, o)
        target = o.get_local_centerline_target(4)
        self.assertOnLine((target['x'], target['y']), .02, .1)
        self.assertEqual(target['pairs'], 1)

    def test_controller_map_slope_bound_is_independent_of_vehicle_yaw(self):
        p, o = self.planner(), self.controller()
        p.vehicle_yaw = o.vehicle_yaw = math.radians(20)
        p.cones = self.cones()
        self.transfer_plan(p, o)
        target = o.get_local_centerline_target(4)
        self.assertIsNotNone(target)
        self.assertOnLine((target['x'], target['y']), 0, 0)

    def test_same_side_clutter_cannot_form_a_complete_pair(self):
        p = self.planner()
        p.cones = [(5, 1.5, 1), (7, 1.5, 1)]
        self.assertIsNone(p.estimate_centerline())
        self.assertFalse(p.update_centerline())

    def test_nearby_clutter_does_not_hide_farther_complete_pairs(self):
        p = self.planner()
        p.cones = [(x, 0, 1) for x in (1, 1.5, 2, 2.5)] + self.cones()
        self.assertTrue(p.update_centerline())
        self.assertEqual(p.centerline_pairs, 2)
        self.assertAlmostEqual(p.centerline_b, 0)

    def test_pairs_do_not_reuse_cones_or_duplicate_rows(self):
        p = self.planner()
        p.cones = [(5, -1.5, 1), (5, 1.5, 1), (5.2, -1.4, 1), (5.2, 1.4, 1)]
        pairs = p.pair_cones(p.select_local_cones())
        self.assertEqual(len(pairs), 1)
        self.assertNotEqual(pairs[0][5], pairs[0][6])

    def test_quality_decreases_for_bad_width_alignment_and_confidence(self):
        args = (3.0, 2.3, 3.7, 1.5)
        perfect = pair_quality(3, 0, 1, 1, *args)
        self.assertEqual(perfect, 1)
        self.assertLess(pair_quality(3.4, 0, 1, 1, *args), perfect)
        self.assertLess(pair_quality(3, .5, 1, 1, *args), perfect)
        self.assertLess(pair_quality(3, 0, .4, 1, *args), perfect)
        self.assertEqual(pair_quality(3.7, 0, 1, 1, *args), 0)

    def test_better_confidence_wins_between_duplicate_row_candidates(self):
        p = self.planner()
        p.cones = [(5, -1.5, .3), (5, -1.5, 1), (5, 1.5, 1)]
        pairs = p.pair_cones(p.select_local_cones())
        self.assertEqual(len(pairs), 1)
        self.assertEqual(pairs[0][8], 1)

    def test_quality_changes_actual_control_blend(self):
        o = self.controller()
        strong = o.local_blend_weight(o.get_local_centerline_target(4))
        self.set_controller_line(o, confidence=.4)
        weak = o.local_blend_weight(o.get_local_centerline_target(4))
        self.assertAlmostEqual(strong, .65)
        self.assertAlmostEqual(weak, .26)
        self.assertEqual(o.local_blend_weight(dict(confidence=0, source='two_pairs')), 0)

    def test_weak_evidence_changes_authoritative_path_less(self):
        changes = []
        for confidence in (1, .4):
            p = self.planner()
            p.cones = self.cones()
            p.update_centerline()
            self.now = p.last_cone_time = 10.1
            p.cones = [(x, y, confidence) for x, y, _ in self.cones(0, .3)]
            self.assertTrue(p.update_centerline())
            changes.append(p.centerline_b)
        self.assertGreater(changes[0], changes[1])
        self.assertAlmostEqual(changes[1], changes[0] * .4)

    def test_actual_control_loop_uses_path_only_at_all_valid_qualities(self):
        errors = []
        for confidence in (1, .4):
            p, o = self.planner(), self.controller()
            p.cones = [(x, y, confidence) for x, y, _ in self.cones(0, .3)]
            self.transfer_plan(p, o)
            o.check_position_sources = lambda: None
            o.finish_triggered = False
            o.pose_received = o.is_started = True
            o.pose_age = 0
            o.input_reason = 'valid'
            o.vehicle_speed = 1
            o.slam_pose_received = True
            o.fssim_pose_received = False
            o.total_distance = 0
            o.visited_indices = set()
            o.target_point_index = 0
            o.lookahead_distance = 4
            o.control_mode = 'slam'
            o.prev_time = Stamp(to_sec=lambda: 9.95)
            o.publish_input_status = lambda reason: None
            o.update_speed_source = lambda: None
            o.update_dynamic_lookahead = lambda: 4
            o.find_lookahead_point = lambda: p.paths[-1].poses[8]
            o.get_local_centerline_target = lambda *args: self.fail('Second target entered control')
            o.filter_tracking_errors = lambda lateral, heading: (lateral, heading)
            o.pid_control = lambda lateral, heading, dt: lateral
            o.can_client_ok = False
            o.publish_steering_command = lambda angle: None
            rows = []
            o.logger = Object(log_control_data=rows.append)
            o._control_loop(None)
            errors.append(rows[-1]['lateral_error'])
            self.assertEqual(rows[-1]['centerline_blend_weight'], 0)
            self.assertEqual(rows[-1]['lookahead_mode'], 'planned_path_only')
        self.assertAlmostEqual(errors[0], errors[1])

    def test_initial_road_direction_need_not_match_map_x(self):
        p = self.planner()
        k = math.tan(math.radians(8))
        p.cones = self.cones(k, .1)
        self.assertTrue(p.update_centerline())
        self.assertAlmostEqual(p.centerline_k, k)
        self.assertAlmostEqual(p.track_reference_k, k)
        for yaw in (-.2, .3):
            p.vehicle_yaw = yaw
            points, ready, _ = p.generate_local_path()
            self.assertTrue(ready)
            for point in points[16:]:
                self.assertOnLine(point, k, .1)

    def test_close_rows_cannot_establish_direction(self):
        p = self.planner()
        p.cones = [(x, .04*x + side*1.5, 1)
                   for x in (5, 7) for side in (-1, 1)]
        self.assertTrue(p.update_centerline())
        self.assertIsNone(p.track_reference_k)
        self.assertEqual(p.centerline_k, 0)

    def test_missing_five_metre_row_still_allows_direction(self):
        p = self.planner()
        p.cones = [(x, .02*x + side*1.5, 1)
                   for x in (5, 15) for side in (-1, 1)]
        self.assertTrue(p.update_centerline())
        self.assertAlmostEqual(p.track_reference_k, .02)

    def test_two_rows_acquire_direction_after_one_pair_start(self):
        p = self.planner()
        p.cones = self.cones(.1, 0)[:2]
        p.update_centerline()
        self.assertIsNone(p.track_reference_k)
        self.now = p.last_cone_time = 10.1
        p.cones = self.cones(.1, 0)
        self.assertTrue(p.update_centerline())
        self.assertAlmostEqual(p.track_reference_k, .1)
        self.assertAlmostEqual(p.centerline_k, .1)
        self.assertAlmostEqual(p.centerline_b, 0)

    def test_zero_quality_prevents_control_loop_tracking_output(self):
        o = self.controller()
        self.set_controller_line(o, confidence=0)
        o.check_position_sources = lambda: None
        o.finish_triggered = False
        o.pose_received = True
        o.input_reason = 'valid'
        reasons = []
        o.publish_input_status = reasons.append
        o.pid_control = lambda *args: self.fail('Invalid quality reached tracking')
        o._control_loop(None)
        self.assertEqual(reasons, ['path_or_observation_stale'])

    def test_single_side_is_lower_quality_and_lower_blend(self):
        p, o = self.planner(), self.controller()
        p.cones = self.cones()
        self.transfer_plan(p, o)
        strong = o.local_blend_weight(o.get_local_centerline_target(4))
        self.now = p.last_cone_time = 10.1
        p.cones = [(5, 1.5, 1)]
        self.transfer_plan(p, o)
        target = o.get_local_centerline_target(4)
        self.assertEqual(target['source'], 'single_side')
        self.assertEqual(target['pairs'], 0)
        self.assertAlmostEqual(target['confidence'], .3)
        self.assertAlmostEqual(o.local_blend_weight(target), .06)
        self.assertLess(o.local_blend_weight(target), strong)

    def test_republishing_a_held_line_does_not_refresh_evidence_quality(self):
        p, o = self.planner(), self.controller()
        p.cones = self.cones()
        self.transfer_plan(p, o)
        self.now = p.last_cone_time = 10.4
        self.receive_now = 100.4
        p.cones = []
        self.transfer_plan(p, o)
        target = o.get_local_centerline_target(4)
        self.assertEqual(target['observation_stamp'], 10.0)
        self.assertAlmostEqual(target['confidence'], .2)
        self.assertAlmostEqual(o.local_blend_weight(target), .13)
        self.now = 10.5
        self.assertFalse(o.path_is_fresh())

    def test_mismatched_path_or_observation_cannot_supply_a_target(self):
        o = self.controller()
        o.planning_status['path_stamp'] = 10.1
        self.assertFalse(o.path_is_fresh())
        self.assertIsNone(o.get_local_centerline_target(4))
        self.set_controller_line(o)
        o.planning_status['centerline']['observation_stamp'] = 9.9
        self.assertIsNone(o.get_local_centerline_target(4))

    def test_missing_or_malformed_quality_inhibits_tracking(self):
        o = self.controller()
        del o.planning_status['centerline']
        self.assertFalse(o.path_is_fresh())
        bad_values = [None, 'bad', float('nan'), float('inf'), -1, 2, True, 0]
        for value in bad_values:
            self.set_controller_line(o, confidence=value)
            self.assertFalse(o.path_is_fresh())
        for field in ('map_k', 'map_b'):
            self.set_controller_line(o)
            o.planning_status['centerline'][field] = float('nan')
            self.assertIsNone(o.get_local_centerline_target(4))

    def test_source_and_pair_count_must_agree(self):
        o = self.controller()
        self.set_controller_line(o, source='single_side', pairs=2)
        self.assertFalse(o.path_is_fresh())
        self.set_controller_line(o)
        o.planning_status['pairs'] = 1
        self.assertFalse(o.path_is_fresh())
        o.planning_status['pairs'] = True
        self.assertFalse(o.path_is_fresh())
        self.set_controller_line(o, source='unknown')
        self.assertFalse(o.path_is_fresh())

    def test_non_dictionary_status_clears_previous_valid_result(self):
        o = self.controller()
        self.assertTrue(o.path_is_fresh())
        o._planning_status_callback(Object(data='[]'))
        self.assertFalse(o.path_is_fresh())
        self.assertIsNone(o.get_local_centerline_target(4))

    def test_invalid_new_observation_does_not_refresh_old_line_quality(self):
        p, o = self.planner(), self.controller()
        p.cones = self.cones()
        self.transfer_plan(p, o)
        self.now = p.last_cone_time = 10.2
        self.receive_now = 100.2
        p.cones = self.cones(.1, .2)
        self.transfer_plan(p, o)
        target = o.get_local_centerline_target(4)
        self.assertAlmostEqual(o.local_centerline_map_k, 0)
        self.assertEqual(target['observation_stamp'], 10.0)
        self.assertAlmostEqual(target['confidence'], .6)


if __name__ == '__main__':
    unittest.main()
