# -*- coding: utf-8 -*-
"""Failure injection without ROS, CAN sockets or physical vehicle output."""
import json
import ast
import os
import sys
import threading
import unittest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from legacy_run_guard import LegacyRunGuard
from test_lidar_fusion_node import load_node, Obj, Scalar, Publisher


def node_methods(filename, names, namespace):
    with open(os.path.join(os.path.dirname(__file__), '..', 'src', filename), 'rb') as stream:
        tree = ast.parse(stream.read(), filename)
    body = [method for cls in tree.body if isinstance(cls, ast.ClassDef)
            for method in cls.body
            if isinstance(method, ast.FunctionDef) and method.name in names]
    module = ast.Module(body=body)
    if 'type_ignores' in ast.Module._fields:
        module.type_ignores = []
    eval(compile(ast.fix_missing_locations(module), filename, 'exec'), namespace)
    return type('Subject', (object,), dict((name, namespace[name]) for name in names))()


class GuardTests(unittest.TestCase):
    def fill(self, g, t):
        for name in g.limits:
            g.observe(name, t)

    def test_missing_at_start_is_recoverable(self):
        g = LegacyRunGuard()
        self.assertFalse(g.start(10))
        self.assertIsNone(g.stop_reason)
        self.fill(g, 20)
        self.assertTrue(g.start(20))

    def test_stationary_fresh_messages_start_and_keep_running(self):
        g = LegacyRunGuard()
        for t in (10, 11, 12, 13):
            self.fill(g, t)
            self.assertTrue(g.start(t))
            self.assertEqual([], g.evaluate(t))
            self.assertIsNone(g.stop_reason)

    def test_single_missing_frame_is_tolerated(self):
        g = LegacyRunGuard()
        self.fill(g, 10)
        g.start(10)
        self.assertEqual([], g.evaluate(10.2))

    def test_each_required_stream_can_stop_and_latch(self):
        for missing in ('pose', 'imu', 'path', 'road'):
            g = LegacyRunGuard()
            self.fill(g, 10)
            g.start(10)
            for name in g.limits:
                if name != missing:
                    g.observe(name, 12)
            g.evaluate(12)
            self.assertEqual(missing+'_stale', g.stop_reason)
            self.fill(g, 13)
            self.assertFalse(g.start(13))
            self.assertEqual(missing+'_stale', g.stop_reason)

    def test_republication_does_not_extend_observation_life(self):
        g = LegacyRunGuard()
        self.fill(g, 10)
        g.start(10)
        self.fill(g, 12)
        g.observe('road', 12, source_age=2)
        g.evaluate(12)
        self.assertEqual('road_stale', g.stop_reason)

    def test_clock_rewind_before_start_recovers_after_start_latches(self):
        g = LegacyRunGuard()
        g.evaluate(20)
        g.evaluate(10)
        self.assertIsNone(g.stop_reason)
        self.fill(g, 10)
        g.start(10)
        g.evaluate(9)
        self.assertEqual('clock_rewind', g.stop_reason)

    def test_bad_parameters_and_nonfinite_evidence(self):
        for value in (0, -1, float('nan'), float('inf')):
            self.assertRaises(ValueError, LegacyRunGuard, value)
        g = LegacyRunGuard()
        self.assertFalse(g.observe('road', 10, float('nan')))
        self.assertFalse(g.observe('road', 10, -1))


class ActualControllerTests(unittest.TestCase):
    def setUp(self):
        module, self.clock, codec = load_node()
        self.cls = module.LidarTrackingController.__bases__[0]
        ns = self.cls.control_loop.__globals__
        ns['time'] = Obj(time=lambda: self.clock.now)
        ns['rospy'].loginfo_throttle = lambda *a: None
        ns['rospy'].logerr_throttle = lambda *a: None
        ns['rospy'].logerr = lambda *a: None
        self.o = self.cls.__new__(self.cls)
        o = self.o
        o.state_lock = threading.RLock()
        o.shutting_down = False
        o.run_guard = LegacyRunGuard()
        o.is_started = False
        o.pose_received = o.path_received = True
        o.finish_triggered = False
        o.vehicle_x = o.vehicle_y = o.vehicle_yaw = 0
        o.current_path = Obj(poses=[Obj(), Obj()])
        o.control_mode = 'slam'
        o.guard_pub = Publisher()
        o.can_client_ok = True
        o.enable_vehicle_output = True
        o.can_client = Obj(send_steering_frame=self.send)
        o.publish_steering_command = lambda v: None
        o.update_finish_distance = lambda: None
        o.check_position_sources = lambda: None
        o._legacy_control_loop = lambda event: self.normal.append(True)
        self.frames, self.normal = [], []

    def send(self, angle, stop=False):
        self.frames.append((angle, stop))
        return True

    def fill(self):
        for name in self.o.run_guard.limits:
            self.o.run_guard.observe(name, self.clock.now)

    def test_waiting_does_not_send_stop_then_starts_without_movement(self):
        self.o.control_loop(None)
        self.assertFalse(self.o.is_started)
        self.assertEqual([], self.frames)
        self.fill()
        self.o.control_loop(None)
        self.assertTrue(self.o.is_started)
        self.assertEqual([True], self.normal)
        self.assertEqual([], self.frames)

    def test_loss_repeats_stop_even_with_no_path_and_recovers_no_auto_resume(self):
        self.fill()
        self.o.control_loop(None)
        self.clock.now += 2
        self.o.path_received = False
        self.o.control_loop(None)
        self.fill()
        self.o.path_received = True
        self.o.control_loop(None)
        self.assertEqual([(0.0, True), (0.0, True)], self.frames)
        self.assertEqual([True], self.normal)

    def test_control_exception_stops_same_tick(self):
        self.fill()
        def fail(event):
            raise ValueError('injected')
        self.o._legacy_control_loop = fail
        self.o.control_loop(None)
        self.assertEqual([(0.0, True)], self.frames)
        self.assertEqual('control_exception', self.o.run_guard.stop_reason)

    def test_each_control_tick_refreshes_pose_instead_of_one_second_timer(self):
        o = self.o
        o.check_position_sources = self.cls.check_position_sources.__get__(o, self.cls)
        o.fssim_available = o.slam_pose_received = True
        o.last_fssim_time = None
        o.last_slam_time = 10
        o.slam_source_age = 0
        o.slam_x, o.slam_y, o.slam_yaw = 1, 2, .1
        o.slam_speed_estimate = 1
        o.max_vehicle_speed_mps = 4.1667
        o.check_start_condition = lambda: None
        o.control_loop(None)
        self.assertEqual((1, 2, .1), (o.vehicle_x, o.vehicle_y, o.vehicle_yaw))
        self.clock.now = 10.05
        o.last_slam_time = 10.05
        o.slam_x, o.slam_y, o.slam_yaw = 1.1, 2.1, .2
        o.control_loop(None)
        self.assertEqual((1.1, 2.1, .2), (o.vehicle_x, o.vehicle_y, o.vehicle_yaw))

    def test_shutdown_cannot_be_followed_by_control_output(self):
        self.fill()
        self.o.shutting_down = True
        self.o.control_loop(None)
        self.assertEqual([], self.frames)
        self.assertEqual([], self.normal)

    def test_stale_road_status_republication_cannot_enable_start(self):
        self.fill()
        self.o.guard_road_callback(Scalar(json.dumps(dict(
            valid=True, observation_age_s=5, stamp=self.clock.now,
            frame_origin=[0,0], frame_yaw=0, k=0, b=0, pairs=2))))
        self.o.control_loop(None)
        self.assertFalse(self.o.is_started)
        self.assertIsNone(self.o.run_guard.stop_reason)
        self.assertEqual([], self.frames)

    def test_finish_can_stop_without_a_path(self):
        self.fill()
        self.o.control_loop(None)
        self.o.path_received = False
        self.o.finish_triggered = True
        self.o.control_loop(None)
        self.assertEqual('finish', self.o.run_guard.stop_reason)
        self.assertEqual([(0.0, True)], self.frames)


class MeasurementAgeTests(unittest.TestCase):
    def test_planner_repeated_fit_retains_measurement_time(self):
        noop = lambda *args: None
        planner = node_methods('zhixian_realtime.py', ['update_centerline'], dict(
            rospy=Obj(loginfo=noop, logwarn_throttle=noop, loginfo_throttle=noop)))
        planner.estimate_centerline = lambda: (0.0, 0.2, 2, True)
        planner.centerline_valid = False
        planner.centerline_alpha = .15
        planner.planning_started = False
        planner.last_cone_time = 10.0
        planner.update_centerline()
        planner.update_centerline()
        self.assertEqual(10.0, planner.centerline_last_valid_time)

    def test_map_publish_retains_last_observation_stamp(self):
        mapping = node_methods('jiantu.py', ['publish_cone_map'], dict(
            ConeArray=lambda: Obj(cones=[]), Header=Obj))
        mapping.cone_lock = threading.RLock()
        mapping.cone_map = []
        mapping.last_cone_observation_stamp = 10.0
        mapping.cone_map_pub = Publisher()
        mapping.publish_cone_map()
        mapping.publish_cone_map()
        self.assertEqual([10.0, 10.0], [m.header.stamp for m in mapping.cone_map_pub.messages])


if __name__ == '__main__':
    unittest.main()
