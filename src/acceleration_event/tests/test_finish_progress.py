#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Python 2/3 offline finish regressions: actual control methods, mocked outputs."""
from __future__ import print_function
import json
import math
import unittest

import test_centerline_geometry as geometry
from data_quality import LongitudinalProgress, PoseBuffer


class ProgressTests(unittest.TestCase):
    def test_first_measurement_freezes_reference(self):
        p = LongitudinalProgress()
        self.assertTrue(p.update(10, 12, -3))
        self.assertEqual(p.origin, (12, -3))
        self.assertEqual(p.reference_stamp, 10)
        p.update(11, 20, 8)
        self.assertEqual(p.origin, (12, -3))
        self.assertEqual(p.progress, 8)

    def test_configured_start_does_not_shift_to_first_measurement(self):
        p = LongitudinalProgress(origin=(10, 5))
        p.update(10, 30, 5)
        self.assertEqual(p.progress, 20)
        self.assertEqual(p.origin, (10, 5))
        self.assertIsNone(p.reference_stamp)

    def test_static_noise_does_not_accumulate_distance(self):
        p = LongitudinalProgress(origin=(0, 0))
        for i in range(1500):
            p.update(10 + .1*i, .05 if i % 2 else -.05, .1 if i % 2 else -.1)
        self.assertLessEqual(abs(p.progress), .05)
        self.assertFalse(p.finished)

    def test_lateral_movement_cannot_increase_progress(self):
        p = LongitudinalProgress(origin=(0, 0))
        for i in range(1000):
            p.update(10 + i, 20, 2 if i % 2 else -2)
        self.assertEqual(p.progress, 20)
        self.assertFalse(p.finished)

    def test_reverse_motion_reduces_progress_without_accumulating_retravel(self):
        p = LongitudinalProgress(origin=(0, 0))
        for stamp, x in ((10, 0), (20, 40), (30, 0), (40, 40), (50, -5)):
            p.update(stamp, x, 0)
        self.assertEqual(p.progress, -5)
        self.assertFalse(p.finished)

    def test_rotated_axis_ignores_perpendicular_displacement(self):
        yaw = math.radians(30)
        c, s = math.cos(yaw), math.sin(yaw)
        p = LongitudinalProgress(axis_yaw=yaw, origin=(10, 20))
        p.update(10, 10+74*c-5*s, 20+74*s+5*c)
        self.assertAlmostEqual(p.progress, 74)
        self.assertFalse(p.finished)
        p.update(11, 10+75.1*c-5*s, 20+75.1*s+5*c)
        self.assertTrue(p.finished)

    def test_finish_threshold_and_latch(self):
        p = LongitudinalProgress(origin=(0, 0))
        p.update(10, 74.99, 0)
        self.assertFalse(p.finished)
        p.update(11, 75, 0)
        self.assertTrue(p.finished)
        self.assertFalse(p.update(12, 0, 0))
        self.assertTrue(p.finished)
        self.assertEqual(p.progress, 75)

    def test_duplicate_and_reverse_timestamps_do_not_confirm(self):
        p = LongitudinalProgress(origin=(0, 0), confirm_samples=2)
        p.update(10, 75.1, 0)
        self.assertFalse(p.update(10, 76, 0))
        self.assertFalse(p.update(9, 76, 0))
        self.assertEqual(p.confirm_count, 1)
        self.assertFalse(p.finished)
        p.update(11, 75.2, 0)
        self.assertTrue(p.finished)

    def test_optional_confirmation_rejects_a_single_crossing_spike(self):
        p = LongitudinalProgress(origin=(0, 0), confirm_samples=2)
        p.update(10, 75.1, 0)
        p.update(11, 74.9, 0)
        self.assertEqual(p.confirm_count, 0)
        p.update(12, 75.1, 0)
        self.assertFalse(p.finished)
        p.update(13, 75.2, 0)
        self.assertTrue(p.finished)

    def test_invalid_measurements_cannot_establish_reference(self):
        p = LongitudinalProgress()
        for stamp, x, y in ((0, 0, 0), (10, float('nan'), 0),
                            (10, 0, float('inf')), (True, 0, 0)):
            self.assertFalse(p.update(stamp, x, y))
        self.assertIsNone(p.origin)
        self.assertIsNone(p.last_stamp)

    def test_invalid_parameters_fail_before_control_starts(self):
        for value in (0, -1, True, float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                LongitudinalProgress(distance=value)
        for value in (0, -1, True, 1.5):
            with self.assertRaises(ValueError):
                LongitudinalProgress(confirm_samples=value)
        with self.assertRaises(ValueError):
            LongitudinalProgress(axis_yaw=float('nan'))
        with self.assertRaises(ValueError):
            LongitudinalProgress(origin=(0, float('inf')))


class FinishControlTests(unittest.TestCase):
    def setUp(self):
        self.fixture = geometry.GeometryTests('test_map_local_round_trip_with_translation_and_rotation')
        self.fixture.setUp()
        namespace = self.fixture.namespace()
        namespace['Cmd'] = namespace['Float64'] = geometry.Object
        names = ['check_position_sources', 'update_finish_distance', 'finish_status',
                 '_control_loop', 'hold_normal_stop', 'publish_input_status',
                 'publish_steering_command']
        self.o = geometry.node_methods('genzong_realtime.py', names, namespace)
        o = self.o
        o.localization_mode = 'slam'
        o.control_mode = 'none'
        o.pose_timeout = .5
        # Keep both ROS and receive clocks deterministic while exercising real buffers.
        for name in ('slam_buffer', 'fssim_buffer'):
            buffer = PoseBuffer()
            setattr(o, name, geometry.Object(add=buffer.add,
                latest=lambda now, timeout, buffer=buffer: buffer.latest(
                    now, timeout, receive_now=self.fixture.receive_now)))
        o.check_start_condition = lambda: None
        o.finish_start_mode = 'first_valid_pose'
        o.finish_progress = LongitudinalProgress()
        o.finish_triggered = False
        o.total_distance = 0
        o.pose_received = False
        o.pose_stamp = None
        o.pose_age = None
        o.input_reason = 'missing'
        o.path_is_fresh = lambda: False
        self.can_frames, self.sim_commands, self.steering, self.status, self.rows = [], [], [], [], []
        o.enable_vehicle_output = True
        o.can_client_ok = True
        o.can_client = geometry.Object(send_steering_frame=lambda angle, stop:
                                      self.can_frames.append((angle, stop)))
        o.cmd_pub = geometry.Object(publish=self.sim_commands.append)
        o.steering_pub = geometry.Object(publish=self.steering.append)
        o.quality_pub = geometry.Object(publish=self.status.append)
        o.logger = geometry.Object(log_control_data=self.rows.append)
        o.pid_control = lambda *args: self.fail('Invalid path reached steering calculation')

    def sample(self, stamp, x, y=0, yaw=0):
        self.fixture.now = stamp
        self.fixture.receive_now = 100 + stamp - 10
        buffer = self.o.slam_buffer if self.o.localization_mode == 'slam' else self.o.fssim_buffer
        self.assertTrue(buffer.add(stamp, x, y, yaw, received=self.fixture.receive_now))
        self.o._control_loop(None)

    def test_reference_and_progress_update_while_waiting_for_path(self):
        self.sample(10, 12, 3)
        self.assertEqual(self.o.finish_progress.origin, (12, 3))
        self.sample(10.1, 13, 3.2)
        self.assertAlmostEqual(self.o.total_distance, 1)
        self.assertEqual(self.o.finish_progress.origin, (12, 3))
        self.assertEqual(self.can_frames, [])
        state = json.loads(self.status[-1].data)
        self.assertEqual(state['finish_reference_x'], 12)
        self.assertEqual(state['finish_reference_stamp'], 10)
        self.assertEqual(state['finish_progress'], 1)

    def test_finish_stops_before_consulting_stale_path(self):
        self.o.finish_progress = LongitudinalProgress(origin=(0, 0))
        self.o.finish_start_mode = 'configured'
        self.sample(10, 74)
        self.o.path_is_fresh = lambda: self.fail('Finish consulted path validity')
        self.sample(10.1, 75.1)
        self.assertTrue(self.o.finish_triggered)
        self.assertEqual(self.can_frames, [(0.0, True)])
        self.assertAlmostEqual(self.rows[-1]['finish_progress'], 75.1)
        self.assertAlmostEqual(self.rows[-1]['total_distance'], 75.1)
        self.assertEqual(self.rows[-1]['input_reason'], 'normal_stop_latched')

    def test_default_reference_does_not_latch_just_because_map_x_is_large(self):
        self.sample(10, 100)
        self.assertEqual(self.o.finish_progress.origin, (100, 0))
        self.assertEqual(self.o.total_distance, 0)
        self.assertFalse(self.o.finish_triggered)

    def test_configured_reference_can_finish_on_first_valid_snapshot(self):
        self.o.finish_start_mode = 'configured'
        self.o.finish_progress = LongitudinalProgress(origin=(0, 0))
        self.sample(10, 75.1)
        self.assertTrue(self.o.finish_triggered)
        self.assertEqual(self.can_frames, [(0.0, True)])

    def test_invalid_and_stale_localization_cannot_move_progress(self):
        self.sample(10, 12)
        self.fixture.now = 11
        self.fixture.receive_now = 101
        self.o._control_loop(None)
        self.assertFalse(self.o.pose_received)
        self.assertEqual(self.o.total_distance, 0)
        self.assertEqual(self.o.finish_progress.origin, (12, 0))
        self.assertEqual(self.can_frames, [])
        # Position jumps rejected by the localization buffer never reach progress.
        self.fixture.now = 11.1
        self.assertFalse(self.o.slam_buffer.add(11.1, 100, 0, 0, received=101.1))
        self.o._control_loop(None)
        self.assertEqual(self.o.total_distance, 0)
        self.assertFalse(self.o.finish_triggered)

    def test_repeated_control_cycles_do_not_count_as_new_finish_measurements(self):
        self.o.finish_progress = LongitudinalProgress(origin=(0, 0), confirm_samples=2)
        self.sample(10, 75.1)
        for unused in range(5):
            self.o._control_loop(None)
        self.assertEqual(self.o.finish_progress.confirm_count, 1)
        self.assertFalse(self.o.finish_triggered)
        self.sample(10.1, 75.2)
        self.assertTrue(self.o.finish_triggered)

    def test_source_loss_resets_pending_confirmation_not_reference(self):
        self.o.finish_progress = LongitudinalProgress(origin=(0, 0), confirm_samples=2)
        self.sample(10, 75.1)
        self.fixture.now = 11
        self.fixture.receive_now = 101
        self.o._control_loop(None)
        self.assertEqual(self.o.finish_progress.confirm_count, 0)
        self.sample(11.1, 75.2)
        self.assertFalse(self.o.finish_triggered)
        self.assertEqual(self.o.finish_progress.origin, (0, 0))
        self.sample(11.2, 75.3)
        self.assertTrue(self.o.finish_triggered)

    def test_fssim_zero_drive_persists_after_localization_expires(self):
        self.o.localization_mode = 'fssim'
        self.o.finish_progress = LongitudinalProgress(origin=(0, 0))
        self.sample(10, 75.1)
        self.fixture.now = 11
        self.fixture.receive_now = 101
        self.o._control_loop(None)
        self.assertEqual(self.o.control_mode, 'none')
        self.assertTrue(self.o.finish_triggered)
        self.assertEqual(self.can_frames, [(0.0, True), (0.0, True)])
        self.assertEqual([(cmd.dc, cmd.delta) for cmd in self.sim_commands], [(0.0, 0.0)] * 2)

    def test_validation_mode_finishes_without_vehicle_output(self):
        self.o.enable_vehicle_output = False
        self.o.can_client_ok = False
        self.o.localization_mode = 'fssim'
        self.o.finish_progress = LongitudinalProgress(origin=(0, 0))
        self.sample(10, 75.1)
        self.assertTrue(self.o.finish_triggered)
        self.assertEqual(self.can_frames, [])
        self.assertEqual(self.sim_commands, [])
        self.assertEqual(self.steering, [])
        self.assertTrue(json.loads(self.status[-1].data)['stop_latched'])

    def test_missing_localization_cannot_establish_default_reference(self):
        self.o._control_loop(None)
        self.assertIsNone(self.o.finish_progress.origin)
        self.assertEqual(self.can_frames, [])


if __name__ == '__main__':
    unittest.main()
