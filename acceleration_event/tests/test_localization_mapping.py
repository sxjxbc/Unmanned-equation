# -*- coding: utf-8 -*-
from __future__ import print_function
import ast
import math
import os
import threading
import unittest
import numpy as np
from test_centerline_geometry import Object, node_methods, SRC
from data_quality import PoseBuffer, StableOrigin, gps_quality_valid, fresh


def classes():
    with open(os.path.join(SRC, 'jiantu.py'), 'rb') as stream:
        tree = ast.parse(stream.read())
    body = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name in ('ConeObject', 'CoordinateTransformer')]
    module = ast.Module(body=body)
    if 'type_ignores' in ast.Module._fields:
        module.type_ignores = []
    ns = dict(np=np, math=math)
    eval(compile(ast.fix_missing_locations(module), 'jiantu.py', 'exec'), ns)
    return ns


class QualityTests(unittest.TestCase):
    def test_valid_and_excessive_covariance(self):
        covariance = [.1, 0., 0., 0., .1, 0., 0., 0., .2]
        for status in (0, 1, 2):
            self.assertTrue(gps_quality_valid(status, 2, covariance, .25))
        covariance[4] = .3
        self.assertFalse(gps_quality_valid(0, 2, covariance, .25))

    def test_unknown_requires_explicit_opt_in_and_no_fix_always_rejected(self):
        self.assertFalse(gps_quality_valid(0, 0, [0.] * 9, .25))
        self.assertTrue(gps_quality_valid(0, 0, [0.] * 9, .25, True))
        self.assertFalse(gps_quality_valid(-1, 0, [0.] * 9, .25, True))

    def test_malformed_covariance(self):
        for covariance in ([0.] * 9, [float('nan')] * 9, [.1] * 8,
                           [.1, .2, 0., .2, .1, 0., 0., 0., .1],
                           [.1, .01, 0., 0., .1, 0., 0., 0., .1]):
            self.assertFalse(gps_quality_valid(0, 3, covariance, .25))

    def test_stationary_average_and_wrapped_yaw(self):
        origin = StableOrigin(duration=1., min_samples=5, max_gap=.3)
        result = None
        for i in range(6):
            result = origin.add(10. + i * .2, 35. + (i % 2) * .000001,
                                120., math.radians(179.5 if i % 2 else -179.5))
        self.assertIsNotNone(result)
        self.assertAlmostEqual(result[0], 35.0000005)
        self.assertAlmostEqual(abs(result[2]), math.pi)

    def test_gap_resets_initialization(self):
        origin = StableOrigin(duration=1., min_samples=3, max_gap=.5)
        origin.add(10., 35., 120., 0.)
        origin.add(10.4, 35., 120., 0.)
        self.assertIsNone(origin.add(11., 35., 120., 0.))
        self.assertEqual(len(origin.samples), 1)

    def test_movement_and_yaw_reset_window(self):
        for lat, yaw in [(35.0001, 0.), (35., .5)]:
            origin = StableOrigin(duration=1., min_samples=3)
            origin.add(10., 35., 120., 0.)
            self.assertIsNone(origin.add(10.2, lat, 120., yaw))
            self.assertEqual(len(origin.samples), 1)

    def test_duplicates_do_not_build_support(self):
        origin = StableOrigin(duration=1., min_samples=3)
        origin.add(10., 35., 120., 0.)
        for stamp in (10., 9.):
            self.assertIsNone(origin.add(stamp, 35., 120., 0.))
        self.assertEqual(len(origin.samples), 1)

    def test_exact_alignment_does_not_require_adjacent_sample(self):
        buffer = PoseBuffer()
        buffer.add(10., 0., 0., 0.)
        buffer.add(10.2, 0., 0., .1)
        self.assertIsNotNone(buffer.at(10.2, .15))
        self.assertIsNone(buffer.at(10.1, .05))

    def test_invalid_initialization_parameters(self):
        for args in (dict(duration=0.), dict(min_samples=1), dict(min_samples=2.5), dict(max_gap=float('nan'))):
            with self.assertRaises(ValueError):
                StableOrigin(**args)


class MappingTests(unittest.TestCase):
    def setUp(self):
        self.ns = classes()

    def subject(self):
        subject = node_methods('jiantu.py', ['update_cone_frame'], dict(math=math, ConeObject=self.ns['ConeObject']))
        subject.cone_lock = threading.RLock()
        subject.cone_map = []
        subject.cone_timeout = 3.
        subject.cone_merge_distance = .8
        subject.cone_merge_count = subject.cone_new_count = 0
        return subject

    def test_confidence_support_is_monotonic_with_constant_detector_quality(self):
        cone = self.ns['ConeObject'](0., 0., 10., .6)
        scores = [cone.confidence]
        for i in range(1, 7):
            cone.update(0., 0., 10. + i, .6)
            scores.append(cone.confidence)
        self.assertEqual(scores, sorted(scores))
        self.assertLessEqual(max(scores), .6)
        cone.update(0., 0., 20., .1)
        self.assertLessEqual(cone.confidence, .1)

    def test_duplicate_timestamp_does_not_inflate_support(self):
        cone = self.ns['ConeObject'](0., 0., 10., .6)
        self.assertFalse(cone.update(1., 0., 10., 1.))
        self.assertEqual(len(cone.observations), 1)
        self.assertEqual(cone.x, 0.)

    def test_fragments_update_existing_object_once(self):
        subject = self.subject()
        subject.update_cone_frame([(0., 0., .8)], 10.)
        subject.update_cone_frame([(.05, 0., .8), (.1, 0., .8)], 10.1)
        self.assertEqual(len(subject.cone_map), 1)
        self.assertEqual(len(subject.cone_map[0].observations), 2)
        self.assertEqual(subject.cone_merge_count, 1)

    def test_new_fragments_keep_strongest_detection(self):
        subject = self.subject()
        subject.update_cone_frame([(0., 0., .4), (.1, 0., .8)], 10.)
        self.assertEqual(len(subject.cone_map), 1)
        self.assertAlmostEqual(subject.cone_map[0].x, .1)
        self.assertEqual(subject.cone_map[0].detection_confidence, .8)

    def test_distinct_cones_preserve_identity(self):
        subject = self.subject()
        subject.update_cone_frame([(0., -1.5, .8), (0., 1.5, .8)], 10.)
        objects = list(subject.cone_map)
        subject.update_cone_frame([(.1, 1.5, .8), (.1, -1.5, .8)], 10.1)
        self.assertEqual(subject.cone_map, objects)
        self.assertEqual([len(c.observations) for c in objects], [2, 2])

    def test_expired_cone_is_not_resurrected(self):
        subject = self.subject()
        subject.update_cone_frame([(0., 0., .8)], 10.)
        old = subject.cone_map[0]
        subject.update_cone_frame([(0., 0., .8)], 14.)
        self.assertIsNot(subject.cone_map[0], old)
        self.assertEqual(len(subject.cone_map[0].observations), 1)

    def test_gps_callback_requires_new_aligned_stable_quality_samples(self):
        now = [10.]
        ros = Object(Time=Object(now=lambda: Object(to_sec=lambda: now[0])),
                     logwarn_throttle=lambda *args: None, loginfo=lambda *args: None)
        subject = node_methods('jiantu.py', ['gps_callback', 'check_initialization', 'is_gps_valid'],
            dict(rospy=ros, math=math, gps_quality_valid=gps_quality_valid, fresh=fresh))
        subject.state_lock = threading.Lock()
        subject.sensor_timeout = .5
        subject.alignment_tolerance = .15
        subject.last_gps_stamp = None
        subject.gps_lat_min, subject.gps_lat_max = 30., 40.
        subject.gps_lon_min, subject.gps_lon_max = 110., 125.
        subject.gps_max_horizontal_variance = .25
        subject.gps_allow_unknown_covariance = False
        subject.initializer = StableOrigin(duration=1., min_samples=5, max_gap=.3)
        subject.imu_history = PoseBuffer()
        subject.pose_history = PoseBuffer()
        subject.transformer = self.ns['CoordinateTransformer']()
        subject._is_initialized = False
        subject.gps_msg_count = 0
        published = []
        subject.publish_vehicle_state = lambda: published.append(True)
        def message(status=0):
            return Object(header=Object(stamp=Object(to_sec=lambda: now[0])),
                latitude=35., longitude=120., status=Object(status=status),
                position_covariance_type=2, position_covariance=[.1, 0., 0., 0., .1, 0., 0., 0., .1])
        for i in range(6):
            now[0] = 10. + i * .2
            subject.imu_history.add(now[0], 0., 0., 0.)
            subject.last_imu_stamp = now[0]
            subject.gps_callback(message())
        self.assertTrue(subject._is_initialized)
        self.assertEqual(len(published), 1)
        subject.gps_callback(message())
        self.assertEqual(len(published), 1)
        now[0] = 11.2
        subject.gps_callback(message(-1))
        self.assertEqual(len(published), 1)
        self.assertEqual(subject.last_gps_stamp, 11.)


if __name__ == '__main__':
    unittest.main()
