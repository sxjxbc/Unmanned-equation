# -*- coding: utf-8 -*-
from __future__ import division
import math
import os
import sys
import unittest
import xml.etree.ElementTree as ET

SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, SRC)
from lidar_road import RoadTracker, MotionHistory, Progress, transform_point, transform_line


def cones(theta=0.0, offset=0.0, rows=(3., 7.)):
    c, s = math.cos(theta), math.sin(theta)
    return [(c*x-s*y, s*x+c*y) for x in rows for y in (offset-1.5, offset+1.5)]


class GeometryTests(unittest.TestCase):
    def ready(self, points):
        road = RoadTracker()
        for stamp in (10., 10.1, 10.2):
            self.assertTrue(road.update(points, stamp, (0., 0., 0.)))
        self.assertTrue(road.ready)
        return road

    def test_multiframe_initialization_not_duplicate_timer(self):
        road = RoadTracker()
        road.update(cones(), 10., (0., 0., 0.))
        for unused in range(10):
            self.assertFalse(road.update(cones(), 10., (0., 0., 0.)))
        self.assertFalse(road.ready)

    def test_left_and_right_offset_produce_correct_target(self):
        for offset in (-0.5, 0.5):
            road = self.ready(cones(offset=offset))
            self.assertAlmostEqual(road.target(10.2, (0.,0.,0.), 3.)[1], offset)

    def test_true_yaw_error_above_old_three_degree_limit(self):
        road = self.ready(cones(theta=math.radians(10)))
        self.assertAlmostEqual(road.line[0], math.radians(10))
        self.assertGreater(road.target(10.2, (0.,0.,0.), 3.)[1], 0.4)

    def test_continuous_lateral_update_after_initialization(self):
        road = self.ready(cones())
        road.update(cones(offset=0.4), 10.3, (0.,0.,0.))
        self.assertAlmostEqual(road.target(10.3, (0.,0.,0.), 3.)[1], 0.4)

    def test_line_motion_compensation_correct_rotation_and_translation(self):
        theta, offset = transform_line((0., 0.), (0.,0.,0.), (1.,.4,.2))
        self.assertAlmostEqual(theta, -.2)
        self.assertAlmostEqual(offset, -.4)
        road = self.ready(cones())
        target = road.target(10.25, (1.,.4,.2), 3.)
        self.assertLess(target[1], -.9)

    def test_one_row_keeps_imu_compensated_direction(self):
        road = self.ready(cones())
        pose = (0.,0.,.15)
        points = [transform_point(p, (0.,0.,0.), pose) for p in cones(rows=(4.,))]
        self.assertTrue(road.update(points, 10.3, pose))
        self.assertAlmostEqual(road.line[0], -.15)
        self.assertEqual(road.kind, 'one_row')

    def test_empty_and_bad_frames_do_not_refresh_observation(self):
        road = self.ready(cones())
        self.assertFalse(road.update([], 10.3, (0.,0.,0.)))
        self.assertFalse(road.update([(float('nan'),0.)], 10.4, (0.,0.,0.)))
        self.assertEqual(road.stamp, 10.2)
        self.assertIsNone(road.target(10.6, (0.,0.,0.), 3.))

    def test_single_side_bounded_by_last_two_sided_frame(self):
        road = self.ready(cones())
        self.assertTrue(road.update([(3.,1.6),(7.,1.6)], 10.3, (0.,0.,0.)))
        self.assertEqual(road.kind, 'single_side')
        self.assertAlmostEqual(road.target(10.3, (0.,0.,0.), 3.)[1], .1)
        self.assertIsNone(road.target(10.6, (0.,0.,0.), 3.))

    def test_outlier_pair_does_not_move_road(self):
        road = self.ready(cones())
        self.assertFalse(road.update(cones(offset=2.), 10.3, (0.,0.,0.)))
        self.assertAlmostEqual(road.line[1], 0.)

    def test_startup_empty_frame_breaks_confirmation(self):
        road = RoadTracker()
        road.update(cones(), 10., (0.,0.,0.))
        road.update([], 10.1, (0.,0.,0.))
        road.update(cones(), 10.2, (0.,0.,0.))
        self.assertEqual(road.support, 1)


class MotionAndFinishTests(unittest.TestCase):
    def test_interpolation_and_wrap(self):
        history = MotionHistory()
        history.add(10., math.radians(179), 1.)
        history.add(10.2, math.radians(-179), 1.)
        pose = history.at(10.1)
        self.assertAlmostEqual(abs(pose[2]), math.pi)
        self.assertAlmostEqual(pose[0], -.1)
        self.assertIsNone(history.at(10.3))
        self.assertFalse(history.add(10.1, 0., 1.))

    def test_motion_gap_clears_old_coordinate_origin(self):
        history = MotionHistory()
        history.add(10., 0., 1.)
        history.add(11., 0., 1.)
        self.assertIsNone(history.at(10.))
        self.assertEqual(history.at(11.), (0.,0.,0.))

    def test_progress_projection_reverse_and_latch(self):
        progress = Progress(2., 5.)
        self.assertTrue(progress.update(10.,0.,0.,10.,0.))
        progress.update(10.1,0.,.3,10.1)
        self.assertEqual(progress.value, 0.)
        progress.update(10.2,-.1,.3,10.2)
        self.assertAlmostEqual(progress.value, -.1)
        for i in range(1,7):
            progress.update(10.2+.1*i, -.1+.4*i, .3, 10.2+.1*i)
        self.assertTrue(progress.finished)
        progress.update(10.9,2.,.3,10.9)
        self.assertTrue(progress.finished)

    def test_progress_rejects_stale_duplicate_and_jump(self):
        progress = Progress(75., 5.)
        progress.update(10.,0.,0.,10.,0.)
        self.assertFalse(progress.update(10.,0.,0.,10.))
        self.assertFalse(progress.update(10.1,50.,0.,10.1))
        self.assertFalse(progress.update(10.2,0.,0.,11.))
        self.assertEqual(progress.value, 0.)


class ConfigurationTests(unittest.TestCase):
    def test_original_effective_controller_and_extrinsics_preserved(self):
        launch = os.path.join(SRC, '..', 'launch')
        old = ET.parse(os.path.join(launch, 'acceleration_event.launch')).getroot()
        new = ET.parse(os.path.join(launch, 'acceleration_lidar.launch')).getroot()
        def params(root, script):
            node = next(n for n in root.findall('node') if n.get('type') == script)
            return dict((p.get('name'),p.get('value')) for p in node.findall('param'))
        baseline = params(old,'genzong_realtime.py')
        fusion = params(new,'genzong_lidar.py')
        for key, value in baseline.items():
            # Legacy guard settings do not apply to the independent fusion gate.
            if key in ('guard_pose_timeout', 'guard_path_timeout', 'guard_road_timeout', 'enable_vehicle_output'):
                continue
            self.assertEqual(value, fusion[key], key)
        for key, value in params(old, 'jiantu.py').items():
            if key.startswith('lidar_'):
                self.assertEqual(value, fusion[key], key)
        args = dict((a.get('name'),a.get('default')) for a in new.findall('arg'))
        self.assertEqual(args['enable_vehicle_output'], 'false')
        self.assertEqual(args['tracking_mode'], 'shadow')


if __name__ == '__main__':
    unittest.main()
