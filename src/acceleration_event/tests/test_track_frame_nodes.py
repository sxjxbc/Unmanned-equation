# -*- coding: utf-8 -*-
"""Exercise actual production planner/controller methods, no ROS or sockets."""
from __future__ import division
import math
import threading
import unittest
import json
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from legacy_run_guard import LegacyRunGuard
from track_frame import TrackFrame, bootstrap_frame, paired_centers, median, body_line
from test_legacy_run_guard import node_methods
from test_lidar_fusion_node import Obj, Scalar, Publisher, Stamp, load_node


class TrackGeometryTests(unittest.TestCase):
    def planner(self, rotation=0.0, offset=1.0, heading=0.0, rows=(5,10,15)):
        self.now = 100.0
        noop = lambda *args: None
        ros = Obj(get_param=lambda k,d: d, loginfo=noop, loginfo_throttle=noop,
                  logwarn_throttle=noop, logerr_throttle=noop,
                  Time=Obj(now=lambda:Stamp(self.now)))
        p = node_methods('zhixian_realtime.py',
                         ['select_local_cones','pair_cones','estimate_centerline',
                          'update_centerline','generate_local_path',
                          'planning_timer','_planning_timer'],
                         dict(math=math, time=Obj(time=lambda:self.now), rospy=ros,
                              TrackFrame=TrackFrame, bootstrap_frame=bootstrap_frame,
                              paired_centers=paired_centers, median=median,
                              json=json,String=Scalar))
        defaults = dict(track_width=3., track_forward_min=-8.,track_forward_max=25.,
                        max_local_cones=20, max_pair_count=2,pair_row_min_spacing=1.5,
                        pair_max_longitudinal_diff=.8,pair_distance_min=2.,pair_distance_max=4.,
                        centerline_timeout=.5,straight_slope_limit=.06,max_slope_change=.02,
                        slope_alpha=.12,centerline_alpha=.15,last_center_b=None,
                        centerline_valid=False,planning_started=False,centerline_k=0.,
                        centerline_b=0.,centerline_pairs=0,centerline_last_valid_time=None,
                        track_frame=None,last_fit_observation_time=None,yaw_track_world=None,
                        local_path_length=30.,path_point_spacing=.5,max_center_offset=1.5,
                        blend_distance=6.,path_smooth_window=3)
        p.__dict__.update(defaults)
        p.lock = threading.RLock()
        self.axes = TrackFrame(0,0,rotation)
        p.vehicle_x,p.vehicle_y = self.axes.world(0,offset)
        p.vehicle_yaw = rotation+heading
        p.cones = [self.axes.world(x,y)+(1.,) for x in rows for y in (-1.5,1.5)]
        p.last_cone_time = self.now
        return p

    def test_stationary_one_row_initializes_without_motion_or_multi_frame_gate(self):
        p = self.planner(rows=(5,))
        points, ready, updated = p.generate_local_path()
        self.assertTrue(ready and updated)
        self.assertEqual(1,p.centerline_pairs)
        self.assertAlmostEqual(points[-1][1],0.)

    def test_plus_minus_ten_degree_heading_does_not_change_world_path(self):
        paths=[]
        for yaw in (0,math.radians(10),math.radians(-10)):
            p=self.planner(heading=yaw)
            points,ready,updated=p.generate_local_path()
            self.assertTrue(ready and updated)
            self.assertAlmostEqual(p.centerline_k,0.,places=10)
            paths.append(points)
        for points in paths[1:]:
            for a,b in zip(paths[0],points):
                self.assertAlmostEqual(a[0],b[0],places=9)
                self.assertAlmostEqual(a[1],b[1],places=9)

    def test_current_heading_changes_body_line_not_fixed_centerline(self):
        p=self.planner(offset=0.)
        p.generate_local_path()
        k,b=p.centerline_k,p.centerline_b
        for yaw in (-.1745329252,.1745329252):
            p.vehicle_yaw=yaw
            self.now+=.1; p.last_cone_time=self.now
            p.generate_local_path()
            self.assertAlmostEqual(k,p.centerline_k)
            self.assertAlmostEqual(b,p.centerline_b)
            line=body_line(p.track_frame,k,b,(p.vehicle_x,p.vehicle_y,yaw))
            self.assertAlmostEqual(line[0],-math.tan(yaw))

    def test_lateral_movement_changes_return_offset_immediately_not_filtered_old_body_b(self):
        p=self.planner(offset=0.)
        p.generate_local_path()
        p.vehicle_y=1.
        points,ready,_=p.generate_local_path()
        self.assertTrue(ready)
        self.assertAlmostEqual(p.centerline_b,0.)
        self.assertAlmostEqual(points[0][1],1.)
        self.assertAlmostEqual(points[-1][1],0.)

    def test_continuous_rows_after_first_correction_remain_on_same_track(self):
        p=self.planner(offset=.6)
        for step in range(30):
            x=step*.7
            p.vehicle_x=x
            p.vehicle_y=.6*math.exp(-step/10.)
            p.vehicle_yaw=.17*math.sin(step/5.)
            p.cones=[(row,y,1.) for row in (x+4,x+9) for y in (-1.5,1.5)]
            self.now+=.1; p.last_cone_time=self.now
            points,ready,updated=p.generate_local_path()
            self.assertTrue(ready and updated)
            self.assertAlmostEqual(points[-1][1],0.,places=8)

    def test_rigid_rotation_equivariance(self):
        p=self.planner(heading=.1)
        baseline=p.generate_local_path()[0]
        for rotation in (.7,math.pi/2,2.7):
            p=self.planner(rotation=rotation,heading=.1)
            points=p.generate_local_path()[0]
            frame=TrackFrame(0,0,rotation)
            for original,point in zip(baseline,points):
                actual=frame.local(*point)
                self.assertAlmostEqual(original[0],actual[0],places=8)
                self.assertAlmostEqual(original[1],actual[1],places=8)

    def test_repeated_or_missing_observation_does_not_refresh_support_time(self):
        p=self.planner()
        p.generate_local_path()
        self.now+=.2
        _,ready,updated=p.generate_local_path()
        self.assertTrue(ready)
        self.assertFalse(updated)
        self.assertEqual(100.,p.centerline_last_valid_time)
        self.now+=1.0
        p.cones=[]
        p.generate_local_path()
        self.assertEqual(100.,p.centerline_last_valid_time)
        guard=LegacyRunGuard()
        for name in guard.limits: guard.observe(name,100.)
        guard.start(100.)
        for name in ('pose','imu','path'): guard.observe(name,self.now)
        guard.observe('road',self.now,self.now-p.centerline_last_valid_time)
        guard.evaluate(self.now)
        self.assertEqual('road_stale',guard.stop_reason)

    def test_true_fixed_direction_outlier_rejects_whole_observation(self):
        p=self.planner()
        p.generate_local_path()
        old=(p.centerline_k,p.centerline_b,p.centerline_last_valid_time)
        p.cones=[(x,y+.2*x,1.) for x in (5,10) for y in (-1.5,1.5)]
        self.now+=.1; p.last_cone_time=self.now
        self.assertFalse(p.generate_local_path()[2])
        self.assertEqual(old,(p.centerline_k,p.centerline_b,p.centerline_last_valid_time))

    def test_zero_or_one_cone_waits_then_valid_pair_recovers(self):
        p=self.planner()
        cones=p.cones
        for points in ([],cones[:1]):
            p.cones=points
            self.assertFalse(p.generate_local_path()[1])
        p.cones=cones
        self.assertTrue(p.generate_local_path()[1])

    def test_actual_controller_reprojects_each_heading_and_expires_fixed_road(self):
        module,clock,_=load_node()
        cls=module.LidarTrackingController.__bases__[0]
        cls.estimate_local_centerline.__globals__['time']=Obj(time=lambda:clock.now)
        o=cls.__new__(cls)
        o.fixed_road=(TrackFrame(0,0,0),0.,0.,2,10.)
        o.local_timeout=.35
        o.vehicle_x,o.vehicle_y=0.,.3
        o.local_centerline_active=False
        for yaw in (0.,.1745329252,-.1745329252):
            o.vehicle_yaw=yaw
            result=o.estimate_local_centerline()
            self.assertAlmostEqual(result['k'],-math.tan(yaw))
            self.assertAlmostEqual(result['b'],-.3/math.cos(yaw))
        clock.now=10.36
        self.assertIsNone(o.get_local_centerline_target(3.))
        self.assertFalse(o.local_centerline_active)

    def test_actual_planner_status_can_start_guard_and_drive_current_body_target(self):
        p=self.planner(offset=.3,heading=.17)
        p.pose_received=True
        p.road_status_pub=Publisher()
        captured=[]
        p.publish_path=captured.append
        p.planning_timer(None)
        self.assertEqual(1,len(captured))
        message=p.road_status_pub.messages[-1]
        module,clock,_=load_node()
        clock.now=self.now
        cls=module.LidarTrackingController.__bases__[0]
        cls.guard_road_callback.__globals__['time']=Obj(time=lambda:clock.now)
        o=cls.__new__(cls)
        o.state_lock=threading.RLock()
        o.run_guard=LegacyRunGuard()
        o.fixed_road=None
        o.guard_road_callback(message)
        for name in ('pose','imu','path'): o.run_guard.observe(name,self.now)
        self.assertTrue(o.run_guard.start(self.now))
        o.vehicle_x,o.vehicle_y,o.vehicle_yaw=0.,.3,.17
        o.local_timeout=.35
        target=o.get_local_centerline_target(3.)
        self.assertIsNotNone(target)
        self.assertAlmostEqual(target['y'],0.,places=8)

    def test_bad_startup_reference_can_retry_next_observation(self):
        p=self.planner()
        p.cones=[(x,y+.2*x,1.) for x in (5,10) for y in (-1.5,1.5)]
        self.assertFalse(p.generate_local_path()[1])
        self.assertIsNone(p.track_frame)
        p.cones=[(x,y,1.) for x in (5,10) for y in (-1.5,1.5)]
        self.now+=.1; p.last_cone_time=self.now
        self.assertTrue(p.generate_local_path()[1])


if __name__=='__main__':
    unittest.main()
