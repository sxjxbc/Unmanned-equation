"""Offline regressions: no ROS initialization, sockets or vehicle commands."""
import ast
import json
import math
import sys
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace

SRC = Path(__file__).resolve().parents[1] / 'src'
sys.path.insert(0, str(SRC))
from data_quality import PoseBuffer, fresh, centerline_from_status


def methods(filename, names, namespace):
    tree = ast.parse((SRC / filename).read_text(encoding='utf-8'))
    body = [n for c in tree.body if isinstance(c, ast.ClassDef)
            for n in c.body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert len(body) == len(names)
    module = ast.Module(body=[ast.ClassDef(name='Subject', bases=[], keywords=[],
                                         body=body, decorator_list=[])], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), filename, 'exec'), namespace)
    return namespace['Subject']()


class BufferTests(unittest.TestCase):
    def test_missing(self):
        self.assertIsNone(PoseBuffer().latest(10, .5, 100))

    def test_fresh_and_two_clocks(self):
        b = PoseBuffer(); b.add(10, 0, 0, 0, received=100)
        self.assertTrue(b.latest(10.1, .5, 100.1)['valid'])
        self.assertFalse(b.latest(10.6, .5, 100.1)['valid'])
        self.assertFalse(b.latest(10.1, .5, 100.6)['valid'])
        self.assertFalse(b.latest(9, .5, 100.1)['valid'])

    def test_duplicate_and_reverse(self):
        b = PoseBuffer(); b.add(10, 0, 0, 0)
        self.assertFalse(b.add(10, 0, 0, 0))
        self.assertFalse(b.add(9, 0, 0, 0))
        self.assertEqual(len(b.samples), 1)

    def test_invalid(self):
        b = PoseBuffer()
        self.assertFalse(b.add(0, 0, 0, 0))
        self.assertFalse(b.add(10, float('nan'), 0, 0))
        self.assertFalse(b.add(10, 0, 0, 0, speed=float('inf')))

    def test_jumps_and_recovery(self):
        b = PoseBuffer(); b.add(10, 0, 0, 0, received=100)
        self.assertFalse(b.add(10.1, 20, 0, 0))
        self.assertFalse(b.latest(10.1, .5, 100.1)['valid'])
        self.assertFalse(b.add(10.1, 0, 0, 2))
        self.assertTrue(b.add(10.2, .2, 0, .1, received=100.2))
        self.assertTrue(b.latest(10.2, .5, 100.2)['valid'])

    def test_sample_time_speed_not_arrival_time(self):
        b = PoseBuffer(); b.add(10, 0, 0, 0, received=100)
        b.add(10.1, 1, 0, 0, received=100.001)
        self.assertAlmostEqual(b.samples[-1]['speed'], 3)

    def test_speed_not_clipped(self):
        b = PoseBuffer(); b.add(10, 0, 0, 0, speed=8)
        self.assertEqual(b.samples[-1]['speed'], 8)

    def test_interpolation_and_wrap(self):
        b = PoseBuffer(); b.add(10, 0, 0, math.radians(179))
        b.add(10.2, 2, 0, math.radians(-179))
        p = b.at(10.1, .15)
        self.assertAlmostEqual(p['x'], 1)
        self.assertAlmostEqual(abs(p['yaw']), math.pi)
        self.assertIsNone(b.at(11, .15))
        self.assertIsNone(b.at(10.1, .05))

    def test_bad_freshness(self):
        for stamp in (None, 'bad', float('nan'), float('inf'), 0, 11):
            self.assertFalse(fresh(stamp, 10, .5))


class IntegrationTests(unittest.TestCase):
    def controller(self, names):
        self.now = 10.0
        ros = SimpleNamespace(Time=SimpleNamespace(now=lambda: SimpleNamespace(to_sec=lambda:self.now)),
                              logwarn_throttle=lambda *a:None)
        return methods('genzong_realtime.py', names, dict(rospy=ros, fresh=fresh,
                       centerline_from_status=centerline_from_status,
                       monotonic=lambda:100, json=json, String=lambda **kw:kw))

    def test_each_cycle_gets_latest_pose(self):
        o = self.controller(['check_position_sources'])
        o.localization_mode='slam'; o.pose_timeout=.5; o.slam_buffer=PoseBuffer()
        o.check_start_condition=lambda:None
        o.slam_buffer.add(10, 0, 0, 0)
        o.check_position_sources(); self.assertEqual(o.vehicle_x, 0)
        o.slam_buffer.add(10.05, .2, 0, .01)
        self.now=10.05
        o.check_position_sources(); self.assertEqual(o.vehicle_x, .2)
        self.now=11
        o.check_position_sources(); self.assertFalse(o.pose_received)

    def test_path_matches_original_observation(self):
        o=self.controller(['path_is_fresh'])
        o.path_received=True
        o.current_path=SimpleNamespace(header=SimpleNamespace(stamp=SimpleNamespace(to_sec=lambda:10)))
        o.path_receive_time=100; o.status_receive_time=100
        o.path_timeout=.5; o.path_observation_timeout=.5
        o.planning_status=dict(valid=True,path_stamp=10,observation_stamp=9.9)
        o.planning_status['pairs'] = 2
        o.planning_status['centerline'] = dict(map_k=0, map_b=0, pairs=2,
            source='two_pairs', confidence=1, observation_stamp=9.9)
        self.assertTrue(o.path_is_fresh())
        o.planning_status['observation_stamp']=9
        self.assertFalse(o.path_is_fresh())
        o.planning_status['observation_stamp']=9.9
        o.planning_status['path_stamp']=9.8
        self.assertFalse(o.path_is_fresh())

    def test_latched_stop_precedes_all_tracking_checks(self):
        o=self.controller(['_control_loop'])
        calls=[]
        o.check_position_sources=lambda:calls.append('pose')
        o.finish_triggered=True
        o.hold_normal_stop=lambda:calls.append('stop')
        o.publish_input_status=lambda reason:calls.append(reason)
        o._control_loop(None)
        self.assertEqual(calls,['pose','stop','normal_stop_latched'])

    def test_invalid_pose_inhibits_tracking(self):
        o=self.controller(['_control_loop'])
        calls=[]
        o.check_position_sources=lambda:None
        o.finish_triggered=False; o.pose_received=False; o.input_reason='stale'
        o.update_finish_distance=lambda:None
        o.publish_input_status=calls.append
        o._control_loop(None)
        self.assertEqual(calls,['stale'])

    def test_stop_outputs_mocked_can_and_zero_sim_drive(self):
        o=self.controller(['hold_normal_stop'])
        calls=[]
        o.enable_vehicle_output=True
        o.can_client_ok=True
        o.can_client=SimpleNamespace(send_steering_frame=lambda angle,stop:calls.append((angle,stop)))
        o.publish_steering_command=lambda angle:calls.append(angle)
        o.control_mode='fssim'
        o.localization_mode='fssim'
        o.cmd_pub=SimpleNamespace(publish=lambda cmd:calls.append((cmd.dc,cmd.delta)))
        o.hold_normal_stop.__func__.__globals__['Cmd']=SimpleNamespace
        o.hold_normal_stop()
        self.assertEqual(calls,[(0.0,True),0.0,(0.0,0.0)])

    def test_expired_observation_no_new_path(self):
        o=methods('zhixian_realtime.py',['generate_local_path'],dict(fresh=fresh,
                  rospy=SimpleNamespace(Time=SimpleNamespace(now=lambda:SimpleNamespace(to_sec=lambda:10)))))
        o.update_centerline=lambda:False; o.planning_started=True
        o.last_valid_observation_stamp=9; o.observation_hold_timeout=.5
        self.assertEqual(o.generate_local_path(),(None,False,False))

    def test_validation_suppresses_ros_steering(self):
        o=self.controller(['publish_steering_command'])
        o.publish_steering_command.__func__.__globals__['Float64']=SimpleNamespace
        o.enable_vehicle_output=False
        o.steering_pub=SimpleNamespace(publish=lambda msg:self.fail('steering escaped'))
        o.publish_steering_command(.2)

    def test_validation_suppresses_sim_drive(self):
        o=self.controller(['publish_fssim_control'])
        o.enable_vehicle_output=False
        self.assertEqual(o.publish_fssim_control(.2),0.)

    def test_validation_suppresses_stop_sim_output(self):
        o=self.controller(['hold_normal_stop'])
        o.enable_vehicle_output=False; o.can_client_ok=False
        o.publish_steering_command=lambda angle:None
        o.control_mode='fssim'
        o.localization_mode='fssim'
        o.cmd_pub=SimpleNamespace(publish=lambda msg:self.fail('stop output escaped'))
        o.hold_normal_stop()

    def test_empty_lidar_frame_publishes_once_and_clears(self):
        published=[]
        o=methods('jiantu.py',['cone_callback'],dict(fresh=fresh,math=math,
                  rospy=SimpleNamespace(Time=SimpleNamespace(now=lambda:SimpleNamespace(to_sec=lambda:10)),
                                        logwarn_throttle=lambda *a:None)))
        o.cone_detection_count=0; o.sensor_timeout=.5; o.alignment_tolerance=.15
        o.get_state_snapshot=lambda:dict(is_initialized=True,origin_yaw=0)
        o.pose_history=PoseBuffer(); o.pose_history.add(10,0,0,0)
        o.cone_lock=threading.RLock(); o.cone_map=[object()]; o.cone_timeout=3
        o.publish_cone_map=published.append
        msg=SimpleNamespace(header=SimpleNamespace(stamp=SimpleNamespace(to_sec=lambda:10)),poses=[])
        o.cone_callback(msg)
        self.assertEqual(o.cone_map,[]); self.assertEqual(published,[10])


if __name__ == '__main__':
    unittest.main()
