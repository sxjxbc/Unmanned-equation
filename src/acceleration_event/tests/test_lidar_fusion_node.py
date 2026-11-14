# -*- coding: utf-8 -*-
"""Exercise real callbacks and control loop with ROS transport doubles.

These tests do not start ROS or create sockets. They are not vehicle validation.
"""
from __future__ import division
import importlib
import math
import os
import sys
import types
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


class Obj(object):
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class Scalar(object):
    def __init__(self, data=None):
        self.data = data


class Stamp(object):
    def __init__(self, value):
        self.value = value

    def to_sec(self):
        return self.value


class Publisher(object):
    def __init__(self, *args, **kwargs):
        self.messages = []

    def publish(self, message):
        self.messages.append(message)


class SocketDouble(object):
    opened = 0

    def __init__(self):
        SocketDouble.opened += 1
        self.frames = []

    def init(self):
        return True

    def send_steering_frame(self, angle, stop=False):
        self.frames.append((angle, stop))
        return True

    def close(self):
        pass


def load_node():
    saved = dict(sys.modules)
    def module(name, **attrs):
        item = types.ModuleType(name)
        item.__dict__.update(attrs)
        sys.modules[name] = item
        return item
    noop = lambda *args, **kwargs: None
    clock = Obj(now=10., params={})
    module('rospy', init_node=noop, get_param=lambda k,d=None: clock.params.get(k,d),
           Time=Obj(now=lambda: Stamp(clock.now)), Duration=lambda v: v,
           Timer=lambda *a,**kw: Obj(shutdown=noop, reset=kw.get('reset',False)), Subscriber=lambda *a,**kw: None,
           Publisher=Publisher, loginfo=noop, logwarn=noop, logdebug_throttle=noop)
    module('numpy')  # Reused PD/CAN methods do not call NumPy.
    for package, names in [('nav_msgs', ['Path']),
                           ('geometry_msgs', ['PoseArray','PoseStamped']),
                           ('mapping', ['ConeArray']), ('sensor_msgs', ['Imu']),
                           ('std_msgs', ['Float64','String','Header'])]:
        pkg = module(package)
        msg = module(package+'.msg', **dict((n,Scalar) for n in names))
        pkg.msg = msg
    tf = module('tf')
    tf.transformations = module('tf.transformations')
    # Always reload the actual production code against this isolated transport.
    for name in ('genzong_realtime','genzong_lidar'):
        sys.modules.pop(name, None)
    node = importlib.import_module('genzong_lidar')
    # Save the actual codec before replacing only its transport factory.
    codec = node.CANUDPClient
    node.CANUDPClient = SocketDouble
    for name in list(sys.modules):
        if name not in saved:
            del sys.modules[name]
    sys.modules.update(saved)
    return node, clock, codec


class NodeTests(unittest.TestCase):
    def setUp(self):
        self.mod, self.clock, self.codec = load_node()
        SocketDouble.opened = 0

    def node(self, enabled=False, mode='shadow'):
        self.clock.params = {'~tracking_mode': mode, '~enable_vehicle_output': enabled,
                             '~fusion_log_enabled': False,
                             '~max_steering': .5, '~kp_lateral': .6, '~kp_heading': .2}
        return self.mod.LidarTrackingController()

    def sensors(self, node, t, offset=.4, x=0., y=0., empty=False, tick=True):
        self.clock.now = t
        pose = Obj(header=Obj(stamp=Stamp(t), frame_id='map'),
                   pose=Obj(position=Obj(x=x,y=y), orientation=Obj(x=0.,y=0.,z=0.,w=1.)))
        node.pose_callback(pose)
        node.imu_callback(Obj(header=Obj(stamp=Stamp(t)), orientation=Obj(z=0.)))
        points = []
        for px in (3., 7.):
            for py in (offset-1.5, offset+1.5):
                dx,dy=px-node.offset_x,py-node.offset_y
                c,s=math.cos(node.mount_yaw),math.sin(node.mount_yaw)
                points.append(Obj(position=Obj(x=c*dx+s*dy,y=(-s*dx+c*dy)/node.y_sign,z=0.)))
        node.lidar_callback(Obj(header=Obj(stamp=Stamp(t),frame_id='velodyne'),
                               poses=[] if empty else points))
        if tick:
            node.control_loop(None)

    def start(self, node):
        for t in (10.,10.1,10.2):
            self.sensors(node,t)
        self.assertTrue(node.started)

    def test_shadow_even_with_enable_true_has_no_socket(self):
        node = self.node(True, 'shadow')
        self.start(node)
        self.assertEqual(SocketDouble.opened,0)
        self.assertGreater(node.candidate_pub.messages[-1].data,0.)
        node.shutdown()

    def test_control_disabled_has_no_socket(self):
        node=self.node(False,'control')
        self.start(node)
        self.assertEqual(SocketDouble.opened,0)

    def test_pose_snapshot_updates_in_callback_not_one_second_timer(self):
        node=self.node()
        self.sensors(node,10., x=.1,y=.2)
        self.assertEqual((node.vehicle_x,node.vehicle_y),(.1,.2))

    def test_new_lidar_target_changes_steering_while_running(self):
        node=self.node()
        self.start(node)
        self.sensors(node,10.3,offset=.0)
        self.sensors(node,10.4,offset=-.4)
        self.assertLess(node.road.line[1],0.)
        import json
        status=json.loads(node.status_pub.messages[-1].data)
        self.assertLess(status['target'][1],0.)
        self.assertIsNone(node.stop_reason)

    def test_fault_stop_is_latched_and_repeated(self):
        node=self.node(True,'control')
        self.start(node)
        self.sensors(node,10.3,empty=True)
        self.sensors(node,10.4,empty=True)
        self.sensors(node,10.5,empty=True)
        self.sensors(node,10.6,empty=True)
        self.assertEqual(node.stop_reason,'road_stale')
        self.assertEqual(node.can_client.frames[-1],(0.,True))
        self.sensors(node,10.7)
        self.assertEqual(node.can_client.frames[-1],(0.,True))

    def test_finish_precedes_missing_cones_and_stays_latched(self):
        node=self.node(True,'control')
        node.progress.distance=.1
        self.start(node)
        self.sensors(node,10.3,x=.15,empty=True)
        self.assertEqual(node.stop_reason,'finish')
        self.clock.now=11.
        node.control_loop(None)
        self.assertEqual(node.stop_reason,'finish')
        self.assertEqual(node.can_client.frames[-1],(0.,True))

    def test_imu_gap_never_reuses_prior_coordinate_origin(self):
        node=self.node()
        self.start(node)
        self.sensors(node,10.7)
        self.assertEqual(node.stop_reason,'motion_gap')

    def test_clock_rewind_stops_not_reuses_future_observations(self):
        node=self.node()
        self.start(node)
        self.clock.now=9.
        node.control_loop(None)
        self.assertEqual(node.stop_reason,'clock_rewind')

    def test_wrong_frame_not_accepted(self):
        node=self.node()
        node.lidar_callback(Obj(header=Obj(stamp=Stamp(10.),frame_id='map'), poses=[]))
        self.assertIsNone(node.pending)

    def test_lateral_target_does_not_use_gps_map_offset(self):
        first=self.node()
        self.start(first)
        first_target=first.road.target(10.2, (0.,0.,0.), 3.)
        second=self.node()
        for t in (10.,10.1,10.2):
            self.sensors(second,t,y=20.)
        self.assertEqual(first_target,second.road.target(10.2, (0.,0.,0.), 3.))

    def test_unaligned_detection_waits_without_latest_pose_substitution(self):
        node=self.node()
        self.sensors(node,10.)
        node.pending=(10.05, [(3.,-1.5),(3.,1.5)])
        self.clock.now=10.06
        node.process_detection(10.06)
        self.assertIsNotNone(node.pending)
        self.assertEqual(node.road.stamp,10.)
        node.imu_callback(Obj(header=Obj(stamp=Stamp(10.06)),orientation=Obj(z=0.)))
        node.process_detection(10.06)
        self.assertIsNone(node.pending)
        self.assertEqual(node.road.stamp,10.05)

    def test_pose_loss_latches_stop_even_with_recent_road(self):
        node=self.node(True,'control')
        self.start(node)
        self.clock.now=10.8
        node.control_loop(None)
        self.assertEqual(node.stop_reason,'pose_stale')
        self.assertEqual(node.can_client.frames[-1],(0.,True))

    def test_pd_method_is_original_implementation(self):
        self.assertIs(self.mod.LidarTrackingController.pid_control,
                      self.mod.PathTrackingController.pid_control)
        self.assertIs(self.mod.LidarTrackingController.filter_tracking_errors,
                      self.mod.PathTrackingController.filter_tracking_errors)

    def test_driver_imu_before_newer_gps_timestamp_still_starts(self):
        node=self.node()
        for t in (10.,10.1,10.2,10.3,10.4):
            self.sensors(node,t,tick=False)
            # The real driver stamps GPS after IMU; mapping stamps later still.
            self.clock.now=t+.002
            node.pose_callback(Obj(header=Obj(stamp=Stamp(t+.002),frame_id='map'),
                pose=Obj(position=Obj(x=0.,y=0.),orientation=Obj(x=0.,y=0.,z=0.,w=1.))))
            node.control_loop(None)
        self.assertTrue(node.started)

    def test_ros_timer_survives_rewind_to_execute_stop_branch(self):
        node=self.node()
        self.assertTrue(node.control_timer.reset)

    def test_expired_reference_before_start_can_initialize_again(self):
        node=self.node()
        points=[(3.,-1.5),(3.,1.5)]
        for t in (10.,10.1,10.2):
            node.road.update(points,t,(0.,0.,0.))
        self.assertTrue(node.road.ready)
        node.motion.add(10.7,0.,0.)
        node.pending=(10.7,points)
        node.process_detection(10.7)
        self.assertFalse(node.road.ready)
        self.assertEqual(node.road.support,1)

    def test_send_failure_latches_stop(self):
        node=self.node(True,'control')
        self.start(node)
        node.can_client.send_steering_frame=lambda *a,**kw: False
        self.sensors(node,10.3)
        self.assertEqual(node.stop_reason,'can_send_failed')

    def test_control_cannot_send_after_shutdown(self):
        node=self.node(True,'control')
        self.start(node)
        node.shutdown()
        count=len(node.can_client.frames)
        self.clock.now=10.25
        node.control_loop(None)
        self.assertEqual(len(node.can_client.frames),count)

    def test_can_packet_remains_original_normal_and_stop(self):
        class Sink(object):
            def __init__(self): self.frames=[]
            def sendto(self,packet,address):
                self.frames.append(bytearray(packet))
                return len(packet)
        client=self.codec()
        client.sock=Sink()
        # Existing logdebug is only reached by the real codec, never socket I/O.
        self.mod.rospy.logdebug=lambda *a,**k: None
        self.mod.rospy.logerr=lambda *a,**k: None
        self.assertTrue(client.send_steering_frame(0.))
        self.assertTrue(client.send_steering_frame(0.,stop=True))
        normal,stop=client.sock.frames
        self.assertEqual(list(normal[:5]),[8,0,0,2,16])
        self.assertEqual(normal[5]+256*normal[6],500)
        self.assertEqual(stop[5]+256*stop[6],2000)


if __name__ == '__main__':
    unittest.main()
