# -*- coding: utf-8 -*-
"""Real constructor and stop/shutdown paths, using transport doubles only."""
import os
import sys
import unittest
import xml.etree.ElementTree as ET
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from test_lidar_fusion_node import load_node, Obj, Publisher, Scalar, SocketDouble
from test_legacy_run_guard import node_methods


class StartupTests(unittest.TestCase):
    def make(self, enabled=False, client_factory=None):
        mod, clock, codec = load_node()
        cls = mod.LidarTrackingController.__bases__[0]
        ns = cls.__init__.__globals__
        clock.params = {'~enable_vehicle_output':enabled}
        noop = lambda *args, **kwargs: None
        ns['ControlLogger'] = lambda path: Obj(close=noop,log_control_data=noop)
        ns['CANUDPClient'] = client_factory or SocketDouble
        ns['rospy'].logerr = noop
        ns['rospy'].logerr_throttle = noop
        ns['rospy'].loginfo_throttle = noop
        ns['rospy'].logwarn_throttle = noop
        ns['FSSIM_AVAILABLE'] = True
        ns['Cmd'] = Scalar
        ns['State'] = Scalar
        self.topics = []
        def publisher(topic,*args,**kwargs):
            self.topics.append(topic)
            return Publisher()
        ns['rospy'].Publisher = publisher
        SocketDouble.opened = 0
        return cls()

    def test_disabled_constructor_never_creates_socket_or_command_publishers(self):
        o=self.make(False)
        self.assertIsNone(o.can_client)
        self.assertEqual(SocketDouble.opened,0)
        self.assertNotIn('/control/steering',self.topics)
        self.assertNotIn('/fssim/cmd',self.topics)
        self.assertIn('/control/legacy_candidate_steering',self.topics)
        o.shutdown()
        self.assertEqual(SocketDouble.opened,0)

    def test_disabled_stop_and_shutdown_with_simulator_available_remain_output_free(self):
        o=self.make(False)
        o.control_mode='fssim'
        self.assertFalse(o.send_guard_stop())
        self.assertEqual(o.publish_fssim_control(.2),0.)
        o.shutdown()
        self.assertEqual(SocketDouble.opened,0)
        self.assertIsNone(o.cmd_pub)

    def test_enabled_configuration_retains_original_output_routes(self):
        o=self.make(True)
        self.assertEqual(SocketDouble.opened,1)
        self.assertIn('/control/steering',self.topics)
        self.assertIn('/fssim/cmd',self.topics)
        o.shutdown()

    def test_can_setup_failure_waits_and_retries_then_can_start_without_motion(self):
        class InitiallyUnavailable(SocketDouble):
            attempts = 0
            def init(self):
                self.attempts += 1
                return self.attempts >= 2
        o=self.make(True,InitiallyUnavailable)
        self.assertFalse(o.can_client_ok)
        o.pose_received=o.path_received=True
        o.current_path=Obj(poses=[Obj(),Obj()])
        now=o.check_start_condition.__globals__['time'].time()
        for name in o.run_guard.limits:
            o.run_guard.observe(name,now)
        o.check_start_condition()
        self.assertFalse(o.is_started)
        self.assertIsNone(o.run_guard.stop_reason)
        o.last_can_retry_time=now-2
        o.refresh_output_readiness()
        self.assertTrue(o.can_client_ok)
        o.check_start_condition()
        self.assertTrue(o.is_started)
        self.assertEqual(2,o.can_client.attempts)
        self.assertEqual([],o.can_client.frames)
        o.shutdown()

    def test_can_retry_is_rate_limited_and_does_not_restart_started_run(self):
        class Unavailable(SocketDouble):
            attempts = 0
            def init(self):
                self.attempts += 1
                return False
        o=self.make(True,Unavailable)
        o.refresh_output_readiness()
        self.assertEqual(1,o.can_client.attempts)
        o.last_can_retry_time-=2
        o.refresh_output_readiness()
        self.assertEqual(2,o.can_client.attempts)
        o.last_can_retry_time-=2
        o.is_started=True
        o.refresh_output_readiness()
        self.assertEqual(2,o.can_client.attempts)
        o.shutdown()

    def test_status_reports_can_unavailable_instead_of_running(self):
        class Unavailable(SocketDouble):
            def init(self):
                return False
        o=self.make(True,Unavailable)
        o.control_loop(None)
        import json
        status=json.loads(o.guard_pub.messages[-1].data)
        self.assertEqual('waiting',status['state'])
        self.assertFalse(status['output_ready'])
        self.assertIn('can_unavailable',status['waiting_for'])
        o.shutdown()

    def test_disabled_headless_mapping_does_not_enter_visualization(self):
        calls=[]
        o=node_methods('jiantu.py',['run'],dict(rospy=Obj(
            loginfo=lambda *a:None,spin=lambda:calls.append('spin'))))
        o.enable_visualization=False
        o.run()
        self.assertEqual(['spin'],calls)

    def test_validation_include_arguments_are_supported_and_force_disabled(self):
        launch=os.path.join(os.path.dirname(__file__),'..','launch')
        root=ET.parse(os.path.join(launch,'acceleration_event.launch')).getroot()
        allowed=set(a.get('name') for a in root.findall('arg'))
        for filename in ('validation.launch','live_mapping_check.launch'):
            entry=ET.parse(os.path.join(launch,filename)).getroot()
            includes=[i for i in entry.findall('include') if 'acceleration_event.launch' in i.get('file')]
            self.assertEqual(1,len(includes))
            args=dict((a.get('name'),a.get('value')) for a in includes[0].findall('arg'))
            self.assertTrue(set(args)<=allowed)
            self.assertEqual('false',args['enable_vehicle_output'])
            self.assertEqual('false',args['enable_visualization'])


if __name__=='__main__':
    unittest.main()
