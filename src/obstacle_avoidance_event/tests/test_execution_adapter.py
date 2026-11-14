import os
import sys
import ast
import math
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'src'))
from obstacle_avoidance_event.execution_adapter import LocalExecutionAdapter


def candidate(angle=.1, speed=1.):
    return dict(candidate_valid=True, steering=angle, speed_limit=speed)


class ExecutionTests(unittest.TestCase):
    def test_explicit_sign_and_recording(self):
        adapter = LocalExecutionAdapter(-1, .5)
        result = adapter.step(candidate(), 1., 1.)
        self.assertEqual(result['steering'], -.1)
        self.assertFalse(result['vehicle_output'])
        self.assertEqual(adapter.client.requests, [dict(steering_angle=-.1, stop=False)])

    def test_wait_then_permission_loss_latches(self):
        adapter = LocalExecutionAdapter(1, .5)
        self.assertEqual(adapter.step(dict(candidate_valid=False), 0., 1.)['action'], 'wait')
        self.assertEqual(adapter.client.requests, [])
        adapter.step(candidate(), 2., 2.)
        self.assertEqual(adapter.step(dict(candidate_valid=False), 3., 3.)['action'], 'stop')
        self.assertEqual(adapter.step(candidate(), 4., 4.)['action'], 'stop')

    def test_emergency_and_task_stop(self):
        for key in ('emergency', 'stop_requested'):
            adapter = LocalExecutionAdapter(1, .5)
            result = adapter.step(candidate(), 1., 1., **{key: True})
            self.assertEqual(result['action'], 'stop')
            self.assertEqual(adapter.client.requests[-1], dict(steering_angle=0., stop=True))

    def test_invalid_or_stale_requests_never_steer(self):
        for request, stamp in ((candidate(float('nan')), 1.), (candidate(.6), 1.),
                               (candidate(speed=-1), 1.), (candidate(), .1),
                               (candidate(), 2.), (candidate(speed=0), 1.)):
            adapter = LocalExecutionAdapter(1, .5)
            self.assertEqual(adapter.step(request, stamp, 1.)['action'], 'stop')
            self.assertTrue(all(r['stop'] for r in adapter.client.requests))

    def test_clock_and_fault(self):
        adapter = LocalExecutionAdapter(1, .5)
        adapter.step(candidate(), 2., 2.)
        self.assertEqual(adapter.step(candidate(), 1., 1.)['action'], 'stop')
        adapter = LocalExecutionAdapter(1, .5)
        self.assertEqual(adapter.step(dict(fault_latched=True, reason='collision'), 1., 1.)['reason'], 'collision')

    def test_output_cannot_be_enabled(self):
        with self.assertRaises(ValueError):
            LocalExecutionAdapter(1, .5, vehicle_output=True)

    def test_original_packet_implementation_without_ros_or_network(self):
        # Compile only the existing class; never import the ROS controller,
        # never call init(), and give it an in-memory socket replacement.
        legacy = os.path.abspath(os.path.join(ROOT, '..', 'acceleration_event', 'src', 'genzong_realtime.py'))
        with open(legacy, 'rb') as stream:
            tree = ast.parse(stream.read(), filename=legacy)
        tree.body = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'CANUDPClient']
        self.assertEqual(len(tree.body), 1)
        class Logger(object):
            def __getattr__(self, name):
                return lambda *args: None
        class MemorySocket(object):
            def __init__(self): self.packets = []
            def sendto(self, packet, address):
                self.packets.append(bytearray(packet))
                return len(packet)
        scope = dict(math=math, rospy=Logger(), CAN_ID_STEERING=0x210)
        eval(compile(tree, legacy, 'exec'), scope)
        client = scope['CANUDPClient']()
        client.sock = MemorySocket()
        adapter = LocalExecutionAdapter(1, .5)
        adapter.step(candidate(.1), 1., 1.)
        adapter.step(candidate(), 2., 2., stop_requested=True)
        for request in adapter.client.requests:
            self.assertTrue(client.send_steering_frame(**request))
        for packet, raw in zip(client.sock.packets, (int(round(500+.1*1000/math.pi)), 2000)):
            self.assertEqual(list(packet[:5]), [8, 0, 0, 2, 16])
            self.assertEqual(len(packet), 13)
            self.assertEqual(packet[5]+256*packet[6], raw)
            self.assertEqual(list(packet[7:]), [0]*6)


if __name__ == '__main__':
    unittest.main()
