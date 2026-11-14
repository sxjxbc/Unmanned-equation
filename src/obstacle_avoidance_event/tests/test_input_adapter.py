import os
import sys
import math
import unittest
ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0,os.path.join(ROOT,'src'))
from obstacle_avoidance_event.input_adapter import PoseBuffer,AlignedIngress,planar_pose


class Object(object):
    def __init__(self,**kwargs):self.__dict__.update(kwargs)


def message(stamp=1.1,frame='map',yaw=0.):
    position=Object(x=0.,y=0.,z=0.)
    orientation=Object(x=0.,y=0.,z=math.sin(yaw/2),w=math.cos(yaw/2))
    return Object(header=Object(frame_id=frame,stamp=Object(to_sec=lambda:stamp)),
                  pose=Object(position=position,orientation=orientation),
                  poses=[Object(position=Object(x=1.,y=0.))])


class InputTests(unittest.TestCase):
    def test_interpolate_position_and_wrapped_heading(self):
        buffer=PoseBuffer(.3)
        buffer.add(1.,(0.,0.,math.radians(179)))
        buffer.add(1.2,(2.,4.,math.radians(-179)))
        p=buffer.at(1.1)
        self.assertAlmostEqual(p[0],1.)
        self.assertAlmostEqual(p[1],2.)
        self.assertAlmostEqual(abs(p[2]),math.pi)

    def test_no_extrapolation_or_large_gap(self):
        buffer=PoseBuffer(.1)
        buffer.add(1.,(0,0,0));buffer.add(2.,(1,0,0))
        for stamp in (.9,1.5,2.1):
            with self.assertRaises(ValueError):buffer.at(stamp)
        self.assertEqual(buffer.at(2.),(1,0,0))

    def test_bounded_monotonic_samples(self):
        buffer=PoseBuffer(capacity=2)
        for t in (1.,2.,3.):buffer.add(t,(0,0,0))
        self.assertEqual(len(buffer.samples),2)
        for t in (3.,2.,float('nan')):
            with self.assertRaises(ValueError):buffer.add(t,(0,0,0))

    def test_quaternion_contract(self):
        msg=message(yaw=.5)
        self.assertAlmostEqual(planar_pose(msg)[2],.5)
        msg.pose.orientation.z=90
        with self.assertRaises(ValueError):planar_pose(msg)
        msg=message();msg.pose.orientation.x=.1
        with self.assertRaises(ValueError):planar_pose(msg)

    def test_alignment_transform_without_planning_permission(self):
        ingress=AlignedIngress(cones_frame='rear_axle',pose_reference='rear_axle',max_gap=.3)
        a,b=message(1.),message(1.2)
        b.pose.position.x=2.
        ingress.add_pose(a);ingress.add_pose(b)
        out=ingress.align(message(frame='rear_axle'),1.2)
        self.assertAlmostEqual(out['cones'][0][0],2.)
        self.assertFalse(out['valid_for_planning'])
        self.assertIn('hardware_feedback',out['missing'])

    def test_unknown_reference_frames_and_age_reject(self):
        with self.assertRaises(ValueError):AlignedIngress().align(message(),1.2)
        ingress=AlignedIngress(cones_frame='rear_axle',pose_reference='rear_axle')
        with self.assertRaises(ValueError):ingress.add_pose(message(frame='imu'))
        for msg,now in ((message(frame='lidar'),1.2),(message(frame='rear_axle'),2.),
                        (message(frame='rear_axle'),1.)):
            with self.assertRaises(ValueError):ingress.align(msg,now)

    def test_missing_bracket_then_future_pose_arrives(self):
        ingress=AlignedIngress(cones_frame='rear_axle',pose_reference='rear_axle',max_gap=.3)
        ingress.add_pose(message(1.))
        with self.assertRaises(ValueError):ingress.align(message(frame='rear_axle'),1.15)
        ingress.add_pose(message(1.2))
        self.assertEqual(len(ingress.align(message(frame='rear_axle'),1.2)['cones']),1)

    def test_invalid_cones(self):
        ingress=AlignedIngress(cones_frame='rear_axle',pose_reference='rear_axle')
        ingress.add_pose(message())
        msg=message(frame='rear_axle');msg.poses[0].position.x=float('nan')
        with self.assertRaises(ValueError):ingress.align(msg,1.2)


if __name__=='__main__':unittest.main()
