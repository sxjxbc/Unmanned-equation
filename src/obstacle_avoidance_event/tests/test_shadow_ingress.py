"""Exercise real node callbacks with ROS message/API doubles, never ROS runtime."""
import ast
import json
import os
import threading
import unittest
from test_input_adapter import Object,message,AlignedIngress,ROOT


class Publisher(object):
    def __init__(self,*args,**kwargs):self.messages=[]
    def publish(self,msg):self.messages.append(msg)


class String(object):
    def __init__(self,data):self.data=data


class Path(object):
    def __init__(self):self.header=Object();self.poses=[]


class FakeRos(object):
    def __init__(self,configured):
        self.params={'~pose_reference':'rear_axle','~aligned_cones_frame':'rear_axle'} if configured else {}
        self.Time=Object(now=lambda:Object(to_sec=lambda:1.2))
    def get_param(self,name,default):return self.params.get(name,default)
    Publisher=Publisher
    def Subscriber(self,*args,**kwargs):pass
    def Timer(self,*args):pass
    def Duration(self,value):return value


def node(configured=True):
    filename=os.path.join(ROOT,'scripts','shadow_node.py')
    with open(filename,'rb') as stream:tree=ast.parse(stream.read(),filename=filename)
    tree.body=[n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='ShadowNode']
    scope=dict(rospy=FakeRos(configured),threading=threading,json=json,
               AlignedIngress=AlignedIngress,String=String,Path=Path,PoseStamped=Object,PoseArray=Object)
    eval(compile(tree,filename,'exec'),scope)
    return scope['ShadowNode']()


class ShadowIngressTests(unittest.TestCase):
    def test_callbacks_align_but_cannot_authorize_output(self):
        n=node()
        n.on_pose(message(1.));n.on_cones(message(frame='rear_axle'))
        n.tick(None)
        self.assertFalse(json.loads(n.status.messages[-1].data)['alignment_ready'])
        n.on_pose(message(1.2));n.tick(None)
        status=json.loads(n.status.messages[-1].data)
        self.assertTrue(status['alignment_ready'])
        self.assertNotIn('hardware_feedback',status['waiting_for'])
        self.assertEqual(status['control_mode'],'legacy_compatible')
        self.assertIsNone(status['actual_steering'])
        self.assertFalse(status['valid'])
        control=json.loads(n.control.messages[-1].data)
        self.assertFalse(control['valid']);self.assertFalse(control['vehicle_output'])
        self.assertEqual(n.path.messages[-1].poses,[])

    def test_default_reference_remains_unconfirmed(self):
        n=node(False)
        n.on_pose(message());n.on_cones(message(frame='rear_axle'));n.tick(None)
        status=json.loads(n.status.messages[-1].data)
        self.assertEqual(status['alignment_reason'],'rear_axle_reference_unconfirmed')

    def test_bad_pose_invalidates_buffer(self):
        n=node();n.on_pose(message(1.));n.on_pose(message(1.2))
        n.on_cones(message(frame='rear_axle'));n.tick(None)
        self.assertTrue(json.loads(n.status.messages[-1].data)['alignment_ready'])
        n.on_pose(message(1.3,frame='imu'));n.tick(None)
        self.assertFalse(json.loads(n.status.messages[-1].data)['alignment_ready'])
        self.assertEqual(n.ingress.poses.samples,[])


if __name__=='__main__':unittest.main()
