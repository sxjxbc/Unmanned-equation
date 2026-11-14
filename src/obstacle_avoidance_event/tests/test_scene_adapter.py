import ast
import math
import os
import sys
import unittest
ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0]=[os.path.join(ROOT,'src'),os.path.join(ROOT,'scripts')]
from obstacle_avoidance_event.legacy_input import LegacyCoordinates
from obstacle_avoidance_event.scene_adapter import SceneAssembler
from obstacle_avoidance_event.model import transform
from obstacle_avoidance_event.pipeline import Pipeline
from offline_demo import scene,VEHICLE


def inputs():
    s=scene('straight')
    aligned=dict(stamp=s.stamp,pose=s.pose,cones=s.cones,frame='map',pose_reference='rear_axle')
    free=dict(free=s.free,stamp=s.stamp,frame='map',version=1,source='synthetic')
    feedback=dict(speed=0.,steering=0.,stamp=s.stamp,source='synthetic')
    return aligned,free,feedback


class SceneAdapterTests(unittest.TestCase):
    def test_explicit_obstacles_are_preserved(self):
        a,f,b=inputs();a['obstacles']=[(3.,0.,.5)]
        s=SceneAssembler(synthetic=True).build(a,f,b,1.,1)
        self.assertEqual(s.obstacles,a['obstacles'])
        self.assertFalse(Pipeline(VEHICLE).step(s,1.)['valid'])

    def test_assembled_scene_runs_real_pipeline(self):
        a,f,b=inputs()
        s=SceneAssembler(synthetic=True).build(a,f,b,1.,1)
        self.assertTrue(Pipeline(VEHICLE).step(s,1.)['valid'])

    def test_synthetic_data_cannot_enter_hardware_mode(self):
        a,f,b=inputs()
        with self.assertRaises(ValueError):SceneAssembler().build(a,f,b,1.,1)
        f['source']='confirmed_boundary'
        with self.assertRaises(ValueError):SceneAssembler().build(a,f,b,1.,1)

    def test_explicit_hardware_contract(self):
        # A contract unit test with declared fake records, not vehicle evidence.
        a,f,b=inputs();f['source']='confirmed_boundary';b['source']='hardware'
        self.assertEqual(SceneAssembler().build(a,f,b,1.,1).feedback,(0.,0.,1.))

    def test_missing_or_mismatched_evidence(self):
        edits=[('a','pose_reference','imu'),('f','frame','other'),('f','version',2),
               ('f','stamp',.9),('b','stamp',.1),('b','source','command'),
               ('b','speed',-1.),('a','stamp',2.)]
        for group,key,value in edits:
            a,f,b=inputs();{'a':a,'f':f,'b':b}[group][key]=value
            with self.assertRaises(ValueError):SceneAssembler(synthetic=True).build(a,f,b,1.,1)
        a,f,b=inputs()
        for missing_free,missing_feedback in ((None,b),(f,None)):
            with self.assertRaises(ValueError):
                SceneAssembler(synthetic=True).build(a,missing_free,missing_feedback,1.,1)

    def test_rejected_scene_does_not_commit_and_duplicate_rejected(self):
        a,f,b=inputs();assembler=SceneAssembler(synthetic=True)
        with self.assertRaises(ValueError):assembler.build(a,None,b,1.,1)
        self.assertIsNone(assembler.last_stamp)
        assembler.build(a,f,b,1.,1)
        with self.assertRaises(ValueError):assembler.build(a,f,b,1.,1)
        a['stamp']=1.1;f['stamp']=1.1;b['stamp']=1.1;f['version']=2
        with self.assertRaises(ValueError):assembler.build(a,f,b,1.1,2)


class LegacyCoordinateTests(unittest.TestCase):
    def test_unknown_reference_never_assumed(self):
        with self.assertRaises(ValueError):LegacyCoordinates(1.,0.,-1,.5)
        with self.assertRaises(ValueError):LegacyCoordinates(1.,0.,-1,.5,convention_confirmed=True)

    def test_matches_original_methods_and_basis_conversion(self):
        filename=os.path.join(ROOT,'..','acceleration_event','src','jiantu.py')
        with open(filename,'rb') as stream:tree=ast.parse(stream.read(),filename=filename)
        cls=[n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='FSAESLAMNode'][0]
        cls.body=[n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name in
                  ('transform_lidar_to_vehicle','transform_vehicle_to_world')]
        tree.body=[cls]
        scope=dict(math=math)
        eval(compile(tree,filename,'exec'),scope)
        original=scope['FSAESLAMNode']()
        for ysign in (-1,1):
            for yawoffset in (0.,.5,-5.):
                converter=LegacyCoordinates(1.76618,-.11304,ysign,yawoffset,
                                           rear_in_imu=(-.7,.03),convention_confirmed=True)
                original.lidar_offset_x=1.76618;original.lidar_offset_y=-.11304
                original.lidar_y_sign=ysign;original.lidar_yaw_offset_deg=yawoffset
                for current,origin in ((0.,0.),(179.,-179.),(-179.,179.),(90.,0.)):
                    state=dict(current_x=3.,current_y=2.,current_yaw=current,origin_yaw=origin)
                    for x,y in ((1.,2.),(4.,-1.),(0.,0.)):
                        oldbody=original.transform_lidar_to_vehicle(x,y)
                        self.assertEqual(oldbody,converter.lidar_to_legacy_body(x,y))
                        oldmap=original.transform_vehicle_to_world(*oldbody,state=state)
                        newmap=transform(converter.lidar_to_rear(x,y),converter.pose_to_rear(state))
                        self.assertAlmostEqual(newmap[0],oldmap[0])
                        self.assertAlmostEqual(newmap[1],-oldmap[1])

    def test_zero_heading_rear_offset(self):
        converter=LegacyCoordinates(1.,0.,-1,0.,rear_in_imu=(-.5,.1),convention_confirmed=True)
        self.assertEqual(converter.pose_to_rear(dict(current_x=2.,current_y=3.,current_yaw=0.,origin_yaw=0.)),
                         (1.5,-2.9,0.))


if __name__=='__main__':unittest.main()
