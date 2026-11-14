import os,sys,unittest,json,io
ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0]=[os.path.join(ROOT,'src'),os.path.join(ROOT,'scripts')]
from obstacle_avoidance_event.geometry import imu_body_vehicle,body_disks,lidar_point_in_imu,imu_pose_to_rear_axle
from obstacle_avoidance_event.pipeline import Pipeline
from offline_demo import scene

class ImuProfileTests(unittest.TestCase):
    def setUp(self):
        with io.open(os.path.join(ROOT,'config/design_dimensions.json'),encoding='utf-8') as f:self.p=json.load(f)
    def test_length(self):
        v=imu_body_vehicle(self.p,.05)
        self.assertAlmostEqual(v['front']+v['rear'],2.93)
        self.assertAlmostEqual(v['rear'],1.16382)
    def test_footprint_uses_imu_origin(self):
        import math
        ds=list(body_disks((0,0),0,imu_body_vehicle(self.p,0)))
        for x in (-1.16382,1.76618):
            self.assertTrue(any(math.hypot(x-c[0],1.4-c[1])<=r for c,r in ds))
    def test_translation_direction(self):
        identity=((1,0,0),(0,1,0),(0,0,1))
        self.assertEqual(lidar_point_in_imu((0,0,0),self.p,identity),(1.76618,-.11304,.42850))
    def test_unknown_rotation_rejected(self):
        with self.assertRaises(ValueError):lidar_point_in_imu((0,0,0),self.p,None)
    def test_unknown_axle_not_fabricated(self):
        unknown=dict(self.p,imu_to_rear_axle_m=None)
        with self.assertRaises(ValueError):imu_pose_to_rear_axle((0,0,0),unknown)
    def test_user_rear_offset_direction(self):
        rear=imu_pose_to_rear_axle((3.,4.,0.),self.p)
        self.assertAlmostEqual(rear[0],2.537)
        self.assertAlmostEqual(rear[1],4.010)
        self.assertTrue(self.p['imu_to_rear_axle_measurement']['lateral_approximate'])
    def test_rear_offset_rotates_with_heading(self):
        import math
        rear=imu_pose_to_rear_axle((3.,4.,math.pi/2.),self.p)
        self.assertAlmostEqual(rear[0],2.990)
        self.assertAlmostEqual(rear[1],3.537)
        rear=imu_pose_to_rear_axle((3.,4.,math.pi),self.p)
        self.assertAlmostEqual(rear[0],3.463)
        self.assertAlmostEqual(rear[1],3.990)
    def test_controller_reference_guard(self):
        r=Pipeline(imu_body_vehicle(self.p,.05)).step(scene('straight'),1,True)
        self.assertEqual(r['reason'],'imu_profile_requires_rear_axle_adapter')
if __name__=='__main__':unittest.main()
