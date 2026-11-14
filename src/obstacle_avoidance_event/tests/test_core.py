import os,sys,math,unittest
ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0]=[os.path.join(ROOT,'src'),os.path.join(ROOT,'scripts')]
from offline_demo import scene,VEHICLE
from obstacle_avoidance_event.pipeline import Pipeline
from obstacle_avoidance_event.model import LocalMap,transform
from obstacle_avoidance_event.gates import triangulate
from obstacle_avoidance_event.trajectory import spline,curvature

class CoreTests(unittest.TestCase):
    def test_straight(self):
        r=Pipeline(VEHICLE).step(scene('straight'),1,True)
        self.assertTrue(r['valid']);self.assertGreater(len(r['path']),50)
        self.assertFalse(r['vehicle_output'])
    def test_s_curve(self):
        r=Pipeline(VEHICLE).step(scene('s_curve'),1,True)
        self.assertTrue(r['valid'])
        self.assertGreater(max(r['curvature']),.01)
        self.assertLess(min(r['curvature']),-.01)
    def test_ready_requires_authorization(self):
        self.assertEqual(Pipeline(VEHICLE).step(scene('straight'),1)['state'],'READY')
    def test_missing_vehicle(self):
        self.assertEqual(Pipeline().step(scene('straight'),1)['reason'],'vehicle_parameters_unavailable')
    def test_missing_free(self):
        s=scene('straight');s.free=None
        self.assertEqual(Pipeline(VEHICLE).step(s,1)['reason'],'free_space_unavailable')
    def test_missing_feedback(self):
        s=scene('straight');s.feedback=None
        self.assertEqual(Pipeline(VEHICLE).step(s,1)['reason'],'feedback_unavailable')
    def test_stale_latches(self):
        p=Pipeline(VEHICLE);p.step(scene('straight'),1,True)
        self.assertEqual(p.step(scene('straight'),2,True)['state'],'STOP_LATCHED')
        self.assertEqual(p.step(scene('straight'),1,True)['reason'],'stop_latched')
    def test_obstacle(self):
        s=scene('straight');s.obstacles=[(4,0,1)]
        self.assertEqual(Pipeline(VEHICLE).step(s,1)['reason'],'collision')
    def test_unknown_space(self):
        s=scene('straight');s.free=lambda p,r:False
        self.assertEqual(Pipeline(VEHICLE).step(s,1)['reason'],'unknown_or_boundary')
    def test_degenerate(self):
        with self.assertRaises(ValueError):triangulate([(0,0),(1,0),(2,0)])
    def test_duplicates(self):
        with self.assertRaises(ValueError):triangulate([(0,0),(1,0),(1,0)])
    def test_pose_alignment(self):
        with self.assertRaises(ValueError):LocalMap().update([(1,0)],(0,0,0),1,1.1)
    def test_map_reset(self):
        p=Pipeline(VEHICLE);p.step(scene('straight'),1,True)
        s=scene('straight');s.version=2
        self.assertEqual(p.step(s,1,True)['reason'],'map_version_changed')
    def test_nan_feedback(self):
        s=scene('straight');s.feedback=(float('nan'),0,1)
        self.assertEqual(Pipeline(VEHICLE).step(s,1)['reason'],'invalid_feedback')
    def test_excess_speed(self):
        s=scene('straight');s.feedback=(9,0,1)
        self.assertEqual(Pipeline(VEHICLE).step(s,1)['reason'],'speed_outside_envelope')
    def test_coordinate(self):
        p=transform((1,0),(0,0,math.pi/2))
        self.assertAlmostEqual(p[0],0);self.assertAlmostEqual(p[1],1)
    def test_parametric_hairpin_geometry(self):
        p=spline([(0,0),(3,0),(5,2),(3,4),(0,4)])
        self.assertGreater(max(x for x,y in p),p[-1][0])
        self.assertEqual(len(curvature(p)),len(p))
    def test_missing_cones_fails(self):
        s=scene('straight');s.cones=s.cones[:2]
        self.assertFalse(Pipeline(VEHICLE).step(s,1)['valid'])

if __name__=='__main__':unittest.main()
