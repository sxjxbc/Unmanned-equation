import os,sys,math,unittest
ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0]=[os.path.join(ROOT,'src'),os.path.join(ROOT,'scripts')]
from obstacle_avoidance_event.corridor import Corridor
from obstacle_avoidance_event.geometry import lidar_point_in_imu,imu_point_in_map

class ContinuousCorridorTests(unittest.TestCase):
    def corridor(self):return Corridor([(0,1.5),(4,1.5),(8,1.5)],[(0,-1.5),(2,-1.5),(6,-1.5),(8,-1.5)],True)
    def test_unequal_sides(self):self.assertTrue(self.corridor().free((4,0),1.4))
    def test_disk_crossing_boundary(self):self.assertFalse(self.corridor().free((4,1.2),.4))
    def test_end_cap(self):self.assertFalse(self.corridor().free((.1,0),.2))
    def test_unconfirmed_identity(self):
        with self.assertRaises(ValueError):Corridor([(0,1),(4,1)],[(0,-1),(4,-1)])
    def test_gap(self):
        with self.assertRaises(ValueError):Corridor([(0,1),(9,1)],[(0,-1),(9,-1)],True)
    def test_reflection_rejected(self):
        with self.assertRaises(ValueError):lidar_point_in_imu((1,2,3),{'lidar_translation_in_imu_m':[0,0,0]},((1,0,0),(0,-1,0),(0,0,1)))
    def test_composed_imu_map(self):
        p=imu_point_in_map((1,0,2),(3,4,math.pi/2))
        self.assertAlmostEqual(p[0],3);self.assertAlmostEqual(p[1],5);self.assertEqual(p[2],2)
    def test_rotation_scale_rejected(self):
        with self.assertRaises(ValueError):lidar_point_in_imu((0,0,0),{},((2,0,0),(0,1,0),(0,0,1)))
    def test_synthetic_boundaries_reject_overwide_union(self):
        from corridor_geometry_demo import evaluate
        results=evaluate()
        self.assertEqual(len(results),12)
        self.assertTrue(all(not r['all_fit'] for r in results))
if __name__=='__main__':unittest.main()
