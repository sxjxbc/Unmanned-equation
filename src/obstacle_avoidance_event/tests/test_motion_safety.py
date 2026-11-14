import os,sys,unittest
ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0]=[os.path.join(ROOT,'src'),os.path.join(ROOT,'scripts')]
from closed_loop_demo import run
from offline_demo import scene,VEHICLE
from obstacle_avoidance_event.simulation import audit_motion

class MotionSafetyTests(unittest.TestCase):
    def test_delayed_s_curve(self):
        r=run('s_curve',duration=70,delay_steps=3)
        self.assertTrue(r['goal_reached'],r['reason']);self.assertTrue(r['motion_safe'])
        self.assertLess(r['max_error'],.2)
    def test_missing_observation_stops(self):
        r=run('straight',missing_at=30)
        self.assertTrue(r['stopped']);self.assertFalse(r['goal_reached'])
        self.assertEqual(r['reason'],'invalid_point_count_or_duplicates')
        self.assertTrue(r['motion_safe'])
    def test_tunneling_obstacle(self):
        s=scene('straight');s.obstacles=[(2,0,.1)];s.free=lambda p,r:True;s.cones=[]
        self.assertEqual(audit_motion((1,0,0),(3,0,0),s,VEHICLE,2),'actual_motion_collision')
    def test_turn_body_bound(self):
        s=scene('straight');s.free=lambda p,r:False
        self.assertEqual(audit_motion((1,0,0),(1,0,1),s,VEHICLE,0),'actual_motion_unknown_or_boundary')
if __name__=='__main__':unittest.main()
