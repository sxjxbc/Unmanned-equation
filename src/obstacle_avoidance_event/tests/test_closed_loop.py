import os,sys,unittest,math
ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0]=[os.path.join(ROOT,'src'),os.path.join(ROOT,'scripts')]
from closed_loop_demo import run
from offline_demo import VEHICLE
from obstacle_avoidance_event.simulation import Bicycle

class ClosedLoopTests(unittest.TestCase):
    def test_straight_goal_and_stop(self):
        r=run('straight',duration=70)
        self.assertTrue(r['goal_reached']);self.assertTrue(r['stopped']);self.assertIsNone(r['reason'])
        self.assertLess(r['max_error'],.01)
    def test_s_curve_goal_tracking(self):
        r=run('s_curve',duration=70)
        self.assertTrue(r['goal_reached']);self.assertTrue(r['stopped']);self.assertIsNone(r['reason'])
        self.assertLess(r['max_error'],.2)
        self.assertTrue(all(not row['vehicle_output'] for row in r['rows']))
    def test_stale_stop_latched_and_brakes(self):
        r=run('straight',stale_at=30)
        self.assertEqual(r['reason'],'observation_stale');self.assertTrue(r['stopped'])
        failed=[row for row in r['rows'] if not row['valid']]
        self.assertTrue(failed);self.assertTrue(all(row['state']=='STOP_LATCHED' for row in failed))
        self.assertLess(failed[-1]['x']-failed[0]['x'],.1)
    def test_actuator_rate_limits(self):
        b=Bicycle(VEHICLE);b.step(.7,.5,.01)
        self.assertLessEqual(b.steering,VEHICLE['steer_rate']*.01)
        self.assertLessEqual(b.speed,.00300001)
    def test_left_steer_moves_left(self):
        b=Bicycle(VEHICLE)
        for i in range(40):b.step(.2,.2,.05)
        self.assertGreater(b.pose[1],0);self.assertGreater(b.pose[2],0)
    def test_invalid_timestep(self):
        with self.assertRaises(ValueError):Bicycle(VEHICLE).step(0,0,0)

if __name__=='__main__':unittest.main()
