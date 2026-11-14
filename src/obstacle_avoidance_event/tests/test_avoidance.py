import os,sys,unittest
ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0]=[os.path.join(ROOT,'src'),os.path.join(ROOT,'scripts')]
from offline_demo import scene,VEHICLE
from obstacle_avoidance_event.pipeline import Pipeline
from obstacle_avoidance_event.trajectory import validate

class AvoidanceTests(unittest.TestCase):
    def test_center_obstacle_deviates(self):
        s=scene('straight');s.obstacles=[(5,0,.2)]
        r=Pipeline(VEHICLE).step(s,1,True)
        self.assertTrue(r['valid'],r['reason']);self.assertNotEqual(r['selected_offset'],0)
        validate(r['path'],s,VEHICLE)
    def test_blocked_corridor_rejected(self):
        s=scene('straight');s.obstacles=[(5,0,1.2)]
        r=Pipeline(VEHICLE).step(s,1,True)
        self.assertFalse(r['valid']);self.assertEqual(r['reason'],'collision')
    def test_obstacle_closed_loop_success(self):
        from closed_loop_demo import run
        r=run('straight',duration=70,obstacles=[(5,0,.2)])
        self.assertTrue(r['goal_reached'])
        self.assertIsNone(r['reason'])
        self.assertTrue(r['stopped']);self.assertTrue(r['motion_safe'])
        self.assertLess(abs(r['rows'][-1]['y']),.12)
    def test_committed_path_revalidated(self):
        p=Pipeline(VEHICLE);s=scene('straight');s.obstacles=[(5,0,.2)]
        self.assertTrue(p.step(s,1,True)['valid'])
        saved=list(p.committed_path)
        s.stamp=1.1;s.feedback=(0,0,1.1);s.free=lambda point,radius:False
        result=p.step(s,1.1,True)
        self.assertFalse(result['valid']);self.assertEqual(result['state'],'STOP_LATCHED')
        self.assertEqual(saved,p.committed_path)
    def test_delayed_avoidance(self):
        from closed_loop_demo import run
        r=run('straight',duration=70,obstacles=[(5,0,.2)],delay_steps=3)
        self.assertTrue(r['goal_reached']);self.assertIsNone(r['reason'])
        self.assertTrue(r['motion_safe']);self.assertTrue(r['stopped'])
    def test_clear_route_unchanged(self):
        r=Pipeline(VEHICLE).step(scene('straight'),1,True)
        self.assertEqual(r['selected_offset'],0)
    def test_unknown_side_cannot_be_used(self):
        s=scene('straight');s.obstacles=[(5,0,.2)]
        original=s.free;s.free=lambda p,r:original(p,r) and abs(p[1])+r<.5
        r=Pipeline(VEHICLE).step(s,1,True)
        self.assertFalse(r['valid'])
if __name__=='__main__':unittest.main()
