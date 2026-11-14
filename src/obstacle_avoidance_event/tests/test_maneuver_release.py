import os,sys,unittest
ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0]=[os.path.join(ROOT,'src'),os.path.join(ROOT,'scripts')]
from closed_loop_demo import run

class ManeuverReleaseTests(unittest.TestCase):
    def test_release_and_continue(self):
        r=run('straight',duration=85,obstacles=[(5,0,.2)],goal_x=10)
        self.assertTrue(r['goal_reached']);self.assertIsNone(r['reason'])
        self.assertTrue(r['motion_safe']);self.assertTrue(r['stopped'])
        released=[i for i,row in enumerate(r['rows']) if row['maneuver_released']]
        self.assertEqual(len(released),1)
        self.assertTrue(all(row['selected_offset']==0 for row in r['rows'][released[0]:]))
        self.assertLess(abs(r['rows'][-1]['y']),.12)
    def test_two_center_obstacles(self):
        r=run('straight',duration=95,obstacles=[(4,0,.2),(7,0,.2)],goal_x=10)
        self.assertTrue(r['goal_reached']);self.assertIsNone(r['reason'])
        self.assertTrue(r['motion_safe']);self.assertTrue(r['stopped'])
    def test_alternating_obstacles_pass(self):
        r=run('straight',duration=95,obstacles=[(4,.2,.2),(7,-.2,.2)],goal_x=10)
        self.assertTrue(r['goal_reached']);self.assertIsNone(r['reason'])
        self.assertTrue(r['stopped']);self.assertTrue(r['motion_safe'])
if __name__=='__main__':unittest.main()
