import os,sys,unittest
ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0]=[os.path.join(ROOT,'src'),os.path.join(ROOT,'scripts')]
from closed_loop_demo import run

class MultiStageTests(unittest.TestCase):
    def test_mirrored_alternating(self):
        r=run('straight',duration=95,obstacles=[(4,-.2,.2),(7,.2,.2)],goal_x=10)
        self.assertTrue(r['goal_reached'],r['reason']);self.assertIsNone(r['reason'])
        self.assertTrue(r['motion_safe']);self.assertTrue(r['stopped'])
    def test_delayed_alternating(self):
        r=run('straight',duration=95,obstacles=[(4,.2,.2),(7,-.2,.2)],goal_x=10,delay_steps=3)
        self.assertTrue(r['goal_reached'],r['reason']);self.assertIsNone(r['reason'])
        self.assertTrue(r['motion_safe']);self.assertTrue(r['stopped'])
    def test_blocked_not_forced(self):
        r=run('straight',duration=35,obstacles=[(4,0,1.2),(7,0,1.2)])
        self.assertFalse(r['goal_reached']);self.assertEqual(r['reason'],'collision')
        self.assertTrue(r['motion_safe']);self.assertTrue(r['stopped'])
if __name__=='__main__':unittest.main()
