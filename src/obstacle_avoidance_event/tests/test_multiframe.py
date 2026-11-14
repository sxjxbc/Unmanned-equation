import os,sys,unittest
ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0]=[os.path.join(ROOT,'src'),os.path.join(ROOT,'scripts')]
from offline_demo import scene,VEHICLE
from obstacle_avoidance_event.pipeline import Pipeline

class MultiFrameTests(unittest.TestCase):
    def frame(self, stamp, shift=0.):
        s=scene('straight');s.stamp=stamp;s.feedback=(0.,0.,stamp)
        s.cones=list(reversed([(x+shift,y) for x,y in s.cones]))
        return s
    def test_reordered_jitter_preserves_identity(self):
        p=Pipeline(VEHICLE);a=p.step(scene('straight'),1,True)
        b=p.step(self.frame(1.1,.01),1.1,True)
        self.assertTrue(b['valid'],b['reason'])
        self.assertEqual(a['stable_gate_sequence'],b['stable_gate_sequence'])
    def test_rejected_frame_does_not_commit_tracker(self):
        p=Pipeline(VEHICLE);p.step(scene('straight'),1,True)
        old=dict(p.tracker.points);s=self.frame(1.1,.01);s.obstacles=[(4,0,1)]
        out=p.step(s,1.1,True)
        self.assertEqual(out['state'],'STOP_LATCHED');self.assertEqual(p.tracker.points,old)
        self.assertEqual(p.tracker.stamp,1.)
    def test_duplicate_stamp_rejected(self):
        p=Pipeline(VEHICLE);p.step(scene('straight'),1,True)
        self.assertEqual(p.step(scene('straight'),1,True)['reason'],'non_monotonic_observation')
    def test_pose_jump_rejected(self):
        p=Pipeline(VEHICLE);p.step(scene('straight'),1,True)
        s=self.frame(1.1);s.pose=(2.,0.,0.)
        result=p.step(s,1.1,True)
        self.assertFalse(result['valid']);self.assertIn(result['reason'],('route_progress_jump','pose_progress_jump'))
    def test_unrelated_map_loses_identity(self):
        p=Pipeline(VEHICLE);p.step(scene('straight'),1,True)
        s=self.frame(1.1,.5)
        result=p.step(s,1.1,True)
        self.assertFalse(result['valid']);self.assertEqual(result['reason'],'route_identity_lost')
    def test_sequential_frames(self):
        p=Pipeline(VEHICLE)
        for i in range(10):
            stamp=1+i*.1;s=self.frame(stamp,.005*i)
            out=p.step(s,stamp,True)
            self.assertTrue(out['valid'],out['reason']);self.assertFalse(out['vehicle_output'])

if __name__=='__main__':unittest.main()
