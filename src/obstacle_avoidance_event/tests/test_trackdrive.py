import os,sys,unittest
ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0,os.path.join(ROOT,'src'))
from obstacle_avoidance_event.trackdrive import TrackdriveMission

class TrackdriveTests(unittest.TestCase):
    def armed(self):
        m=TrackdriveMission(200,3,.5,.2)
        m.step(0,194,0,ready=True)
        m.step(5,194,0,ready=True,go=True)
        return m
    def send(self,m,t,s,v=2,gate=False):
        return m.step(t,s,v,gate=gate,ready=True,driving_ack=True,sound_done=True)
    def test_supervisor_blocks_without_go(self):
        from obstacle_avoidance_event.pipeline import Pipeline
        from obstacle_avoidance_event.trackdrive import TrackdriveSupervisor
        sys.path.insert(0,os.path.join(ROOT,'scripts'))
        from offline_demo import scene,VEHICLE
        s=TrackdriveSupervisor(Pipeline(VEHICLE),TrackdriveMission(200,3,.5,.2))
        r=s.step(scene('straight'),1,194,ready=True)
        self.assertFalse(r['candidate_valid']);self.assertEqual(r['speed_limit'],0)
        self.assertFalse(r['vehicle_output'])
    def test_no_go_no_motion(self):
        m=TrackdriveMission(200,3,.5,.2)
        self.assertFalse(m.step(10,194,0,ready=True)['motion_permitted'])
    def test_early_go_ignored(self):
        m=TrackdriveMission(200,3,.5,.2);m.step(0,194,0,ready=True)
        self.assertEqual(m.step(1,194,0,ready=True,go=True)['phase'],'WAIT_READY')
    def test_sound_ack_required(self):
        m=self.armed();self.assertFalse(m.step(6,194,0,ready=True,driving_ack=True)['motion_permitted'])
    def test_three_laps_and_stop(self):
        m=self.armed();self.send(m,6,196);self.send(m,7,198)
        first=self.send(m,8,0,gate=True);self.assertEqual(first['laps'],0)
        t=8
        for lap in range(3):
            for s in range(2,200,2):t+=1;self.send(m,t,s)
            t+=1;out=self.send(m,t,0,gate=True)
            self.assertEqual(out['laps'],lap+1)
        self.assertEqual(out['phase'],'STOPPING')
        final=self.send(m,t+1,1,0)
        self.assertEqual(final['phase'],'COMPLETE_CANDIDATE')
    def test_jitter_not_a_lap(self):
        m=self.armed();self.send(m,6,196)
        self.assertEqual(self.send(m,7,195.5)['phase'],'FAULT')
    def test_gate_confirmation(self):
        m=self.armed();self.send(m,6,196);self.send(m,7,198)
        self.assertEqual(self.send(m,8,0)['reason'],'unconfirmed_finish_gate')
    def test_emergency_latched(self):
        m=self.armed();self.assertEqual(m.step(6,194,0,emergency=True)['phase'],'FAULT')
        self.assertEqual(self.send(m,7,194)['phase'],'FAULT')
if __name__=='__main__':unittest.main()
