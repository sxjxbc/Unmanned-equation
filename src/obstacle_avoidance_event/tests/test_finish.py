import os,sys,unittest
ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0,os.path.join(ROOT,'src'))
from obstacle_avoidance_event.finish import StartupGate,FinishController

def signals(t,speed=0,**overrides):
    d=dict(emergency=False,ready=True,go=False,driving=True,sound_done=True,speed=speed,as_finished=False,ebs_active=True,drive_disabled=True)
    d.update(overrides)
    return dict((k,(v,t)) for k,v in d.items())
class FinishTests(unittest.TestCase):
    def controller(self):return FinishController(1.76618,1.16382,1.4,.5,.2)
    def test_laps_to_finish_demo(self):
        sys.path.insert(0,os.path.join(ROOT,'scripts'))
        from trackdrive_finish_demo import evaluate
        r=evaluate();self.assertEqual(r['phase'],'FINISHED_CONFIRMED')
        self.assertEqual(r['laps'],3);self.assertLessEqual(r['front_after'],30)
        self.assertFalse(r['vehicle_output'])
    def test_front_in_budget(self):
        r=self.controller().step(1,27,0,0,signals(1))
        self.assertAlmostEqual(r['remaining'],30-27-1.76618-.5)
    def test_front_outside_rejected(self):
        self.assertEqual(self.controller().step(1,29,0,0,signals(1))['phase'],'FAULT')
    def test_not_complete_before_body_passes_line(self):
        self.assertEqual(self.controller().step(1,0,0,0,signals(1))['phase'],'STOPPING')
    def test_dwell_and_hardware_ack(self):
        f=self.controller();self.assertEqual(f.step(1,10,0,0,signals(1))['phase'],'STOPPING')
        self.assertEqual(f.step(1.6,10,0,0,signals(1.6))['phase'],'COMPLETE_REQUEST')
        self.assertEqual(f.step(1.7,10,0,0,signals(1.7,as_finished=True))['phase'],'FINISHED_CONFIRMED')
    def test_stale_speed(self):
        self.assertEqual(self.controller().step(2,10,0,0,signals(1))['phase'],'FAULT')
    def test_braking_budget(self):
        self.assertEqual(self.controller().step(1,20,6,0,signals(1,6))['reason'],'insufficient_braking_distance')
    def test_no_go(self):self.assertFalse(StartupGate().step(0,signals(0))['permitted'])
    def test_go_after_delay(self):
        g=StartupGate();g.step(0,signals(0));self.assertTrue(g.step(5,signals(5,go=True))['permitted'])
    def test_held_early_go_not_accepted(self):
        g=StartupGate();g.step(0,signals(0,go=True));self.assertFalse(g.step(5,signals(5,go=True))['permitted'])
    def test_sound_required(self):
        g=StartupGate();g.step(0,signals(0));self.assertFalse(g.step(5,signals(5,go=True,sound_done=False))['permitted'])
    def test_emergency_latched(self):
        g=StartupGate();g.step(0,signals(0,emergency=True));self.assertFalse(g.step(5,signals(5,go=True))['permitted'])
if __name__=='__main__':unittest.main()
