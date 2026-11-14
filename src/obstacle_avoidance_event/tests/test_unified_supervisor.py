import os,sys,math,unittest
ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0]=[os.path.join(ROOT,'src'),os.path.join(ROOT,'scripts')]
from obstacle_avoidance_event.closed_route import ClosedRoute,RouteProgress
from obstacle_avoidance_event.trackdrive import TrackdriveMission,TrackdriveSupervisor
from obstacle_avoidance_event.finish import FinishController
from obstacle_avoidance_event.model import Scene
from trackdrive_finish_demo import feedback

class PlannerStub(object):
    # Deliberately only tests arbitration, not geometric planning.
    def step(self,*args,**kwargs):return dict(valid=True,speed_limit=2.,steering=.1)

class UnifiedSupervisorTests(unittest.TestCase):
    def coordinator(self):
        route=ClosedRoute([(32*math.cos(2*math.pi*i/400.),32*math.sin(2*math.pi*i/400.)) for i in range(400)])
        return TrackdriveSupervisor(PlannerStub(),TrackdriveMission(route.length,3,.5,.2),
            RouteProgress(route,3),FinishController(1.76618,1.16382,1.4,.5,.2))
    def scene(self,t):
        a=-.08
        return Scene([], (32*math.cos(a),32*math.sin(a),a+math.pi/2),t,feedback=(0.,0.,t))
    def test_legacy_bypass_rejected(self):
        s=self.coordinator();r=s.step(self.scene(0),0,progress=100,ready=True,go=True)
        self.assertEqual(r['reason'],'legacy_authorization_rejected');self.assertFalse(r['candidate_valid'])
    def test_requires_timestamped_signals(self):
        self.assertEqual(self.coordinator().step(self.scene(0),0)['reason'],'timestamped_signals_required')
    def test_no_go_then_valid_go(self):
        s=self.coordinator();r=s.step(self.scene(0),0,signals=feedback(0))
        self.assertFalse(r['candidate_valid'])
        r=s.step(self.scene(5),5,signals=feedback(5,go=True))
        self.assertTrue(r['candidate_valid'],r['reason']);self.assertFalse(r['vehicle_output'])
    def test_stale_latched(self):
        s=self.coordinator();s.step(self.scene(0),0,signals=feedback(0))
        s.step(self.scene(5),5,signals=feedback(5,go=True))
        r=s.step(self.scene(6),6,signals=feedback(0,go=True))
        self.assertFalse(r['candidate_valid']);self.assertIsNotNone(r['reason'])
        self.assertEqual(s.step(self.scene(7),7,signals=feedback(7,go=True))['reason'],r['reason'])
    def test_emergency_latched(self):
        s=self.coordinator();r=s.step(self.scene(0),0,signals=feedback(0,emergency=True))
        self.assertEqual(r['reason'],'res_emergency')
    def test_pre_go_stale_recovers_without_auto_go(self):
        s=self.coordinator();s.step(self.scene(0),0,signals=feedback(0))
        r=s.step(self.scene(1),1,signals=feedback(0))
        self.assertTrue(r['waiting']);self.assertIsNone(s.fault)
        self.assertFalse(s.step(self.scene(2),2,signals=feedback(2,go=True))['candidate_valid'])
        self.assertFalse(s.step(self.scene(3),3,signals=feedback(3))['candidate_valid'])
        self.assertTrue(s.step(self.scene(8),8,signals=feedback(8,go=True))['candidate_valid'])
    def test_missing_signals_wait_without_latching(self):
        s=self.coordinator();r=s.step(self.scene(0),0)
        self.assertTrue(r['waiting']);self.assertIsNone(s.fault)
        self.assertFalse(r['candidate_valid'])
    def test_progress_jump_latches(self):
        s=self.coordinator();s.step(self.scene(0),0,signals=feedback(0))
        obs=self.scene(.1);obs.pose=(0,32,math.pi)
        r=s.step(obs,.1,signals=feedback(.1))
        self.assertFalse(r['candidate_valid']);self.assertIsNotNone(r['reason'])
if __name__=='__main__':unittest.main()
