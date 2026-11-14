import os,sys,unittest
ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0]=[os.path.join(ROOT,'src'),os.path.join(ROOT,'scripts')]
from obstacle_avoidance_event.compatibility import LegacySpeedEstimator,LegacyCompatibleRunner,SOURCE
from obstacle_avoidance_event.pipeline import Pipeline
from obstacle_avoidance_event.scene_adapter import SceneAssembler
from obstacle_avoidance_event.execution_adapter import LocalExecutionAdapter
from legacy_compatible_demo import run
from offline_demo import scene,VEHICLE


def sample(stamp=1.,kind='straight'):
    s=scene(kind)
    return (dict(pose=s.pose,cones=s.cones,stamp=stamp,frame='map',pose_reference='rear_axle'),
            dict(free=s.free,stamp=stamp,frame='map',version=1,source='synthetic'))


class CompatibilityTests(unittest.TestCase):
    def test_legacy_command_step_and_stop_bypass(self):
        adapter=LocalExecutionAdapter(1,.5,max_steering_step=.012)
        candidate=dict(candidate_valid=True,steering=.4,speed_limit=.2)
        self.assertAlmostEqual(adapter.step(candidate,1.,1.)['steering'],.012)
        candidate['steering']=-.4
        self.assertAlmostEqual(adapter.step(candidate,1.1,1.1)['steering'],0.)
        self.assertEqual(adapter.step(candidate,1.2,1.2,stop_requested=True)['action'],'stop')
        self.assertTrue(adapter.client.requests[-1]['stop'])

    def test_estimator_matches_legacy_filter_and_clip(self):
        e=LegacySpeedEstimator(4.1667)
        with self.assertRaises(ValueError):e.update((0,0,0),1.,1.)
        v=e.update((.2,0,0),1.1,1.1)
        self.assertAlmostEqual(v['speed'],.6)
        self.assertEqual(v['source'],SOURCE);self.assertIsNone(v['steering'])
        v=e.update((.4,0,0),1.2,1.2)
        self.assertAlmostEqual(v['speed'],1.02)

    def test_estimator_gap_duplicate_future_and_stale(self):
        e=LegacySpeedEstimator(1)
        with self.assertRaises(ValueError):e.update((0,0,0),1.,1.)
        for stamp,now in ((1.,1.1),(3.,2.),(1.1,3.)):
            with self.assertRaises(ValueError):e.update((0,0,0),stamp,now)
        with self.assertRaises(ValueError):e.update((0,0,0),3.,3.)

    def test_strict_still_requires_real_feedback(self):
        s=scene('straight');s.feedback=None
        s.speed_estimate=dict(speed=0.,stamp=1.,source=SOURCE)
        self.assertEqual(Pipeline(VEHICLE).step(s,1.)['reason'],'feedback_unavailable')

    def test_compatible_curve_without_measured_wheel(self):
        s=scene('s_curve');s.feedback=None
        s.speed_estimate=dict(speed=0.,stamp=1.,source=SOURCE)
        result=Pipeline(VEHICLE,mode='legacy_compatible').step(s,1.,True)
        self.assertTrue(result['valid'],result['reason'])
        self.assertIsNone(result['actual_steering'])
        self.assertFalse(result['vehicle_output'])

    def test_bad_estimate_and_overspeed_rejected(self):
        for estimate in (None,dict(speed=0.,stamp=1.,source='hardware'),
                         dict(speed=0.,stamp=None,source=SOURCE),
                         dict(speed=10.,stamp=1.,source=SOURCE)):
            s=scene('straight');s.feedback=None;s.speed_estimate=estimate
            self.assertFalse(Pipeline(VEHICLE,mode='legacy_compatible').step(s,1.)['valid'])

    def test_scene_never_fabricates_measured_steering(self):
        a,f=sample();estimated=dict(speed=0.,stamp=1.,source=SOURCE,steering=None)
        s=SceneAssembler(synthetic=True,mode='legacy_compatible').build(a,f,estimated,1.,1)
        self.assertIsNone(s.feedback)
        self.assertIsNone(s.speed_estimate['steering'])
        estimated['steering']=.1
        with self.assertRaises(ValueError):
            SceneAssembler(synthetic=True,mode='legacy_compatible').build(a,f,estimated,1.,1)

    def test_wait_run_stop_without_res_as_feedback(self):
        for kind in ('straight','s_curve'):
            result=run(kind)
            self.assertEqual([r['execution']['action'] for r in result['rows']],['wait','steer','steer','stop'])
            self.assertTrue(result['recorded_requests'][-1]['stop'])
            self.assertTrue(all(r['hardware_as_state'] is None and r['hardware_go'] is None for r in result['rows']))

    def test_data_loss_latches_after_running(self):
        r=LegacyCompatibleRunner(VEHICLE,1,synthetic=True)
        a,f=sample();r.step(a,f,1.)
        a,f=sample(1.1);self.assertTrue(r.step(a,f,1.1)['candidate_valid'])
        a,f=sample(1.2);self.assertEqual(r.step(a,None,1.2)['execution']['action'],'stop')
        a,f=sample(1.3);self.assertEqual(r.step(a,f,1.3)['execution']['action'],'stop')

    def test_collision_and_stale_still_stop(self):
        for kind in ('collision','stale'):
            r=LegacyCompatibleRunner(VEHICLE,1,synthetic=True)
            a,f=sample();r.step(a,f,1.)
            a,f=sample(1.1);r.step(a,f,1.1)
            a,f=sample(1.2)
            if kind=='collision':a['obstacles']=[(3.,0.,1.)]
            out=r.step(a,f,3. if kind=='stale' else 1.2)
            self.assertEqual(out['execution']['action'],'stop')

    def test_external_emergency_and_unknown_mode(self):
        r=LegacyCompatibleRunner(VEHICLE,1,synthetic=True)
        a,f=sample();self.assertEqual(r.step(a,f,1.,emergency=True)['execution']['action'],'stop')
        with self.assertRaises(ValueError):Pipeline(VEHICLE,mode='other')


if __name__=='__main__':unittest.main()
