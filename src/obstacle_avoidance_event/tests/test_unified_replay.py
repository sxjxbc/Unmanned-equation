import os,sys,unittest
ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0]=[os.path.join(ROOT,'src'),os.path.join(ROOT,'scripts')]
from unified_replay_demo import run

class UnifiedReplayTests(unittest.TestCase):
    def test_real_planner_three_laps_and_stop_geometry_gate(self):
        result=run()
        self.assertEqual(result['last']['laps'],3)
        self.assertEqual(result['last']['reason'],'stop_geometry_unconfirmed')
        self.assertFalse(result['last']['candidate_valid'])
        self.assertTrue(all(r['reason'] is None for r in result['rows'][:-1]))
        self.assertFalse(result['rows'][0]['candidate_valid'])
        self.assertTrue(result['rows'][1]['candidate_valid'])
        self.assertFalse(result['vehicle_output'])
        self.assertEqual(result['rows'][0]['execution']['action'],'wait')
        self.assertEqual(result['rows'][1]['execution']['action'],'steer')
        self.assertEqual(result['last']['execution']['action'],'stop')
        self.assertTrue(result['recorded_requests'][-1]['stop'])
        self.assertFalse(any(r['execution']['vehicle_output'] for r in result['rows']))
if __name__=='__main__':unittest.main()
