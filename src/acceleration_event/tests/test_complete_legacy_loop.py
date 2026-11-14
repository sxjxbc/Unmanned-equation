#!/usr/bin/env python3
"""Full production control loop through the explicitly assumed offline plant."""
import importlib.util
from pathlib import Path
import unittest

spec=importlib.util.spec_from_file_location('legacy_loop_validation',
    Path(__file__).resolve().parents[1]/'scripts'/'legacy_loop_validation.py')
loop=importlib.util.module_from_spec(spec)
spec.loader.exec_module(loop)


class CompleteLoopTests(unittest.TestCase):
    def test_stationary_start_executes_real_control_every_tick(self):
        summary,records=loop.run_case('stationary',speed=0.,duration=3.)
        self.assertTrue(summary['started'])
        self.assertIsNone(summary['stop_reason'])
        self.assertEqual(60,summary['full_control_records'])
        self.assertEqual([],summary['errors'])
        self.assertAlmostEqual(.6,summary['final_y'])

    def test_simple_offset_cases_are_symmetric_under_assumed_plant(self):
        right,rr=loop.run_case('right')
        left,lr=loop.run_case('left',initial_y=-.6)
        self.assertLess(rr[0]['command'],0.)
        self.assertGreater(lr[0]['command'],0.)
        self.assertLess(abs(right['final_y']),.1)
        self.assertLess(abs(left['final_y']),.1)
        self.assertAlmostEqual(right['final_y'],-left['final_y'],places=8)
        self.assertEqual(240,right['full_control_records'])

    def test_one_row_continuous_control_does_not_require_second_row(self):
        summary,_=loop.run_case('one_row',rows=1)
        self.assertTrue(summary['started'])
        self.assertIsNone(summary['stop_reason'])
        self.assertLess(abs(summary['final_y']),.1)

    def test_short_road_gap_recovers_without_latch(self):
        summary,_=loop.run_case('short_gap',drop=('road',1.,1.2),duration=4.)
        self.assertIsNone(summary['stop_reason'])
        self.assertEqual(80,summary['full_control_records'])

    def test_long_gaps_latch_stop_and_do_not_resume_on_return(self):
        for source in ('road','pose','imu'):
            summary,records=loop.run_case('loss_'+source,drop=(source,1.,2.5),duration=4.)
            self.assertEqual(source+'_stale',summary['stop_reason'])
            stopped=[r for r in records if r['state']=='stopped']
            self.assertTrue(stopped)
            self.assertEqual('stopped',records[-1]['state'])
            self.assertTrue(all(r['command']==0. for r in stopped))
            self.assertEqual([],summary['errors'])


if __name__=='__main__':
    unittest.main()
