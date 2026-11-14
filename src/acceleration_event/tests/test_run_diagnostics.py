import os
import sys
import unittest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from run_diagnostics_core import RunDiagnostics


class DiagnosticsTest(unittest.TestCase):
    def codes(self, core, now):
        return set(f['code'] for f in core.evaluate(now))

    def test_stationary_before_start_is_not_fault(self):
        d = RunDiagnostics()
        for t in range(5):
            d.ingest('gps', t, (0, 0), t)
        self.assertEqual([], d.evaluate(4))

    def test_static_position_while_armed_is_only_suspected(self):
        d = RunDiagnostics()
        d.active = True
        d.evaluate(1)
        for i in range(31):
            d.ingest('gps', 1+i/10.0, (0, 0), 1+i/10.0)
        matches = [f for f in d.evaluate(4) if f['code'] == 'POSITION_STATIC_GPS']
        self.assertEqual('suspected', matches[0]['confidence'])

    def test_moving_position_not_reported_static(self):
        d = RunDiagnostics()
        d.active = True
        d.evaluate(1)
        for i in range(31):
            d.ingest('pose', 1+i/10.0, (i/10.0, 0), 1+i/10.0)
        self.assertNotIn('POSITION_STATIC_POSE', self.codes(d, 4))

    def test_header_can_freeze_while_receipts_continue(self):
        d = RunDiagnostics()
        d.active = True
        d.ingest('imu', 10, 42, 1)
        self.assertIn('BAD_STAMP_IMU', self.codes(d, 10))
        self.assertNotIn('STALE_IMU', self.codes(d, 10))

    def test_empty_and_missing_and_fusion_rejection(self):
        d = RunDiagnostics('lidar')
        d.active = True
        d.evaluate(1)
        d.ingest('lidar', 2, 0, 2)
        d.ingest('status', 2, dict(enabled=False, input_reason='geometry_rejected', stop_reason='road_stale'))
        codes = self.codes(d, 2)
        self.assertTrue({'EMPTY_LIDAR', 'MISSING_GPS', 'LIDAR_REJECTED',
                         'CONTROLLER_STOP', 'VEHICLE_OUTPUT_DISABLED'} <= codes)
        self.assertNotIn('MISSING_PATH', codes)

    def test_clock_reset_discards_old_evidence(self):
        d = RunDiagnostics()
        d.active = True
        d.ingest('pose', 10, (0, 0), 10)
        d.evaluate(10)
        self.assertIn('CLOCK_REWIND', self.codes(d, 2))
        self.assertEqual({}, d.latest)


if __name__ == '__main__':
    unittest.main()
