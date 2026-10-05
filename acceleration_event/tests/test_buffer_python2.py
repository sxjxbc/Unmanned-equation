#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Original Python2-compatible offline checks, standard library only."""
from __future__ import print_function
import os
import sys
import math
import unittest
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))
from data_quality import PoseBuffer, fresh, monotonic


class OriginalRuntimeTests(unittest.TestCase):
    def test_monotonic(self):
        first = monotonic()
        self.assertGreaterEqual(monotonic(), first)

    def test_freshness_both_clocks(self):
        buffer = PoseBuffer()
        buffer.add(10., 0., 0., 0., received=100.)
        self.assertTrue(buffer.latest(10.1, .5, 100.1)['valid'])
        self.assertFalse(buffer.latest(10.1, .5, 101.)['valid'])

    def test_order_and_jump(self):
        buffer = PoseBuffer()
        buffer.add(10., 0., 0., 0.)
        self.assertFalse(buffer.add(10., 0., 0., 0.))
        self.assertFalse(buffer.add(10.1, 20., 0., 0.))
        self.assertTrue(buffer.add(10.2, .2, 0., 0.))

    def test_interpolation(self):
        buffer = PoseBuffer()
        buffer.add(10., 0., 0., math.radians(179))
        buffer.add(10.2, 2., 0., math.radians(-179))
        pose = buffer.at(10.1, .15)
        self.assertAlmostEqual(pose['x'], 1.)
        self.assertAlmostEqual(abs(pose['yaw']), math.pi)

    def test_invalid_timestamp(self):
        self.assertFalse(fresh(None, 10., .5))
        self.assertFalse(fresh(float('nan'), 10., .5))
        self.assertFalse(fresh(0., 10., .5))


if __name__ == '__main__':
    unittest.main(verbosity=2)
