#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Run-only failure latch. Waiting is recoverable and never emits a stop."""
import math


class LegacyRunGuard(object):
    def __init__(self, pose_timeout=0.8, path_timeout=1.0, road_timeout=1.0):
        self.limits = dict(pose=pose_timeout, imu=pose_timeout,
                           path=path_timeout, road=road_timeout)
        if any(not (0 < v < float('inf')) for v in self.limits.values()):
            raise ValueError('Guard timeouts must be finite and positive')
        self.samples = {}
        self.running = False
        self.stop_reason = None
        self.last_tick = None

    def observe(self, name, now, source_age=0.0):
        if name not in self.limits:
            raise ValueError(name)
        if any(math.isnan(v) or math.isinf(v) for v in (now, source_age)):
            return False
        if source_age < -0.1:
            return False
        self.samples[name] = (now, max(0.0, source_age))
        return True

    def evaluate(self, now):
        if self.last_tick is not None and now < self.last_tick:
            self.samples.clear()
            if self.running and self.stop_reason is None:
                self.stop_reason = 'clock_rewind'
        self.last_tick = now
        reasons = []
        for name in ('pose', 'imu', 'path', 'road'):
            sample = self.samples.get(name)
            if sample is None or not (0 <= now-sample[0] and
                                      now-sample[0]+sample[1] <= self.limits[name]):
                reasons.append(name+'_stale')
        if self.running and reasons and self.stop_reason is None:
            self.stop_reason = reasons[0]
        return reasons

    def start(self, now):
        if not self.evaluate(now) and self.stop_reason is None:
            self.running = True
            return True
        return False

    def stop(self, reason):
        if self.running and self.stop_reason is None:
            self.stop_reason = reason
