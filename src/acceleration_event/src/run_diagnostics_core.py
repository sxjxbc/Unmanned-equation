#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Read-only run evidence evaluator; ROS independent, Python 2/3 compatible."""
from __future__ import division
import math
from collections import deque


def finite(value):
    return not (math.isnan(value) or math.isinf(value))


class RunDiagnostics(object):
    def __init__(self, mode='legacy', timeout=0.7, window=3.0):
        if mode not in ('legacy', 'lidar'):
            raise ValueError('mode must be legacy or lidar')
        self.mode, self.timeout, self.window = mode, timeout, window
        self.latest = {}
        self.history = {}
        self.invalid = {}
        self.active = False
        self.active_since = None
        self.last_time = None

    def ingest(self, name, receipt, value, stamp=None):
        self.latest[name] = dict(receipt=receipt, stamp=stamp, value=value)
        history = self.history.setdefault(name, deque(maxlen=2000))
        history.append((receipt, value))
        while history and receipt-history[0][0] > self.window:
            history.popleft()

    def evaluate(self, now):
        findings = []

        def add(code, evidence, confidence='observed'):
            findings.append(dict(code=code, confidence=confidence, evidence=evidence))

        if self.last_time is not None and now < self.last_time:
            self.latest.clear()
            self.history.clear()
            self.active_since = now if self.active else None
            add('CLOCK_REWIND', 'ROS time moved backwards; evidence window reset')
        self.last_time = now
        if not self.active:
            self.active_since = None
            return findings
        if self.active_since is None:
            self.active_since = now
        required = ['gps', 'imu', 'pose', 'lidar', 'steering']
        if self.mode == 'legacy':
            required += ['path', 'map']
        else:
            required += ['status']
        for name in required:
            sample = self.latest.get(name)
            if sample is None:
                if now-self.active_since >= self.timeout:
                    add('MISSING_'+name.upper(), 'No valid message since monitor start')
                continue
            age = now-sample['receipt']
            if age > self.timeout:
                add('STALE_'+name.upper(), 'receipt_age_s=%.3f' % age)
            stamp = sample['stamp']
            if stamp is not None:
                if stamp <= 0:
                    add('ZERO_STAMP_'+name.upper(), 'header stamp is zero')
                elif now-stamp > self.timeout or stamp-now > 0.1:
                    add('BAD_STAMP_'+name.upper(), 'header_age_s=%.3f' % (now-stamp))
        for name, when in self.invalid.items():
            if now-when < self.timeout:
                add('INVALID_'+name.upper(), 'Malformed or non-finite input rejected')

        def fresh(name):
            s = self.latest.get(name)
            return s is not None and 0 <= now-s['receipt'] <= self.timeout

        for name in ('lidar', 'map', 'path'):
            if fresh(name) and self.latest[name]['value'] == 0:
                add('EMPTY_'+name.upper(), 'Messages arrive but contain zero points')
        for name in ('gps', 'pose'):
            h = [(t, v) for t, v in self.history.get(name, [])
                 if now-self.window <= t <= now and t >= self.active_since]
            if fresh(name) and len(h) >= 3 and h[-1][0]-h[0][0] >= self.window*0.8:
                origin = h[0][1]
                extent = max(math.hypot(v[0]-origin[0], v[1]-origin[1]) for _, v in h)
                if extent < 0.05:
                    add('POSITION_STATIC_'+name.upper(),
                        'extent_m=%.4f over %.2fs; valid only if vehicle actually moving' %
                        (extent, h[-1][0]-h[0][0]), 'suspected')
        if fresh('status'):
            s = self.latest['status']['value']
            if s.get('stop_reason'):
                add('CONTROLLER_STOP', str(s['stop_reason']))
            if s.get('input_reason') in ('geometry_rejected', 'waiting_time_alignment', 'observation_expired'):
                add('LIDAR_REJECTED', str(s['input_reason']))
            if not s.get('enabled'):
                add('VEHICLE_OUTPUT_DISABLED', 'Fusion is shadow mode or output disabled')
            if s.get('enabled') and s.get('started') and not s.get('udp_sent'):
                add('UDP_SEND_NOT_CONFIRMED', 'Controller reports udp_sent=false')
        if fresh('legacy_status'):
            s = self.latest['legacy_status']['value']
            if s.get('stop_reason'):
                add('LEGACY_CONTROLLER_STOP', str(s['stop_reason']))
        if fresh('steering'):
            h = [(t, v) for t, v in self.history.get('steering', [])
                 if now-self.window <= t <= now and t >= self.active_since]
            if h and h[-1][0]-h[0][0] >= self.window*0.8:
                if min(abs(v) for _, v in h) >= 0.48:
                    add('STEERING_NEAR_LIMIT', 'Command magnitude >=0.48 rad throughout window', 'suspected')
        return findings
