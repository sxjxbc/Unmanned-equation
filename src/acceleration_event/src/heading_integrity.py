#!/usr/bin/env python
"""Heading continuity gate for the project's degree-valued orientation.z."""
from __future__ import division

import math
import numbers


def finite(value):
    return isinstance(value, numbers.Real) and not isinstance(value, bool) and not (
        math.isnan(value) or math.isinf(value))


def shortest_step_deg(previous, current):
    # Normalize before subtraction to avoid overflow for large finite inputs.
    # This is only a comparison; accepted headings keep their original value.
    return abs(((current % 360.0)-(previous % 360.0)+180.0) % 360.0-180.0)


class HeadingIntegrityGuard(object):
    def __init__(self, max_rate_dps=180.0, jump_tolerance_deg=5.0,
                 max_gap_s=1.0, max_age_s=0.5, future_tolerance_s=0.1,
                 bootstrap_samples=3):
        positive = (max_rate_dps, jump_tolerance_deg, max_gap_s, max_age_s)
        if any(not finite(value) or value <= 0 for value in positive):
            raise ValueError('Heading continuity limits must be finite and positive')
        if not finite(future_tolerance_s) or future_tolerance_s < 0:
            raise ValueError('Heading future tolerance must be finite and nonnegative')
        if (isinstance(bootstrap_samples, bool) or
                not isinstance(bootstrap_samples, numbers.Integral) or bootstrap_samples < 2):
            raise ValueError('Heading bootstrap requires at least two samples')
        self.max_rate_dps, self.jump_tolerance_deg = max_rate_dps, jump_tolerance_deg
        self.max_gap_s, self.max_age_s = max_gap_s, max_age_s
        self.future_tolerance_s, self.bootstrap_samples = future_tolerance_s, bootstrap_samples
        self.heading, self.stamp = None, None
        self.ready = False
        self.seed = None
        self.counts = dict(received=0, accepted=0, rejected=0)
        self.last_reason = 'waiting'
        self.last_step_deg, self.last_limit_deg = None, None

    def reject(self, reason, reset_seed=False):
        self.counts['rejected'] += 1
        self.counts[reason] = self.counts.get(reason, 0)+1
        self.last_reason = reason
        if reset_seed and not self.ready:
            self.seed = None
        return False

    def accept(self, heading, stamp, now):
        self.counts['received'] += 1
        self.last_step_deg, self.last_limit_deg = None, None
        if not finite(heading):
            return self.reject('nonfinite', reset_seed=True)
        if not finite(stamp) or not finite(now) or stamp <= 0 or now <= 0:
            return self.reject('invalid_stamp', reset_seed=True)
        age = now-stamp
        if age > self.max_age_s:
            return self.reject('stale', reset_seed=True)
        if age < -self.future_tolerance_s:
            return self.reject('future', reset_seed=True)
        if self.ready:
            reference = (self.heading, self.stamp, self.bootstrap_samples)
        else:
            reference = self.seed
            if reference is not None and stamp-reference[1] > self.max_gap_s:
                reference = None
                self.seed = None
        if reference is not None:
            dt = stamp-reference[1]
            if dt <= 0:
                return self.reject('out_of_order')
            self.last_step_deg = shortest_step_deg(reference[0], heading)
            self.last_limit_deg = (self.jump_tolerance_deg +
                                   self.max_rate_dps*min(dt, self.max_gap_s))
            if self.last_step_deg > self.last_limit_deg:
                if not self.ready:
                    self.seed = (heading, stamp, 1)
                return self.reject('jump')
        if not self.ready:
            count = reference[2]+1 if reference is not None else 1
            self.seed = (heading, stamp, count)
            if count < self.bootstrap_samples:
                return self.reject('warming_up')
        self.heading, self.stamp = heading, stamp
        self.ready = True
        self.seed = None
        self.counts['accepted'] += 1
        self.last_reason = 'accepted'
        return True

    def is_fresh(self, now):
        return bool(self.ready and finite(now) and
                    -self.future_tolerance_s <= now-self.stamp <= self.max_age_s)

    def snapshot(self, now):
        age = now-self.stamp if self.stamp is not None and finite(now) else None
        return dict(ready=self.ready, fresh=self.is_fresh(now), counts=dict(self.counts),
                    last_reason=self.last_reason, last_step_deg=self.last_step_deg,
                    last_limit_deg=self.last_limit_deg, accepted_heading=self.heading,
                    accepted_stamp=self.stamp, accepted_age_s=age)
