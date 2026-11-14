#!/usr/bin/env python
"""GPS continuity gate, independent of ROS and serial framing."""
from __future__ import division
import math
import numbers


def finite(value):
    return isinstance(value, numbers.Real) and not isinstance(value, bool) and not (
        math.isnan(value) or math.isinf(value))


def distance_m(lat1, lon1, lat2, lon2):
    lat1, lat2 = math.radians(lat1), math.radians(lat2)
    dlat, dlon = lat2-lat1, math.radians(lon2-lon1)
    a = math.sin(dlat/2.0)**2 + math.cos(lat1)*math.cos(lat2)*math.sin(dlon/2.0)**2
    return 12742000.0*math.asin(math.sqrt(max(0.0, min(1.0, a))))


class GpsIntegrityGuard(object):
    def __init__(self, max_speed_mps=8.0, jump_tolerance_m=2.0,
                 max_gap_s=1.0, max_age_s=0.5, future_tolerance_s=0.1,
                 bootstrap_samples=3, lat_min=30.0, lat_max=40.0,
                 lon_min=110.0, lon_max=125.0):
        positive = (max_speed_mps, jump_tolerance_m, max_gap_s, max_age_s)
        if any(not finite(value) or value <= 0 for value in positive):
            raise ValueError('GPS continuity limits must be finite and positive')
        if not finite(future_tolerance_s) or future_tolerance_s < 0:
            raise ValueError('GPS future tolerance must be finite and nonnegative')
        if (isinstance(bootstrap_samples, bool) or
                not isinstance(bootstrap_samples, numbers.Integral) or bootstrap_samples < 2):
            raise ValueError('GPS bootstrap requires at least two samples')
        if (not all(finite(value) for value in (lat_min, lat_max, lon_min, lon_max)) or
                not -90 <= lat_min < lat_max <= 90 or not -180 <= lon_min < lon_max <= 180):
            raise ValueError('Invalid GPS geographic bounds')
        self.max_speed_mps, self.jump_tolerance_m = max_speed_mps, jump_tolerance_m
        self.max_gap_s, self.max_age_s = max_gap_s, max_age_s
        self.future_tolerance_s, self.bootstrap_samples = future_tolerance_s, bootstrap_samples
        self.lat_min, self.lat_max, self.lon_min, self.lon_max = lat_min, lat_max, lon_min, lon_max
        self.latitude, self.longitude, self.stamp = None, None, None
        self.ready = False
        self.seed = None
        self.counts = dict(received=0, accepted=0, rejected=0)
        self.last_reason = 'waiting'
        self.last_distance_m, self.last_limit_m = None, None

    def reject(self, reason, reset_seed=False):
        self.counts['rejected'] += 1
        self.counts[reason] = self.counts.get(reason, 0)+1
        self.last_reason = reason
        if reset_seed and not self.ready:
            self.seed = None
        return False

    def accept(self, latitude, longitude, stamp, now):
        self.counts['received'] += 1
        self.last_distance_m, self.last_limit_m = None, None
        if not all(finite(value) for value in (latitude, longitude)):
            return self.reject('nonfinite', reset_seed=True)
        if not (self.lat_min <= latitude <= self.lat_max and
                self.lon_min <= longitude <= self.lon_max):
            return self.reject('bounds', reset_seed=True)
        if not finite(stamp) or not finite(now) or stamp <= 0 or now <= 0:
            return self.reject('invalid_stamp', reset_seed=True)
        age = now-stamp
        if age > self.max_age_s:
            return self.reject('stale', reset_seed=True)
        if age < -self.future_tolerance_s:
            return self.reject('future', reset_seed=True)
        if self.ready:
            reference = (self.latitude, self.longitude, self.stamp, self.bootstrap_samples)
        else:
            reference = self.seed
            if reference is not None and stamp-reference[2] > self.max_gap_s:
                reference = None
                self.seed = None
        if reference is not None:
            dt = stamp-reference[2]
            if dt <= 0:
                return self.reject('out_of_order')
            self.last_distance_m = distance_m(reference[0], reference[1], latitude, longitude)
            self.last_limit_m = self.jump_tolerance_m+self.max_speed_mps*min(dt, self.max_gap_s)
            if self.last_distance_m > self.last_limit_m:
                # Bad startup fixes can be replaced; accepted anchors cannot.
                if not self.ready:
                    self.seed = (latitude, longitude, stamp, 1)
                return self.reject('jump')
        if not self.ready:
            count = reference[3]+1 if reference is not None else 1
            self.seed = (latitude, longitude, stamp, count)
            if count < self.bootstrap_samples:
                return self.reject('warming_up')
        self.latitude, self.longitude, self.stamp = latitude, longitude, stamp
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
                    last_reason=self.last_reason, last_distance_m=self.last_distance_m,
                    last_limit_m=self.last_limit_m, accepted_stamp=self.stamp,
                    accepted_age_s=age, accepted_lat=self.latitude, accepted_lon=self.longitude)
