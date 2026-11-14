#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Fixed track coordinates, independent of ROS and vehicle heading changes."""
from __future__ import division
import math


def finite(values):
    return all(not (math.isnan(v) or math.isinf(v)) for v in values)


def median(values):
    values = sorted(values)
    n = len(values)
    return (values[n//2] if n % 2 else (values[n//2-1]+values[n//2])/2.0)


class TrackFrame(object):
    def __init__(self, x, y, yaw):
        self.x, self.y, self.yaw = x, y, yaw

    def local(self, x, y):
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        dx, dy = x-self.x, y-self.y
        return c*dx+s*dy, -s*dx+c*dy

    def world(self, x, y):
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        return self.x+c*x-s*y, self.y+s*x+c*y


def paired_centers(cones, frame, vehicle, config):
    """Select independent rows in fixed coordinates; no left/right sign assumption."""
    vs, vd = frame.local(vehicle[0], vehicle[1])
    local = []
    for wx, wy, confidence in cones:
        if not finite((wx, wy, confidence)):
            continue
        x, y = frame.local(wx, wy)
        if (config.track_forward_min <= x-vs <= config.track_forward_max and
                abs(y-vd) <= 6.0):
            local.append((x, y))
    local.sort(key=lambda p: abs(p[0]-vs))
    local = local[:config.max_local_cones]
    candidates = []
    for i, (x1, y1) in enumerate(local):
        for j in range(i+1, len(local)):
            x2, y2 = local[j]
            dx, width = abs(x1-x2), math.hypot(x1-x2, y1-y2)
            if (dx <= config.pair_max_longitudinal_diff and
                    config.pair_distance_min <= width <= config.pair_distance_max):
                cx, cy = (x1+x2)/2.0, (y1+y2)/2.0
                score = abs(width-config.track_width)+.35*dx+.01*abs(cx-vs)
                candidates.append((score, cx, cy, i, j, width))
    candidates.sort()
    chosen, used = [], set()
    for item in candidates:
        _, cx, cy, i, j, width = item
        if i in used or j in used:
            continue
        if any(abs(cx-p[0]) < config.pair_row_min_spacing for p in chosen):
            continue
        chosen.append((cx, cy, width, local[i], local[j]))
        used.update((i, j))
        if len(chosen) >= config.max_pair_count:
            break
    return sorted(chosen)


def bootstrap_frame(cones, vehicle, config):
    """One complete row suffices, as before; orient normal toward initial heading."""
    temporary = TrackFrame(vehicle[0], vehicle[1], vehicle[2])
    rows = paired_centers(cones, temporary, vehicle, config)
    if not rows:
        return None
    a, b = rows[0][3:5]
    normal = math.atan2(b[1]-a[1], b[0]-a[0])+math.pi/2.0
    if math.cos(normal) < 0:
        normal += math.pi
    return TrackFrame(vehicle[0], vehicle[1], temporary.yaw+normal)


def body_line(frame, k, b, vehicle):
    """Transform fixed d=k*s+b into body y=kb*x+bb without stale body caches."""
    vs, vd = frame.local(vehicle[0], vehicle[1])
    delta = vehicle[2]-frame.yaw
    c, s = math.cos(delta), math.sin(delta)
    denominator = c+k*s
    if abs(denominator) < 1e-6:
        return None
    return (k*c-s)/denominator, (k*vs+b-vd)/denominator
