#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""ROS-independent geometry. Project coordinates: x forward, y right.

Motion poses below are short-term dead reckoning, never GPS lateral positions.
All times are sensor/ROS time in seconds. Python 2.7 compatible.
"""
from __future__ import division
import math
from collections import deque


def finite(*values):
    return all(not math.isnan(v) and not math.isinf(v) for v in values)


def wrap(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


def fresh(stamp, now, timeout):
    return finite(stamp, now) and stamp > 0 and 0 <= now - stamp <= timeout


def transform_point(point, source, target):
    x, y = point
    c, s = math.cos(source[2]), math.sin(source[2])
    wx, wy = source[0] + c*x - s*y, source[1] + s*x + c*y
    dx, dy = wx-target[0], wy-target[1]
    c, s = math.cos(target[2]), math.sin(target[2])
    return c*dx+s*dy, -s*dx+c*dy


def transform_line(line, source, target):
    """Line is (forward tangent angle, signed left-normal offset)."""
    theta, offset = line
    point = (-math.sin(theta)*offset, math.cos(theta)*offset)
    x, y = transform_point(point, source, target)
    angle = wrap(theta + source[2] - target[2])
    return angle, -math.sin(angle)*x + math.cos(angle)*y


class MotionHistory(object):
    def __init__(self, max_gap=0.35):
        self.samples = deque(maxlen=400)
        self.max_gap = max_gap

    def add(self, stamp, yaw, speed):
        if not finite(stamp, yaw, speed) or stamp <= 0 or speed < 0:
            return False
        if self.samples and stamp <= self.samples[-1][0]:
            return False
        x = y = 0.0
        if self.samples:
            old = self.samples[-1]
            dt = stamp-old[0]
            if dt > self.max_gap:
                self.samples.clear()
            else:
                heading = old[3] + wrap(yaw-old[3])*0.5
                distance = 0.5*(old[4]+speed)*dt
                x = old[1] + distance*math.cos(heading)
                y = old[2] + distance*math.sin(heading)
        self.samples.append((stamp, x, y, yaw, speed))
        return True

    def at(self, stamp):
        if not self.samples or stamp < self.samples[0][0] or stamp > self.samples[-1][0]:
            return None
        for row in self.samples:
            if row[0] == stamp:
                return row[1:4]
        for a, b in zip(self.samples, list(self.samples)[1:]):
            if a[0] < stamp < b[0]:
                w = (stamp-a[0])/(b[0]-a[0])
                return (a[1]+w*(b[1]-a[1]), a[2]+w*(b[2]-a[2]),
                        wrap(a[3]+w*wrap(b[3]-a[3])))
        return None


class RoadTracker(object):
    def __init__(self, width=3.0, min_width=2.0, max_width=4.0,
                 pair_dx=2.5, timeout=0.35, init_frames=3,
                 angle_gate=0.35, offset_gate=0.5, residual=0.25):
        self.width, self.min_width, self.max_width = width, min_width, max_width
        self.pair_dx, self.timeout = pair_dx, timeout
        self.init_frames = init_frames
        self.angle_gate, self.offset_gate, self.residual = angle_gate, offset_gate, residual
        self.reset()

    def reset(self):
        """Clear a pre-start reference; active-run faults require node restart."""
        self.line = self.pose = self.reference = None
        self.stamp = self.last_input = 0.0
        self.support = 0
        self.ready = False
        self.kind = 'waiting'
        self.pairs = 0
        self.dual_stamp = 0.0

    def reject(self):
        if not self.ready:
            self.support = 0
            self.line = None
        return False

    def current(self, stamp, pose):
        if self.line is None or not fresh(self.stamp, stamp, self.timeout):
            return None
        if not fresh(self.dual_stamp, stamp, self.timeout):
            return None
        return transform_line(self.line, self.pose, pose)

    def update(self, points, stamp, pose):
        if stamp <= self.last_input or not finite(stamp) or stamp <= 0:
            return False
        self.last_input = stamp
        prior = self.current(stamp, pose)
        if self.ready and prior is None:
            # Reacquiring a different road after expiry requires a new run.
            return False
        if not self.ready and prior is None:
            self.support = 0
        points = [p for p in points if finite(*p)]
        candidates = []
        for i, a in enumerate(points):
            for j in range(i+1, len(points)):
                b = points[j]
                dx, dy = b[0]-a[0], b[1]-a[1]
                if dy < 0:
                    dx, dy = -dx, -dy
                distance = math.hypot(dx, dy)
                if not self.min_width <= distance <= self.max_width:
                    continue
                theta = math.atan2(-dx, dy)
                seed = prior[0] if prior else theta
                # Before a reference exists, require a plausible cross-track row.
                if prior is None and (abs(dx) > self.pair_dx or abs(theta) > math.pi/4):
                    continue
                along = abs(math.cos(seed)*dx + math.sin(seed)*dy)
                if along > self.pair_dx:
                    continue
                center = ((a[0]+b[0])*0.5, (a[1]+b[1])*0.5)
                offset = -math.sin(seed)*center[0]+math.cos(seed)*center[1]
                if prior and abs(offset-prior[1]) > self.offset_gate:
                    continue
                if prior and abs(wrap(theta-prior[0])) > self.angle_gate:
                    continue
                score = abs(distance-self.width) + along
                candidates.append((score, i, j, theta, center))
        candidates.sort()
        selected, used = [], set()
        for item in candidates:
            if item[1] not in used and item[2] not in used:
                selected.append(item)
                used.update(item[1:3])
        if not selected:
            # A single boundary is only usable with a fresh identified prior.
            if not self.ready or prior is None or len(points) < 2:
                return self.reject()
            n = (-math.sin(prior[0]), math.cos(prior[0]))
            offsets = [n[0]*p[0]+n[1]*p[1] for p in points]
            side = 1 if sum(offsets)/len(offsets) > prior[1] else -1
            expected = prior[1]+side*self.width/2
            if any(abs(d-expected) > self.residual for d in offsets):
                return self.reject()
            # Single-side updates cannot renew the full-observation lifetime.
            offset = sum(offsets)/len(offsets)-side*self.width/2
            self.line, self.pose, self.stamp = (prior[0], offset), pose, stamp
            self.kind, self.pairs = 'single_side', 0
            return True
        theta = prior[0] if prior else selected[0][3]
        if not self.ready and prior:
            theta = wrap(prior[0] + wrap(selected[0][3]-prior[0])/(self.support+1))
        # Two separated midpoints supply a direction independent of car yaw.
        centers = [s[4] for s in selected]
        ordered = sorted(centers)
        direction_observed = False
        if len(ordered) >= 2:
            dx, dy = ordered[-1][0]-ordered[0][0], ordered[-1][1]-ordered[0][1]
            if math.hypot(dx, dy) >= 1.0:
                theta = math.atan2(dy, dx)
                direction_observed = True
        if prior and abs(wrap(theta-prior[0])) > self.angle_gate:
            return self.reject()
        n = (-math.sin(theta), math.cos(theta))
        # Shared direction, independently fitted boundary offsets.
        left, right = [], []
        for item in selected:
            offsets = sorted(n[0]*points[k][0]+n[1]*points[k][1] for k in item[1:3])
            left.append(offsets[0])
            right.append(offsets[1])
        lo, hi = sum(left)/len(left), sum(right)/len(right)
        if not self.min_width <= hi-lo <= self.max_width:
            return self.reject()
        if any(abs(d-lo) > self.residual for d in left) or any(abs(d-hi) > self.residual for d in right):
            return self.reject()
        offset = (lo+hi)/2
        if prior and abs(offset-prior[1]) > self.offset_gate:
            return self.reject()
        # No averaging of coefficients expressed in different body frames.
        self.line, self.pose, self.stamp = (theta, offset), pose, stamp
        self.dual_stamp = stamp
        self.pairs = len(selected)
        self.support += 1
        self.kind = 'two_rows' if direction_observed else 'one_row'
        if self.support >= self.init_frames:
            self.ready = True
            if self.reference is None:
                self.reference = wrap(theta+pose[2])
        return True

    def target(self, stamp, pose, lookahead):
        line = self.current(stamp, pose)
        if not self.ready or line is None or abs(line[0]) >= math.pi/3:
            return None
        theta, offset = line
        y = (offset + math.sin(theta)*lookahead)/math.cos(theta)
        # Same local target clamp as the original local-centerline controller.
        return lookahead, max(-1.0, min(1.0, y))


class Progress(object):
    def __init__(self, distance, max_speed, timeout=0.5):
        self.distance, self.max_speed, self.timeout = distance, max_speed, timeout
        self.origin = self.last = None
        self.axis = None
        self.value = 0.0
        self.finished = False

    def update(self, stamp, x, y, now, axis=None):
        if not fresh(stamp, now, self.timeout) or not finite(x, y):
            return False
        if self.last:
            dt = stamp-self.last[0]
            if dt <= 0:
                return False
            if math.hypot(x-self.last[1], y-self.last[2]) > self.max_speed*dt + 0.3:
                return False
        if self.origin is None:
            if axis is None:
                return False
            self.origin, self.axis = (x, y), axis
        self.last = (stamp, x, y)
        self.value = ((x-self.origin[0])*math.cos(self.axis) +
                      (y-self.origin[1])*math.sin(self.axis))
        self.finished = self.finished or self.value >= self.distance
        return True
