#!/usr/bin/env python
"""ROS-independent measurement validation and timestamp alignment."""
import math
import numbers
import threading
import time


def _monotonic_clock():
    if hasattr(time, 'monotonic'):
        return time.monotonic
    # Python 2 on the original Linux environment: no extra pip dependency.
    import ctypes
    import sys
    if sys.platform == 'win32':
        return time.clock
    class Timespec(ctypes.Structure):
        _fields_ = [('seconds', ctypes.c_long), ('nanoseconds', ctypes.c_long)]
    library = ctypes.CDLL('librt.so.1', use_errno=True)
    function = library.clock_gettime
    function.argtypes = [ctypes.c_int, ctypes.POINTER(Timespec)]
    function.restype = ctypes.c_int
    def clock():
        value = Timespec()
        if function(1, ctypes.byref(value)) != 0:
            raise OSError(ctypes.get_errno(), 'clock_gettime failed')
        return value.seconds + value.nanoseconds * 1e-9
    return clock


monotonic = _monotonic_clock()


def finite(value):
    return not (math.isnan(value) or math.isinf(value))


def fresh(stamp, now, timeout):
    return (isinstance(stamp, (int, float)) and finite(stamp) and
            stamp > 0 and 0.0 <= now - stamp <= timeout)


def map_line_to_local(k, b, x, y, yaw):
    """Transform map Y=k*X+b into the current vehicle frame.

    A near-perpendicular line cannot be represented as local y=k*x+b.
    """
    c, s = math.cos(yaw), math.sin(yaw)
    denominator = c + k * s
    if abs(denominator) < 1e-6:
        return None
    return ((k * c - s) / denominator,
            (k * x + b - y) / denominator)


def local_line_to_map(k, b, x, y, yaw):
    """Transform vehicle y=k*x+b into fixed map coordinates."""
    c, s = math.cos(yaw), math.sin(yaw)
    denominator = c - k * s
    if abs(denominator) < 1e-6:
        return None
    map_k = (s + k * c) / denominator
    return map_k, y - map_k * x + b / denominator


def pair_quality(width, longitudinal_difference, first_confidence, second_confidence,
                 track_width, width_min, width_max, longitudinal_max):
    """Conservative evidence quality, not a calibrated detection probability."""
    width_tolerance = max(track_width - width_min, width_max - track_width)
    if width_tolerance <= 0 or longitudinal_max <= 0:
        return 0.0
    width_score = max(0.0, 1.0 - abs(width - track_width) / width_tolerance)
    row_score = max(0.0, 1.0 - longitudinal_difference / longitudinal_max)
    confidence = max(0.0, min(1.0, first_confidence, second_confidence))
    return confidence * width_score * row_score


def centerline_from_status(status, now, timeout):
    """Validate the planner's sole road estimate and decay original evidence age."""
    if not isinstance(status, dict) or status.get('valid') is not True or timeout <= 0:
        return None
    line = status.get('centerline')
    if not isinstance(line, dict):
        return None
    for name in ('map_k', 'map_b', 'confidence', 'observation_stamp'):
        value = line.get(name)
        if isinstance(value, bool) or not isinstance(value, numbers.Real) or not finite(value):
            return None
    pairs = line.get('pairs')
    if isinstance(pairs, bool) or not isinstance(pairs, numbers.Integral) or pairs < 0:
        return None
    source = line.get('source')
    if not ((source == 'single_side' and pairs == 0) or
            (source == 'one_pair' and pairs == 1) or
            (source == 'two_pairs' and pairs >= 2)):
        return None
    confidence = line['confidence']
    stamp = line['observation_stamp']
    status_pairs = status.get('pairs')
    status_stamp = status.get('observation_stamp')
    if (isinstance(status_pairs, bool) or not isinstance(status_pairs, numbers.Integral) or
            isinstance(status_stamp, bool) or not isinstance(status_stamp, numbers.Real)):
        return None
    if not (0.0 < confidence <= 1.0 and stamp == status_stamp and
            pairs == status_pairs and fresh(stamp, now, timeout)):
        return None
    result = dict(line)
    result['confidence'] = confidence * max(0.0, 1.0 - (now - stamp) / timeout)
    return result if result['confidence'] > 0.0 else None


class LongitudinalProgress(object):
    """Signed displacement along a fixed map axis; finish is permanently latched.

    The caller supplies accepted, fresh localization measurements. No path or
    steering state participates in establishing the reference or updating progress.
    """
    def __init__(self, distance=75.0, axis_yaw=0.0, origin=None, confirm_samples=1):
        if (isinstance(distance, bool) or not isinstance(distance, numbers.Real) or
                not finite(distance) or distance <= 0):
            raise ValueError('finish distance must be a finite positive number')
        if (isinstance(axis_yaw, bool) or not isinstance(axis_yaw, numbers.Real) or
                not finite(axis_yaw)):
            raise ValueError('finish axis yaw must be finite radians')
        if (isinstance(confirm_samples, bool) or
                not isinstance(confirm_samples, numbers.Integral) or confirm_samples < 1):
            raise ValueError('finish confirmation requires at least one measurement')
        self.distance = float(distance)
        self.axis_yaw = float(axis_yaw)
        self.axis_x, self.axis_y = math.cos(axis_yaw), math.sin(axis_yaw)
        self.origin = None
        if origin is not None:
            if (len(origin) != 2 or any(isinstance(v, bool) or
                    not isinstance(v, numbers.Real) or not finite(v) for v in origin)):
                raise ValueError('finish origin must be two finite map coordinates')
            self.origin = (float(origin[0]), float(origin[1]))
        self.reference_stamp = None
        self.last_stamp = None
        self.progress = 0.0
        self.confirm_samples = confirm_samples
        self.confirm_count = 0
        self.finished = False

    def update(self, stamp, x, y):
        if self.finished:
            return False
        if any(isinstance(v, bool) or not isinstance(v, numbers.Real) or not finite(v)
               for v in (stamp, x, y)) or stamp <= 0:
            return False
        if self.last_stamp is not None and stamp <= self.last_stamp:
            return False
        if self.origin is None:
            self.origin = (float(x), float(y))
            self.reference_stamp = stamp
        self.last_stamp = stamp
        self.progress = ((x - self.origin[0]) * self.axis_x +
                         (y - self.origin[1]) * self.axis_y)
        if self.progress >= self.distance:
            self.confirm_count += 1
        else:
            self.confirm_count = 0
        self.finished = self.confirm_count >= self.confirm_samples
        return True


class PoseBuffer(object):
    def __init__(self, max_speed=12.0, jump_margin=0.5, history_seconds=5.0):
        self.lock = threading.RLock()
        self.samples = []
        self.max_speed = max_speed
        self.jump_margin = jump_margin
        self.history_seconds = history_seconds
        self.reason = 'missing'

    def add(self, stamp, x, y, yaw, received=None, speed=None):
        values = (stamp, x, y, yaw)
        if not all(finite(float(v)) for v in values) or stamp <= 0:
            self.reason = 'invalid'
            return False
        if speed is not None and (not finite(speed) or speed < 0):
            self.reason = 'invalid_speed'
            return False
        with self.lock:
            previous = self.samples[-1] if self.samples else None
            if previous:
                dt = stamp - previous['stamp']
                if dt <= 0:
                    self.reason = 'out_of_order'
                    return False
                distance = math.hypot(x - previous['x'], y - previous['y'])
                if distance > self.jump_margin + self.max_speed * dt:
                    self.reason = 'position_jump'
                    return False
                angle = abs(math.atan2(math.sin(yaw - previous['yaw']), math.cos(yaw - previous['yaw'])))
                if angle > 0.3 + 3.0 * dt:
                    self.reason = 'heading_jump'
                    return False
                if speed is None:
                    raw = distance / dt
                    speed = 0.7 * previous['speed'] + 0.3 * raw
            if speed is None:
                speed = 0.0
            self.samples.append(dict(stamp=stamp, x=x, y=y, yaw=yaw,
                                     speed=speed, received=monotonic() if received is None else received))
            self.samples = [s for s in self.samples if stamp - s['stamp'] <= self.history_seconds]
            self.reason = 'valid'
            return True

    def latest(self, ros_now, timeout, receive_now=None):
        with self.lock:
            if not self.samples:
                return None
            sample = dict(self.samples[-1])
            age = ros_now - sample['stamp']
            arrival_age = (monotonic() if receive_now is None else receive_now) - sample['received']
            sample['age'] = age
            sample['valid'] = (self.reason == 'valid' and 0 <= age <= timeout and
                               0 <= arrival_age <= timeout)
            sample['reason'] = self.reason if self.reason != 'valid' else ('valid' if sample['valid'] else 'stale')
            return sample

    def at(self, stamp, tolerance):
        with self.lock:
            samples = list(self.samples)
        # Exact measurements need no interpolation or neighboring time tolerance.
        for sample in samples:
            if sample['stamp'] == stamp:
                return dict(sample)
        for first, second in zip(samples, samples[1:]):
            if first['stamp'] <= stamp <= second['stamp']:
                if max(stamp - first['stamp'], second['stamp'] - stamp) > tolerance:
                    return None
                ratio = (stamp - first['stamp']) / (second['stamp'] - first['stamp'])
                result = dict(first)
                for key in ('x', 'y', 'speed'):
                    result[key] += ratio * (second[key] - first[key])
                angle = math.atan2(math.sin(second['yaw'] - first['yaw']), math.cos(second['yaw'] - first['yaw']))
                result['yaw'] = math.atan2(math.sin(first['yaw'] + ratio * angle), math.cos(first['yaw'] + ratio * angle))
                result['stamp'] = stamp
                return result
        if samples:
            nearest = min(samples, key=lambda s: abs(s['stamp'] - stamp))
            if abs(nearest['stamp'] - stamp) <= tolerance:
                return dict(nearest)
        return None


def gps_quality_valid(status, covariance_type, covariance, max_variance, allow_unknown=False):
    """Check horizontal covariance (m^2); unknown is an explicit replay opt-out."""
    if isinstance(status, bool) or not isinstance(status, numbers.Integral) or status not in (0, 1, 2):
        return False
    if covariance_type == 0:
        return bool(allow_unknown)
    if covariance_type not in (1, 2, 3) or len(covariance) != 9:
        return False
    if any(not isinstance(v, numbers.Real) or isinstance(v, bool) or not finite(v) for v in covariance):
        return False
    xx, yy, xy, yx = covariance[0], covariance[4], covariance[1], covariance[3]
    if not (0 < xx <= max_variance and 0 < yy <= max_variance):
        return False
    return abs(xy - yx) <= 1e-6 and xy * xy <= xx * yy + 1e-12


class StableOrigin(object):
    """Continuous time-aligned stationary observations; yaw is in radians."""
    def __init__(self, duration=3.0, min_samples=10, max_gap=0.5,
                 position_tolerance=0.30, yaw_tolerance=math.radians(3.0)):
        if any(not finite(v) or v <= 0 for v in
               (duration, max_gap, position_tolerance, yaw_tolerance)):
            raise ValueError('Initialization tolerances must be positive and finite')
        if isinstance(min_samples, bool) or not isinstance(min_samples, numbers.Integral) or min_samples < 2:
            raise ValueError('Initialization requires at least two samples')
        self.duration, self.min_samples, self.max_gap = duration, min_samples, max_gap
        self.position_tolerance, self.yaw_tolerance = position_tolerance, yaw_tolerance
        self.samples = []

    def reset(self):
        self.samples = []

    def add(self, stamp, lat, lon, yaw):
        if not all(finite(v) for v in (stamp, lat, lon, yaw)) or stamp <= 0:
            self.reset()
            return None
        if self.samples and stamp <= self.samples[-1][0]:
            return None
        if self.samples and stamp - self.samples[-1][0] > self.max_gap:
            self.reset()
        candidate = self.samples + [(stamp, lat, lon, yaw)]
        # Bound memory at high input rates while retaining temporal coverage.
        candidate = [s for s in candidate if stamp - s[0] <= self.duration + self.max_gap]
        mean_lat = sum(s[1] for s in candidate) / len(candidate)
        mean_lon = sum(s[2] for s in candidate) / len(candidate)
        mean_yaw = math.atan2(sum(math.sin(s[3]) for s in candidate),
                              sum(math.cos(s[3]) for s in candidate))
        scale = 6371000.0 * math.pi / 180.0
        stable = all(math.hypot((s[1] - mean_lat) * scale,
                               (s[2] - mean_lon) * scale * math.cos(math.radians(mean_lat))) <= self.position_tolerance
                     and abs(math.atan2(math.sin(s[3] - mean_yaw), math.cos(s[3] - mean_yaw))) <= self.yaw_tolerance
                     for s in candidate)
        self.samples = candidate if stable else [candidate[-1]]
        if stable and len(candidate) >= self.min_samples and stamp - candidate[0][0] >= self.duration - 1e-9:
            return mean_lat, mean_lon, mean_yaw
        return None
