"""ROS-free contracts. Coordinates: x forward, y left, radians."""
import math

def distance(a, b):
    return math.hypot(a[0]-b[0], a[1]-b[1])

def wrap(a):
    return math.atan2(math.sin(a), math.cos(a))

def transform(point, pose):
    x, y, yaw = pose
    c, s = math.cos(yaw), math.sin(yaw)
    return (x+c*point[0]-s*point[1], y+s*point[0]+c*point[1])

def fresh(stamp, now, budget):
    return stamp is not None and 0 <= now-stamp <= budget

class Scene(object):
    def __init__(self, cones, pose, stamp, version=1, free=None,
                 obstacles=None, feedback=None, speed_estimate=None):
        self.cones = cones
        self.pose = pose
        self.stamp = stamp
        self.version = version
        # free(center, radius) certifies the WHOLE disk, not only its center.
        self.free = free
        self.obstacles = obstacles or []  # x,y,radius in map coordinates
        self.feedback = feedback  # speed, measured steering, source stamp
        self.speed_estimate = speed_estimate # Explicit estimate; never hardware feedback.

class LocalMap(object):
    """Snapshot map; requires a pose at the measurement stamp, no latest-pose fallback."""
    def __init__(self):
        self.scene = None
        self.last_stamp = None

    def update(self, cones, pose, stamp, pose_stamp, version=1, **kwargs):
        if abs(stamp-pose_stamp) > 0.001:
            raise ValueError('pose_time_mismatch')
        if self.last_stamp is not None and stamp <= self.last_stamp:
            raise ValueError('non_monotonic_observation')
        if not all((not math.isnan(v) and not math.isinf(v)) for p in cones for v in p):
            raise ValueError('nonfinite_cone')
        self.last_stamp = stamp
        self.scene = Scene([transform(p, pose) for p in cones], pose,
                           stamp, version, **kwargs)
        return self.scene
