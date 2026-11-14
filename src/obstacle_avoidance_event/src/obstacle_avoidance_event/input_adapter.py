"""ROS-free, bounded pose interpolation and explicitly framed observation ingress."""
import math
from .model import wrap, transform
from .execution_adapter import finite


class PoseBuffer(object):
    def __init__(self, max_gap=.2, capacity=200):
        if not finite(max_gap) or max_gap <= 0 or capacity < 2:
            raise ValueError('invalid_pose_buffer_configuration')
        self.max_gap, self.capacity = max_gap, capacity
        self.samples = []

    def add(self, stamp, pose):
        if not finite(stamp) or stamp <= 0 or len(pose) != 3 or not all(finite(v) for v in pose):
            raise ValueError('invalid_pose_sample')
        if self.samples and stamp <= self.samples[-1][0]:
            raise ValueError('non_monotonic_pose')
        self.samples.append((stamp, tuple(pose)))
        self.samples = self.samples[-self.capacity:]

    def at(self, stamp):
        if not finite(stamp) or stamp <= 0:
            raise ValueError('invalid_observation_stamp')
        for t, pose in self.samples:
            if t == stamp:
                return pose
        for (ta, a), (tb, b) in zip(self.samples, self.samples[1:]):
            if ta < stamp < tb:
                if tb-ta > self.max_gap:
                    raise ValueError('pose_interpolation_gap')
                fraction = (stamp-ta)/(tb-ta)
                return (a[0]+fraction*(b[0]-a[0]), a[1]+fraction*(b[1]-a[1]),
                        wrap(a[2]+fraction*wrap(b[2]-a[2])))
        raise ValueError('pose_bracket_unavailable')


def planar_pose(message):
    """Accept PoseStamped quaternion only, never raw /imu_data heading degrees."""
    p, q = message.pose.position, message.pose.orientation
    values = (p.x, p.y, p.z, q.x, q.y, q.z, q.w)
    if not all(finite(v) for v in values):
        raise ValueError('nonfinite_pose_message')
    if abs(sum(v*v for v in (q.x, q.y, q.z, q.w))-1.) > .001:
        raise ValueError('invalid_pose_quaternion')
    if abs(q.x) > .001 or abs(q.y) > .001:
        raise ValueError('nonplanar_pose_unsupported')
    return (p.x, p.y, math.atan2(2*q.w*q.z, 1-2*q.z*q.z))


class AlignedIngress(object):
    """Produce alignment diagnostics, not a certified Scene or hardware feedback.

    Cones must already be in the declared rear-axle frame at their stamp.
    Raw lidar extrinsics and INS-to-rear-axle conversion belong upstream.
    """
    def __init__(self, map_frame='map', cones_frame='', pose_reference='unconfirmed',
                 max_gap=.2, max_age=.5, input_map_frame=None,
                 pose_transform=None, point_transform=None):
        self.map_frame, self.cones_frame = map_frame, cones_frame
        self.pose_reference, self.max_age = pose_reference, max_age
        self.input_map_frame=input_map_frame or map_frame
        self.pose_transform,self.point_transform=pose_transform,point_transform
        self.poses = PoseBuffer(max_gap)

    def add_pose(self, message):
        if message.header.frame_id != self.input_map_frame:
            raise ValueError('pose_frame_mismatch')
        pose=planar_pose(message)
        if self.pose_transform is not None:pose=self.pose_transform(pose)
        self.poses.add(message.header.stamp.to_sec(), pose)

    def align(self, message, now):
        if self.pose_reference != 'rear_axle':
            raise ValueError('rear_axle_reference_unconfirmed')
        if not self.cones_frame or message.header.frame_id != self.cones_frame:
            raise ValueError('cones_frame_unconfirmed_or_mismatched')
        stamp = message.header.stamp.to_sec()
        if not finite(now) or not finite(stamp) or not 0 < stamp <= now or now-stamp > self.max_age:
            raise ValueError('observation_unavailable_or_stale')
        points = [(p.position.x, p.position.y) for p in message.poses]
        if len(points) > 120 or not all(finite(v) for point in points for v in point):
            raise ValueError('invalid_cone_observation')
        if self.point_transform is not None:
            points=[self.point_transform(*p) for p in points]
        pose = self.poses.at(stamp)
        return dict(stamp=stamp, pose=pose, cones=[transform(p, pose) for p in points],
                    pose_reference='rear_axle',
                    frame=self.map_frame, valid_for_planning=False,
                    missing=['certified_free_space', 'hardware_feedback', 'closed_map_and_gate'])
