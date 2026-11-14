#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Current-frame LiDAR tracking with the project's existing PD and CAN codec.

The legacy constructor/control loop is deliberately not started. No map/path
subscription influences lateral control. Shadow mode never opens a CAN socket.
"""
from __future__ import division
import json
import math
import os
import threading
from datetime import datetime
from collections import deque
import rospy
from geometry_msgs.msg import PoseArray, PoseStamped
from sensor_msgs.msg import Imu
from std_msgs.msg import Float64, String
from genzong_realtime import PathTrackingController, CANUDPClient
from lidar_road import MotionHistory, RoadTracker, Progress, fresh, finite, wrap


class LidarTrackingController(PathTrackingController):
    def __init__(self):
        rospy.init_node('lidar_tracking_controller')
        self.lock = threading.RLock()
        p = rospy.get_param
        self.mode = p('~tracking_mode', 'shadow')
        if self.mode not in ('shadow', 'control'):
            raise ValueError('tracking_mode must be shadow or control')
        self.output_enabled = self.mode == 'control' and p('~enable_vehicle_output', False)
        # Original defaults; the launch copies every existing controller value,
        # including the last effective value of duplicate parameters.
        defaults = dict(track_width=3.0, max_vehicle_speed_mps=15.0/3.6,
                        max_steering=0.28, finish_distance=75.0,
                        steering_sign=1.0, max_steering_step=0.012,
                        kp_lateral=0.48, kd_lateral=0.05, kp_heading=0.10,
                        kd_heading=0.01, error_filter_alpha=0.12,
                        heading_weight_min=0.35, heading_weight_full_lateral_error=0.80,
                        max_lateral_derivative=0.60, max_heading_derivative=0.35,
                        lookahead_min=2.5, lookahead_max=4.0, lookahead_speed_gain=0.35,
                        local_cone_min_x=0.5, local_cone_max_x=18.0,
                        local_cone_max_abs_y=4.0, max_local_cones=4,
                        pair_distance_min=2.0, pair_distance_max=4.0,
                        pair_max_dx=2.5, local_timeout=0.35)
        for name, default in defaults.items():
            setattr(self, name, p('~'+name, default))
        self.pose_timeout = p('~fusion_pose_timeout', 0.5)
        self.imu_timeout = p('~fusion_imu_timeout', 0.15)
        self.expected_frame = p('~fusion_lidar_frame', 'velodyne').lstrip('/')
        self.motion = MotionHistory(self.local_timeout)
        self.road = RoadTracker(self.track_width, self.pair_distance_min,
                                self.pair_distance_max, self.pair_max_dx,
                                self.local_timeout,
                                p('~fusion_init_frames', 3),
                                p('~fusion_angle_gate', 0.35),
                                p('~fusion_offset_gate', 0.5),
                                p('~fusion_fit_residual', 0.25))
        self.progress = Progress(self.finish_distance, self.max_vehicle_speed_mps*1.2,
                                 self.pose_timeout)
        self.offset_x = p('~lidar_offset_x', 1.76618)
        self.offset_y = p('~lidar_offset_y', -0.11304)
        self.y_sign = p('~lidar_y_sign', -1.0)
        self.mount_yaw = math.radians(p('~lidar_yaw_offset_deg', 0.5))
        self.self_rear = p('~self_filter_rear', 1.2)
        self.self_front = p('~self_filter_front', 2.7)
        self.self_half_width = p('~self_filter_half_width', 0.9)
        self.vehicle_speed = 0.0
        self.vehicle_x = self.vehicle_y = self.vehicle_yaw = 0.0
        self.filtered_lateral_error = self.filtered_heading_error = 0.0
        self.prev_lateral_error = self.prev_heading_error = self.prev_steering_angle = 0.0
        self.pose_sample = self.pending = None
        self.pose_history = deque(maxlen=100)
        self.shutting_down = False
        self.last_detection_stamp = 0.0
        self.last_tick = None
        self.started = False
        self.stop_reason = None
        self.input_reason = 'waiting'
        self.log_stream = None
        self.last_log_flush = 0.0
        if p('~fusion_log_enabled', True):
            log_dir = p('~log_dir', '/tmp/path_tracking_logs')
            try:
                if not os.path.isdir(log_dir):
                    os.makedirs(log_dir)
                filename = 'lidar_fusion_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f') + '.jsonl'
                self.log_stream = open(os.path.join(log_dir, filename), 'a')
            except (IOError, OSError) as error:
                rospy.logwarn('LiDAR diagnostic file unavailable: %s', error)
        self.can_client = None
        self.can_client_ok = False
        if self.output_enabled:
            self.can_client = CANUDPClient()
            self.can_client_ok = self.can_client.init()
            if not self.can_client_ok:
                raise RuntimeError('CAN initialization failed')
        self.status_pub = rospy.Publisher('/control/lidar_status', String, queue_size=10)
        self.candidate_pub = rospy.Publisher('/control/lidar_candidate_steering', Float64, queue_size=10)
        self.steering_pub = rospy.Publisher('/control/steering', Float64, queue_size=10) if self.output_enabled else None
        self.subscribers = [
            rospy.Subscriber('/vehicle_pose', PoseStamped, self.pose_callback, queue_size=20),
            rospy.Subscriber('/imu_data', Imu, self.imu_callback, queue_size=100),
            rospy.Subscriber('/clustered_points', PoseArray, self.lidar_callback, queue_size=1)]
        self.control_timer = rospy.Timer(rospy.Duration(0.05), self.control_loop, reset=True)
        rospy.loginfo('LiDAR fusion: mode=%s vehicle_output=%s; original PD/CAN retained',
                      self.mode, self.output_enabled)

    def pose_callback(self, msg):
        stamp, now = msg.header.stamp.to_sec(), rospy.Time.now().to_sec()
        q = msg.pose.orientation
        x, y = msg.pose.position.x, msg.pose.position.y
        if (msg.header.frame_id.lstrip('/') != 'map' or
                not fresh(stamp, now, self.pose_timeout) or
                not finite(x, y, q.x, q.y, q.z, q.w) or
                abs(q.x*q.x+q.y*q.y+q.z*q.z+q.w*q.w-1.0) > 0.05):
            return
        yaw = math.atan2(2*(q.w*q.z+q.x*q.y), 1-2*(q.y*q.y+q.z*q.z))
        with self.lock:
            old = self.pose_sample
            if old:
                dt = stamp-old[0]
                if dt <= 0 or math.hypot(x-old[1], y-old[2]) > self.max_vehicle_speed_mps*1.2*dt+0.3:
                    return
                raw_speed = math.hypot(x-old[1], y-old[2])/dt
                self.vehicle_speed = min(self.max_vehicle_speed_mps,
                                         0.7*self.vehicle_speed+0.3*raw_speed)
            self.pose_sample = (stamp, x, y, yaw)
            self.pose_history.append(self.pose_sample)
            # Copy all control state immediately, not via a 1 Hz source timer.
            self.vehicle_x, self.vehicle_y, self.vehicle_yaw = x, y, yaw

    def imu_callback(self, msg):
        stamp, now = msg.header.stamp.to_sec(), rospy.Time.now().to_sec()
        # Existing driver contract: orientation.z is heading in DEGREES.
        yaw = math.radians(msg.orientation.z)
        if not fresh(stamp, now, self.pose_timeout) or not finite(yaw):
            return
        with self.lock:
            if not self.pose_sample or not fresh(self.pose_sample[0], now, self.pose_timeout):
                return
            if self.motion.samples and stamp-self.motion.samples[-1][0] > self.local_timeout:
                # A history reset changes its origin. Never reuse an old line.
                if self.started:
                    self.stop_reason = 'motion_gap'
                self.road.line = None
                self.road.support = 0
                if not self.started:
                    self.road.reset()
            self.motion.add(stamp, yaw, self.vehicle_speed)

    def lidar_callback(self, msg):
        stamp, now = msg.header.stamp.to_sec(), rospy.Time.now().to_sec()
        with self.lock:
            if msg.header.frame_id.lstrip('/') != self.expected_frame:
                self.input_reason = 'lidar_frame'
                return
            if not fresh(stamp, now, self.local_timeout) or stamp <= self.last_detection_stamp:
                self.input_reason = 'lidar_stamp'
                return
            self.last_detection_stamp = stamp
            c, s = math.cos(self.mount_yaw), math.sin(self.mount_yaw)
            points = []
            for pose in msg.poses:
                x, y, z = pose.position.x, pose.position.y, pose.position.z
                if not finite(x, y, z):
                    continue
                y *= self.y_sign
                x, y = c*x-s*y+self.offset_x, s*x+c*y+self.offset_y
                if -self.self_rear <= x <= self.self_front and abs(y) <= self.self_half_width:
                    continue
                if (self.local_cone_min_x <= x <= self.local_cone_max_x and
                        abs(y) <= self.local_cone_max_abs_y):
                    points.append((x, y))
            points.sort(key=lambda pt: (pt[0], abs(pt[1])))
            self.pending = (stamp, points[:self.max_local_cones])

    def process_detection(self, now):
        if self.pending is None:
            return
        stamp, points = self.pending
        if not fresh(stamp, now, self.local_timeout):
            self.pending = None
            self.input_reason = 'observation_expired'
            return
        pose = self.motion.at(stamp)
        if pose is None:
            self.input_reason = 'waiting_time_alignment'
            return
        self.pending = None
        if (not self.started and self.road.ready and
                not fresh(self.road.dual_stamp, now, self.local_timeout)):
            # Before the run starts, expired initialization can be retried.
            self.road.reset()
        accepted = self.road.update(points, stamp, pose)
        self.input_reason = 'accepted' if accepted else 'geometry_rejected'

    def control_loop(self, event):
        with self.lock:
            if self.shutting_down:
                return
            now = rospy.Time.now().to_sec()
            if self.last_tick is not None and now < self.last_tick:
                self.stop_reason = 'clock_rewind'
            dt = 0.05 if self.last_tick is None else now-self.last_tick
            self.last_tick = now
            self.process_detection(now)
            sample = self.motion.samples[-1] if self.motion.samples else None
            valid_pose = self.pose_sample and fresh(self.pose_sample[0], now, self.pose_timeout)
            valid_motion = sample and fresh(sample[0], now, self.imu_timeout)
            # Finish processing always precedes road availability checks.
            if self.started and valid_pose:
                self.progress.update(self.pose_sample[0], self.pose_sample[1],
                                     self.pose_sample[2], now)
            if self.progress.finished:
                self.stop_reason = 'finish'
            target = None
            self.update_dynamic_lookahead()
            if valid_pose and valid_motion:
                # Extrapolate only within the bounded IMU age, using forward speed.
                age = now-sample[0]
                pose = (sample[1]+self.vehicle_speed*age*math.cos(sample[3]),
                        sample[2]+self.vehicle_speed*age*math.sin(sample[3]), sample[3])
                target = self.road.target(now, pose, self.lookahead_distance)
            if target is not None and not self.started and not self.stop_reason:
                # Reference at actual controller readiness, mapped to the existing map.
                # GPS/mapping is stamped after IMU by the existing driver.
                # Use the newest fresh historical pose actually bracketed by IMU.
                for initial_pose in reversed(self.pose_history):
                    if not fresh(initial_pose[0], now, self.pose_timeout):
                        continue
                    aligned = self.motion.at(initial_pose[0])
                    if aligned is not None:
                        axis = wrap(initial_pose[3]+self.road.reference-aligned[2])
                        self.started = self.progress.update(
                            initial_pose[0], initial_pose[1], initial_pose[2], now, axis)
                        break
            if self.started and target is None and not self.stop_reason:
                self.stop_reason = ('pose_stale' if not valid_pose else
                                    'imu_stale' if not valid_motion else 'road_stale')
            steering = 0.0
            if self.started and not self.stop_reason and dt > 0:
                # Exactly the original target-point lateral/bearing semantics.
                lateral, heading = target[1], math.atan2(target[1], target[0])
                lateral, heading = self.filter_tracking_errors(lateral, heading)
                steering = self.pid_control(lateral, heading, dt)
            stop = self.stop_reason is not None
            sent = False
            if self.output_enabled and (self.started or stop):
                sent = self.can_client.send_steering_frame(steering, stop=stop)
                if not sent and self.stop_reason is None:
                    self.stop_reason = 'can_send_failed'
                    steering = 0.0
                    stop = True
                    # Retry a stop immediately; subsequent ticks retain it.
                    self.can_client.send_steering_frame(0.0, stop=True)
                self.steering_pub.publish(Float64(steering))
            self.candidate_pub.publish(Float64(steering))
            status = dict(mode=self.mode, enabled=self.output_enabled, started=self.started,
                          control_stamp=now,
                          stop_reason=self.stop_reason, input_reason=self.input_reason,
                          observation_stamp=self.road.stamp, pose_stamp=self.pose_sample[0] if self.pose_sample else None,
                          imu_stamp=sample[0] if sample else None, pairs=self.road.pairs,
                          estimate_kind=self.road.kind, progress=self.progress.value,
                          reference_yaw=self.road.reference,
                          observation_age=now-self.road.stamp if self.road.stamp else None,
                          target=list(target) if target else None, steering=steering, udp_sent=sent)
            encoded = json.dumps(status, allow_nan=False)
            self.status_pub.publish(String(encoded))
            if self.log_stream is not None:
                try:
                    self.log_stream.write(encoded+'\n')
                    if stop or now-self.last_log_flush >= 1.0:
                        self.log_stream.flush()
                        self.last_log_flush = now
                except (IOError, OSError) as error:
                    rospy.logwarn('LiDAR diagnostic file write failed: %s', error)
                    try:
                        self.log_stream.close()
                    except (IOError, OSError):
                        pass
                    self.log_stream = None

    def shutdown(self):
        with self.lock:
            if self.shutting_down:
                return
            self.shutting_down = True
            self.control_timer.shutdown()
            if self.can_client is not None:
                try:
                    self.can_client.send_steering_frame(0.0, stop=True)
                finally:
                    self.can_client.close()
            if self.log_stream is not None:
                self.log_stream.close()
                self.log_stream = None


if __name__ == '__main__':
    node = LidarTrackingController()
    rospy.on_shutdown(node.shutdown)
    rospy.spin()
