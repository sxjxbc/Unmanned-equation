#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Passive ROS recorder. Never publishes vehicle commands or opens CAN."""
from __future__ import division
import io
import json
import math
import os
import threading
from datetime import datetime
import rospy
from geometry_msgs.msg import PoseArray, PoseStamped
from nav_msgs.msg import Path
from sensor_msgs.msg import Imu, NavSatFix
from std_msgs.msg import Bool, Float64, String
from mapping.msg import ConeArray
from run_diagnostics_core import RunDiagnostics, finite


class Monitor(object):
    def __init__(self):
        self.lock = threading.RLock()
        self.core = RunDiagnostics(rospy.get_param('~mode', 'legacy'),
                                   float(rospy.get_param('~timeout', 0.7)))
        self.core.active = bool(rospy.get_param('~run_active', False))
        root = rospy.get_param('~log_dir', '/tmp/run_diagnostics')
        self.directory = os.path.join(root, datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
        os.makedirs(self.directory)
        self.stream = io.open(os.path.join(self.directory, 'evidence.jsonl'), 'w', encoding='utf-8')
        self.counts, self.first, self.last = {}, {}, {}
        self.previous = set()
        self.origin = None
        self.closed = False
        self.publisher = rospy.Publisher('/diagnostics/run_status', String, queue_size=5)
        specs = [('gps', '/GPS_data', NavSatFix), ('imu', '/imu_data', Imu),
                 ('pose', '/vehicle_pose', PoseStamped), ('lidar', '/clustered_points', PoseArray),
                 ('map', '/cone_map', ConeArray), ('path', '/planned_path', Path),
                 ('steering', '/control/steering', Float64), ('status', '/control/lidar_status', String),
                 ('legacy_status', '/control/legacy_status', String)]
        self.subs = [rospy.Subscriber(topic, kind, self.callback, callback_args=name, queue_size=50)
                     for name, topic, kind in specs]
        self.subs.append(rospy.Subscriber('/diagnostics/run_active', Bool, self.arm))
        self.timer = rospy.Timer(rospy.Duration(0.2), self.tick, reset=True)
        rospy.on_shutdown(self.close)
        rospy.loginfo('Run diagnostics: %s | active=%s', self.directory, self.core.active)

    def write(self, value):
        self.stream.write(json.dumps(value, ensure_ascii=True, allow_nan=False)+u'\n')

    def arm(self, msg):
        with self.lock:
            self.core.active = msg.data
            self.core.active_since = None

    def callback(self, msg, name):
        with self.lock:
            if self.closed:
                return
            now = rospy.Time.now().to_sec()
            stamp = msg.header.stamp.to_sec() if hasattr(msg, 'header') else None
            try:
                if name == 'gps':
                    lat, lon = msg.latitude, msg.longitude
                    if not finite(lat) or not finite(lon) or not (-90 <= lat <= 90 and -180 <= lon <= 180):
                        raise ValueError('invalid coordinates')
                    if self.origin is None:
                        self.origin = (lat, lon)
                    value = ((lon-self.origin[1])*math.pi/180*6371000*math.cos(math.radians(self.origin[0])),
                             (lat-self.origin[0])*math.pi/180*6371000)
                    detail = dict(latitude=lat, longitude=lon, fix_status=msg.status.status)
                elif name == 'imu':
                    value = msg.orientation.z  # Project driver: heading in degrees, not quaternion.
                    detail = dict(heading_deg=value)
                elif name == 'pose':
                    value = (msg.pose.position.x, msg.pose.position.y)
                    q = msg.pose.orientation
                    detail = dict(x=value[0], y=value[1], quaternion=[q.x, q.y, q.z, q.w])
                elif name in ('status', 'legacy_status'):
                    value = json.loads(msg.data)
                    if not isinstance(value, dict):
                        raise ValueError('status must be object')
                    detail = value
                elif name == 'steering':
                    value = msg.data
                    detail = dict(command_rad=value)
                else:
                    points = msg.cones if name == 'map' else msg.poses
                    value = len(points)
                    detail = dict(count=value, frame=msg.header.frame_id)
                    if name == 'map':
                        detail['points'] = [[p.x, p.y, p.confidence] for p in points]
                    elif name == 'path':
                        detail['points'] = [[p.pose.position.x, p.pose.position.y] for p in points]
                    else:
                        detail['points'] = [[p.position.x, p.position.y, p.position.z] for p in points]
                # Serialization validates all numerical evidence, including points and status.
                json.dumps(detail, allow_nan=False)
                self.core.ingest(name, now, value, stamp)
                self.write(dict(type='input', source=name, receipt=now, stamp=stamp, data=detail))
            except (ValueError, TypeError, OverflowError) as error:
                self.core.invalid[name] = now
                self.write(dict(type='invalid', source=name, receipt=now, error=str(error)))

    def tick(self, event):
        with self.lock:
            if self.closed:
                return
            now = rospy.Time.now().to_sec()
            findings = self.core.evaluate(now)
            codes = set(f['code'] for f in findings)
            for finding in findings:
                code = finding['code']
                if code not in self.previous:
                    self.counts[code] = self.counts.get(code, 0)+1
                    self.first.setdefault(code, now)
                    rospy.logwarn('RUN DIAGNOSTIC %s: %s', code, finding['evidence'])
                self.last[code] = finding
            self.previous = codes
            status = dict(type='evaluation', stamp=now, active=self.core.active, findings=findings)
            self.write(status)
            self.stream.flush()
            self.publisher.publish(String(json.dumps(status, allow_nan=False)))
            self.report()

    def report(self):
        report = dict(mode=self.core.mode, episodes=self.counts, first_seen=self.first,
                      latest_evidence=self.last,
                      limitations=['Static position requires confirmed physical motion.',
                                   'ROS steering is a command, not measured wheel angle.',
                                   'UDP success does not prove VCU reception or actuator movement.',
                                   'No automatic stop or vehicle control is performed.'])
        target = os.path.join(self.directory, 'report.json')
        with io.open(target, 'w', encoding='utf-8') as output:
            output.write(json.dumps(report, indent=2, ensure_ascii=True)+u'\n')

    def close(self):
        with self.lock:
            if not self.closed:
                self.closed = True
                self.report()
                self.stream.close()


if __name__ == '__main__':
    rospy.init_node('run_diagnostics')
    monitor = Monitor()
    rospy.spin()
