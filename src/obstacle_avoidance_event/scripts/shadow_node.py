#!/usr/bin/env python
"""Read-only ROS ingress. No vehicle transport implementation exists here."""
import json
import threading
import rospy
from geometry_msgs.msg import PoseArray,PoseStamped
from nav_msgs.msg import Path
from std_msgs.msg import String
from obstacle_avoidance_event.input_adapter import AlignedIngress

class ShadowNode(object):
    def __init__(self):
        if rospy.get_param('~vehicle_output',False):
            raise RuntimeError('vehicle_output_not_supported')
        self.mode=rospy.get_param('~control_mode','legacy_compatible')
        if self.mode not in ('strict','legacy_compatible'):raise ValueError('invalid_control_mode')
        self.lock = threading.Lock()
        self.pose = None
        self.cones = None
        self.ingress = AlignedIngress(
            map_frame=rospy.get_param('~map_frame','map'),
            cones_frame=rospy.get_param('~aligned_cones_frame',''),
            pose_reference=rospy.get_param('~pose_reference','unconfirmed'))
        self.pose_error = None
        self.status = rospy.Publisher('~status',String,queue_size=1)
        self.path = rospy.Publisher('~candidate_path',Path,queue_size=1)
        self.control = rospy.Publisher('~candidate_control',String,queue_size=1)
        rospy.Subscriber(rospy.get_param('~pose_topic','/vehicle_pose'),PoseStamped,self.on_pose,queue_size=1)
        rospy.Subscriber(rospy.get_param('~cones_topic','/clustered_points'),PoseArray,self.on_cones,queue_size=1)
        rospy.Timer(rospy.Duration(.2),self.tick)

    def on_pose(self,msg):
        with self.lock:
            self.pose = msg
            try:
                self.ingress.add_pose(msg)
                self.pose_error = None
            except (ValueError,TypeError,AttributeError) as exc:
                self.pose_error = str(exc)
                # A rejected pose invalidates the buffer until new samples arrive.
                self.ingress.poses.samples = []

    def on_cones(self,msg):
        with self.lock:
            self.cones = msg

    def tick(self,event):
        # Existing streams lack time-aligned pose, certified free-space and feedback.
        # Deliberately do not transform latest pose or claim safe paths.
        with self.lock:
            pose,cones=self.pose,self.cones
        now=rospy.Time.now().to_sec()
        def age(msg):
            if msg is None or msg.header.stamp.to_sec()<=0:return None
            return now-msg.header.stamp.to_sec()
        pose_age,cones_age=age(pose),age(cones)
        aligned = None
        with self.lock:
            alignment_reason = self.pose_error
            if alignment_reason is None:
                try:
                    if cones is None:raise ValueError('cones_unavailable')
                    aligned = self.ingress.align(cones,now)
                except (ValueError,TypeError,AttributeError) as exc:
                    alignment_reason = str(exc)
        waiting=[]
        if pose_age is None or not 0<=pose_age<=.8:waiting.append('pose_unavailable_or_stale')
        if cones_age is None or not 0<=cones_age<=.5:waiting.append('cones_unavailable_or_stale')
        if aligned is None:waiting.append(alignment_reason or 'time_aligned_pose_unavailable')
        waiting.extend(['certified_free_space','closed_map_and_gate'])
        if self.mode=='strict':waiting.append('hardware_feedback')
        result = dict(state='WAIT_INPUT' ,valid=False,vehicle_output=False,
                      reason='planning_adapters_unavailable',control_mode=self.mode,
                      actual_steering=None,actual_speed=None,hardware_go=None,hardware_as_state=None,
                      external_CAN_monitoring=self.mode=='legacy_compatible',
                      mission='Trackdrive',mission_adapter='res_as_ack_closed_map_gate_unavailable',
                      pose_received=pose is not None,cones_received=cones is not None,
                      pose_age_s=pose_age,cones_age_s=cones_age,
                      cones_count=len(cones.poses) if cones is not None else None,
                      pose_frame=pose.header.frame_id if pose is not None else None,
                      cones_frame=cones.header.frame_id if cones is not None else None,
                      alignment_ready=aligned is not None,alignment_reason=alignment_reason,
                      aligned_pose=aligned['pose'] if aligned is not None else None,
                      aligned_cones_count=len(aligned['cones']) if aligned is not None else None,
                      waiting_for=waiting)
        self.status.publish(String(data=json.dumps(result)))
        self.control.publish(String(data=json.dumps(dict(valid=False,steering=0,speed_limit=0,vehicle_output=False))))
        empty=Path()
        empty.header.stamp=rospy.Time.now()
        empty.header.frame_id='map'
        self.path.publish(empty)

if __name__ == '__main__':
    rospy.init_node('obstacle_avoidance_shadow')
    ShadowNode()
    rospy.spin()
