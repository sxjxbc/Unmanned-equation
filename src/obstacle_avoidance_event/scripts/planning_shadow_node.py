#!/usr/bin/env python
"""ROS compatibility planning node, private candidates only. No vehicle I/O."""
import json
import math
import threading
import rospy
from geometry_msgs.msg import PoseArray,PoseStamped
from nav_msgs.msg import Path
from std_msgs.msg import String
from obstacle_avoidance_event.planning_bridge import PlanningShadowBridge,configured_ingress


class PlanningShadowNode(object):
    def __init__(self):
        if rospy.get_param('~vehicle_output',False):raise ValueError('vehicle_output_not_supported')
        coordinates=rospy.get_param('~coordinates',{})
        error=None;ingress=None
        try:ingress=configured_ingress(coordinates)
        except (ValueError,KeyError,TypeError) as exc:error=str(exc)
        self.bridge=PlanningShadowBridge(ingress,rospy.get_param('~vehicle',None),configuration_error=error,
                                         steering_sign=rospy.get_param('~execution_steering_sign',None),
                                         boundary_mode=rospy.get_param('~boundary_mode','inferred_shadow'),
                                         trial_permission=rospy.get_param('~trial_permission',False),
                                         ready_frames=rospy.get_param('~ready_frames',10),
                                         candidate_hold=rospy.get_param('~candidate_hold',.2))
        self.lock=threading.Lock()
        self.status=rospy.Publisher('~status',String,queue_size=1)
        self.control=rospy.Publisher('~candidate_control',String,queue_size=1)
        self.path=rospy.Publisher('~candidate_path',Path,queue_size=1)
        rospy.Subscriber(rospy.get_param('~pose_topic','/vehicle_pose'),PoseStamped,self.on_pose,queue_size=1)
        rospy.Subscriber(rospy.get_param('~cones_topic','/clustered_points'),PoseArray,self.on_cones,queue_size=1)
        rospy.Subscriber(rospy.get_param('~boundary_packet_topic','/trackdrive/input/confirmed_corridor'),String,self.on_boundary,queue_size=1)
        rospy.Timer(rospy.Duration(.05),self.tick)

    def on_pose(self,message):
        with self.lock:self.bridge.add_pose(message)

    def on_cones(self,message):
        with self.lock:self.bridge.add_cones(message)

    def on_boundary(self,message):
        with self.lock:
            try:packet=json.loads(message.data)
            except (ValueError,TypeError):packet={}
            self.bridge.add_boundary(packet)

    def tick(self,event):
        now=rospy.Time.now().to_sec()
        with self.lock:result=self.bridge.tick(now)
        execution=result['execution']
        valid=result.get('candidate_valid',False) and execution['action']=='steer'
        status=dict((k,v) for k,v in result.items() if k not in ('planning','execution'))
        status.update(vehicle_output=False,execution_action=execution['action'],
                      scope='local_path_planning_only_not_lap_or_AS_completion')
        self.status.publish(String(data=json.dumps(status,allow_nan=False)))
        self.control.publish(String(data=json.dumps(dict(valid=bool(valid),vehicle_output=False,
            steering=execution.get('steering'),speed_limit=result.get('speed_limit',0.),
            stop_requested=execution['action']=='stop',request_recorded=execution.get('request_recorded',False),
            reason=result.get('reason'),actual_steering=None,actual_speed=None,
            drive_eligible=result.get('drive_eligible',False),source_stamp=result.get('source_stamp'),
            rear_position=result.get('rear_position'),estimated_speed=result.get('output_speed_estimate',result.get('estimated_speed'))),allow_nan=False)))
        path=Path();path.header.stamp=rospy.Time.now()
        path.header.frame_id=self.bridge.ingress.map_frame if self.bridge.ingress else 'trackdrive_map'
        if valid:
            points=result.get('planning',{}).get('path',[])
            for i,point in enumerate(points):
                pose=PoseStamped();pose.header=path.header
                pose.pose.position.x,pose.pose.position.y=point
                a,b=points[max(0,i-1)],points[min(len(points)-1,i+1)]
                yaw=math.atan2(b[1]-a[1],b[0]-a[0])
                pose.pose.orientation.z=math.sin(yaw/2.);pose.pose.orientation.w=math.cos(yaw/2.)
                path.poses.append(pose)
        self.path.publish(path)


if __name__=='__main__':
    rospy.init_node('trackdrive_planning_shadow')
    PlanningShadowNode()
    rospy.spin()
