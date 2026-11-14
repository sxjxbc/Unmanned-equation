#!/usr/bin/env python
"""Explicit opt-in output endpoint; launch defaults disabled.

An authorized deployment wrapper may enable vehicle output at next startup.
"""
import os,sys,json,threading
import rospy
from std_msgs.msg import String
from obstacle_avoidance_event.low_speed_output import LowSpeedOutputGuard


def original_client(mode):
    import rospkg
    packages=rospkg.RosPack()
    sys.path.insert(0,os.path.join(packages.get_path('acceleration_event'),'src'))
    sys.path.insert(0,os.path.join(packages.get_path('ebs_test'),'scripts'))
    from genzong_realtime import CANUDPClient
    from ebs_output import exclusive_client_class
    destination=rospy.get_param('~can_dest_ip','127.0.0.1')
    if mode=='loopback' and destination!='127.0.0.1':raise ValueError('loopback_destination_required')
    if mode=='vehicle' and rospy.get_param('~operator_ack','')!='confirmed_low_speed_test':
        raise ValueError('operator_ack_required')
    if mode=='vehicle' and (not rospy.get_param('~external_speed_limited',False) or destination=='127.0.0.1'):
        raise ValueError('explicit_vehicle_destination_and_external_speed_limit_required')
    client=exclusive_client_class(CANUDPClient)(destination,rospy.get_param('~can_dest_port',20005),
                                              rospy.get_param('~can_self_port',11312))
    client.init();return client


class OutputNode(object):
    def __init__(self):
        self.mode=rospy.get_param('~output_mode','disabled')
        if self.mode not in ('disabled','loopback','vehicle'):raise ValueError('invalid_output_mode')
        self.lock=threading.Lock();client=None
        if self.mode!='disabled':
            # Check before binding, then recheck every timer cycle.
            if self.conflict():raise ValueError('existing_control_publisher')
            client=original_client(self.mode)
        self.guard=LowSpeedOutputGuard(client,enabled=self.mode!='disabled',
            max_age=rospy.get_param('~candidate_timeout',.5),
            max_speed=rospy.get_param('~max_estimated_speed',1.),
            distance_limit=rospy.get_param('~test_distance',3.),
            duration_limit=rospy.get_param('~test_duration',15.))
        self.status=rospy.Publisher('~status',String,queue_size=1)
        rospy.Subscriber(rospy.get_param('~candidate_topic','/trackdrive_planning_shadow/candidate_control'),String,self.on_candidate,queue_size=1)
        rospy.on_shutdown(self.shutdown)
        rospy.Timer(rospy.Duration(.05),self.tick)

    def conflict(self):
        try:
            code,msg,state=rospy.get_master().getSystemState()
            if code!=1:return True
            return any(topic in ('/control/steering','/fssim/cmd') and any(n!=rospy.get_name() for n in names) for topic,names in state[0])
        except Exception:return True

    def on_candidate(self,message):
        with self.lock:
            try:packet=json.loads(message.data)
            except (ValueError,TypeError):packet=None
            self.guard.receive(packet,rospy.Time.now().to_sec())

    def tick(self,event):
        with self.lock:result=self.guard.tick(rospy.Time.now().to_sec(),channel_conflict=self.mode!='disabled' and self.conflict())
        result.update(output_mode=self.mode,vehicle_output=self.mode=='vehicle')
        self.status.publish(String(data=json.dumps(result,allow_nan=False)))

    def shutdown(self):
        with self.lock:self.guard.shutdown()


if __name__=='__main__':
    rospy.init_node('trackdrive_low_speed_output')
    OutputNode();rospy.spin()
