#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
八字绕环闭环tracking
planner -> tracking -> CAN UDP -> virtual_vcu
"""

from __future__ import print_function
import rospy, socket, math, struct
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Float64, UInt8

class Tracking(object):
    def __init__(self):
        rospy.init_node("tracking_udp")
        self.path=[]
        self.last_target_index=0
        self.speed_cmd=0.0
        self.x=rospy.get_param('~start_x', -15.0 - 0.7)
        self.y=0.0
        self.yaw=0.0
        self.finished=False
        self.sock=socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
        self.addr=("192.168.0.7",20005)
        self.pose_topic = rospy.get_param('~pose_topic', '/vehicle_pose')
        self.steer_max_rad = rospy.get_param('~steer_max_rad', math.pi/2)

        rospy.Subscriber("/planned_path",Path,self.path_cb)
        rospy.Subscriber("/target_speed",Float64,self.speed_cb)
        rospy.Subscriber(self.pose_topic, PoseStamped, self.pose_cb, queue_size=10)
        rospy.Subscriber("/race_status",UInt8,self.status_cb)
        rospy.loginfo("[Tracking] Bazi CAN controller started")

    def path_cb(self,msg):
        self.path=[(p.pose.position.x,p.pose.position.y) for p in msg.poses]
        self.last_target_index=0

    def speed_cb(self,msg):
        self.speed_cmd=float(msg.data)

    def pose_cb(self,msg):
        self.x=msg.pose.position.x
        self.y=msg.pose.position.y
        q=msg.pose.orientation
        # yaw from quaternion
        self.yaw=math.atan2(2*(q.w*q.z+q.x*q.y),1-2*(q.y*q.y+q.z*q.z))

    def status_cb(self,msg):
        if msg.data:
            self.finished=True

    def target_point(self):
        # 八字交叉区域不能全局最近点搜索，否则会跳到另一圆环导致方向突变
        if len(self.path) < 2:
            return None
        nearest=self.last_target_index
        best_d=999.0
        start=max(0, nearest-30)
        end=min(len(self.path), nearest+150)
        for i in range(start,end):
            d=math.hypot(self.path[i][0]-self.x,self.path[i][1]-self.y)
            if d < best_d:
                best_d=d
                nearest=i
        self.last_target_index=nearest
        ld = max(4.0, min(8.0, self.speed_cmd * 1.5 + 2.0))
        idx = nearest
        dist = 0.0
        while idx < len(self.path)-1 and dist < ld:
            dist += math.hypot(self.path[idx + 1][0] - self.path[idx][0],
                               self.path[idx + 1][1] - self.path[idx][1])
            idx += 1
        return self.path[idx]

    def steer(self):
        if self.finished:
            return math.pi/2
        p=self.target_point()
        if p is None:
            return 0.0
        dx=p[0]-self.x
        dy=p[1]-self.y

        #偏离路径太远，先停下来（避免打转）
        if len(self.path) > 0:
            #找最近点
            min_d =float('inf')
            for i in range(len(self.path)):
                d = math.hypot(self.path[i][0] - self.x, self.path[i][1]- self.y)
                if d < min_d:
                    min_d = d
            if min_d > 5.0:
                #偏移超过5m，禁止大角度
                rospy.logwarn_throttle(1.0, "[Tracking] off-path %.2fm, clamping steer", min_d)
                return 0.0
        # pure pursuit
        alpha=math.atan2(dy,dx)-self.yaw
        while alpha>math.pi: alpha-=2*math.pi
        while alpha<-math.pi: alpha+=2*math.pi
        steer=0.75*alpha
        # === 临时调试 ===
        rospy.loginfo_throttle(0.2,
            "[DBG] veh=(%.2f,%.2f) yaw=%.1f target=(%.2f,%.2f) "
            "alpha=%.1f steer=%.1f path_len=%d",
            self.x, self.y, math.degrees(self.yaw),
            p[0], p[1],
            math.degrees(alpha), math.degrees(steer),
            len(self.path))
        # === 结束 ===
        return max(-self.steer_max_rad, min(self.steer_max_rad, steer))

    def send_can(self):
        steer=self.steer()
        if self.finished:
            steer_raw=2000
        else:
            steer_raw=int(round(500+steer*1000/math.pi))
            steer_raw=max(0,min(1000,steer_raw))
        # rpm conversion, simple model
        rpm=int(max(0,self.speed_cmd)*60/(2*math.pi*0.3))
        rpm=max(0,min(3000,rpm))
        stop=1 if self.finished else 0
        data=struct.pack("<HHHH",steer_raw,rpm,stop, 0)
        # VCU parser expects 13 bytes, ID at bytes 3/4
        pkt=bytearray(13)
        pkt[0]=8
        pkt[3]=(0x210>>8)&0xff
        pkt[4]=0x210&0xff
        pkt[5:13]=data
        self.sock.sendto(bytes(pkt),self.addr)

    def run(self):
        r=rospy.Rate(50)
        while not rospy.is_shutdown():
            self.send_can()
            rospy.loginfo_throttle(1,"[Tracking] speed %.2f steer %.2f deg"%(self.speed_cmd,math.degrees(self.steer())))
            r.sleep()

if __name__=="__main__":
    Tracking().run()
