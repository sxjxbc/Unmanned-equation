#!/usr/bin/env python
# -*- coding: utf-8 -*-
import rospy
from msgs.msg import ConeDetection, ConeArray, Cone

class Detection2Array:
    def __init__(self):
        rospy.init_node("cone_detection_to_array_node")
        # 发布ConeArray
        self.array_pub = rospy.Publisher("/visual_cone_array", ConeArray, queue_size=10)
        # 订阅你的视觉ConeDetection话题
        rospy.Subscriber("/perception/cones", ConeDetection, self.callback)
        rospy.spin()

    def callback(self, det_msg):
        # 新建ConeArray消息
        array_msg = ConeArray()
        array_msg.header = det_msg.header  # 时间戳、frame_id直接继承
        # 把单个Cone塞进数组
        single_cone = Cone()
        single_cone.position = det_msg.position
        single_cone.color = det_msg.color
        single_cone.confidence = det_msg.confidence
        array_msg.cones.append(single_cone)
        # 发布
        self.array_pub.publish(array_msg)

if __name__ == "__main__":
    try:
        Detection2Array()
    except rospy.ROSInterruptException:
        pass
