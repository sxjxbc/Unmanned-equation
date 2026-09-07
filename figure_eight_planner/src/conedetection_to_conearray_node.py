#!/usr/bin/env python
# -*- coding: utf-8 -*-
<<<<<<< HEAD
"""单锥 ConeDetection → ConeArray 时间窗累积桥接节点。

camera/yolo_camera.py 对每个检测到的锥桶单独发布一条 /perception/cones
(ConeDetection, 含 color/x/y/z)，本节点把时间窗内收到的锥桶累积成
/visual_cone_array (ConeArray) 供 figure_eight_planner 做激光+视觉融合。

【修复记录 2026-09-03】
1. 字段错配：原代码 single_cone.position = det_msg.position —— ConeDetection
   的字段是 x/y/z，Cone 的字段是 x/y/confidence，都没有 position，必然抛
   AttributeError，导致 /visual_cone_array 从不发布。
2. 单锥 vs 数组语义：每条 ConeDetection 只含一个锥，若原样转发成 1 元素
   数组，下游 latest_visual_msg 只会保留最后一条消息，前面锥桶全部丢失；
   现改为 0.5s（可配 ~window_secs）时间窗累积后整体发布。
3. confidence 置 1.0：ConeDetection 无置信度字段，而 Cone.confidence 默认
   0.0 会被 figure_eight 的 cone_confidence_threshold(0.15) 全部过滤掉。
"""
import rospy
from msgs.msg import ConeDetection, ConeArray, Cone


class Detection2Array(object):
    def __init__(self):
        rospy.init_node("cone_detection_to_array_node")
        # 累积窗口(秒)：窗口内的锥桶合并成一次发布
        self.window_secs = rospy.get_param("~window_secs", 0.5)
        # 发布周期(秒)
        self.publish_period = rospy.get_param("~publish_period", 0.1)
        # 缓存 [(收到时刻 rospy.Time, Cone)]
        self._cones = []
        self._frame_id = "camera"

        # 发布ConeArray
        self.array_pub = rospy.Publisher("/visual_cone_array", ConeArray, queue_size=10)
        # 订阅视觉ConeDetection（camera/yolo_camera.py 单锥逐条发布）
        rospy.Subscriber("/perception/cones", ConeDetection, self.callback, queue_size=100)
        # 周期发布，保证下游持续拿到最新锥桶集合
        self.timer = rospy.Timer(rospy.Duration(self.publish_period), self.timer_callback)
        rospy.loginfo("[cone_detection_to_array] 累积窗口=%.1fs 发布周期=%.1fs → /visual_cone_array",
                      self.window_secs, self.publish_period)

    def callback(self, det_msg):
        now = rospy.Time.now()
        cone = Cone()
        cone.color = det_msg.color
        cone.x = det_msg.x
        cone.y = det_msg.y
        cone.confidence = 1.0  # ConeDetection 无置信度字段，给满值避免被下游阈值过滤
        if det_msg.header.frame_id:
            self._frame_id = det_msg.header.frame_id
        self._cones.append((now, cone))

    def timer_callback(self, _event):
        now = rospy.Time.now()
        # 清理窗口外的过期锥桶
        self._cones = [(t, c) for t, c in self._cones
                       if (now - t).to_sec() <= self.window_secs]
        if not self._cones:
            return
        array_msg = ConeArray()
        array_msg.header.stamp = now
        array_msg.header.frame_id = self._frame_id
        array_msg.cones = [c for _, c in self._cones]
        self.array_pub.publish(array_msg)


if __name__ == "__main__":
    try:
        Detection2Array()
        rospy.spin()
=======
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
>>>>>>> 2b94d0e604c68b4a4b697238211388d6cf586749
    except rospy.ROSInterruptException:
        pass
