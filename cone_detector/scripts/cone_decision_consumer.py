#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cone_decision_consumer.py
==========================
示例：下游模块订阅检测结果并处理决策信号

功能:
  - 订阅 /cone_detector/detections
  - 根据最高置信度锥桶输出最终决策
  - 打印检测报告

用法:
  python3 cone_decision_consumer.py
  或:
  rosrun cone_detector cone_decision_consumer.py
"""

import rospy
from cone_detector.msg import ConeDetectionArray, ConeDetection


class DecisionConsumer:

    def __init__(self):
        rospy.init_node("cone_decision_consumer", anonymous=True)
        self.sub = rospy.Subscriber(
            "/cone_detector/detections",
            ConeDetectionArray,
            self.on_detections,
            queue_size=10
        )
        rospy.loginfo("[consumer] 等待检测结果...")

    def on_detections(self, msg: ConeDetectionArray):
        stamp = msg.header.stamp.to_sec()

        if not msg.has_cones:
            rospy.loginfo_throttle(
                2.0,
                f"[consumer] t={stamp:.3f}  无检测  默认状态={msg.dominant_decision}"
            )
            self._execute_decision(msg.dominant_decision)
            return

        # 按置信度排序，取最高的锥桶作为主决策
        best: ConeDetection = max(msg.detections,
                                  key=lambda d: d.confidence)

        rospy.loginfo(
            f"[consumer] t={stamp:.3f}  "
            f"检测到 {len(msg.detections)} 个锥桶  "
            f"主决策: {best.decision}（{best.color}  "
            f"置信度={best.confidence:.2f}  "
            f"中心=({best.center_x:.3f},{best.center_y:.3f})）"
            f"  推理={msg.inference_time*1000:.1f}ms"
        )

        self._execute_decision(best.decision)

    def _execute_decision(self, decision: str):
        """
        在此实现实际决策响应逻辑
        例如: 发布控制指令、写入共享内存、调用服务...

        决策信号说明：
          TURN_LEFT  — 检测到红色锥桶，执行左转
          TURN_RIGHT — 检测到蓝色锥桶，执行右转
          STOP       — 检测到黄色锥桶，停止
          default    — 无锥桶检测，维持当前状态
        """
        decision_map = {
            "TURN_LEFT":  "🔴 左转",
            "TURN_RIGHT": "🔵 右转",
            "STOP":       "🟡 停止",
            "default":    "⬜ 无锥桶 / 维持状态",
        }
        rospy.logdebug(f"[consumer] 执行决策: {decision_map.get(decision, decision)}")
        # TODO: 在此实现控制逻辑（发布 cmd_vel 等）

    def spin(self):
        rospy.spin()


if __name__ == "__main__":
    try:
        DecisionConsumer().spin()
    except rospy.ROSInterruptException:
        pass
