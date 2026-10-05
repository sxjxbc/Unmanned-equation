#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
虚拟 VCU v3.0.2 (兼容 Python 2/3)
"""

import socket, time, math, threading
import rospy
from sensor_msgs.msg import NavSatFix, Imu
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Float64
import tf.transformations as tf_trans

WHEEL_RADIUS = 0.3
WHEELBASE = 1.6
PHYSICAL_MAX_STEER = math.pi / 2.0
# 自行车模型的 tan 保护: 略小于 90°
STEER_MODEL_LIMIT = math.radians(89.0)   # ≈ 1.5533 rad
# yaw_rate 物理上限 (rad/s), 用于避免转弯率无穷大
MAX_YAW_RATE = 3.0                        # 可调
ACCEL = 2.0
DECEL = 3.0
MAX_SPEED = 5.0
UPDATE_RATE = 20.0

LISTEN_IP = '0.0.0.0'
LISTEN_PORT = 20001
CAN_ID = 0x210

FRONT_OVERHANG = 0.7#,需要按照我们车实测标定
INIT_X = -15.0 - FRONT_OVERHANG
INIT_Y = 0.0
INIT_YAW = 0.0

GPS_ORIGIN_LAT = 39.9
GPS_ORIGIN_LON = 116.4
R_EARTH = 6371000.0


# ==================== 模块级工具函数 ====================
def _byte(b):
    """兼容 Python 2 (str) 和 Python 3 (bytes/int)"""
    if isinstance(b, int):
        return b
    return ord(b)


def parse_usr_packet(pkt):
    if len(pkt) < 13:
        return None, None
    dlc = _byte(pkt[0]) & 0x0F
    can_id = (_byte(pkt[3]) << 8) | _byte(pkt[4])
    data = pkt[5:5 + dlc]
    return can_id, data

def decode_0x210(data):
    """[steer int16LE][rpm int16LE][stop int16LE][00 00]"""
    if len(data) < 6:
        return None, None, None

    steer_raw = (_byte(data[1]) << 8) | _byte(data[0])
    rpm_raw   = (_byte(data[3]) << 8) | _byte(data[2])
    stop_raw  = (_byte(data[5]) << 8) | _byte(data[4])

    # 急停逻辑：tracking 发 steer_raw >= 2000 触发 E-STOP
    if steer_raw >= 2000:
        return 0.0, 0.0, 1

    # 正常转向：0~1000 -> -pi/2 ~ +pi/2
    steer_raw = max(0, min(1000, steer_raw))
    steering_rad = (steer_raw - 500) * math.pi / 1000.0

    # 轮速 RPM -> m/s
    speed_mps = rpm_raw * 2.0 * math.pi * WHEEL_RADIUS / 60.0

    return steering_rad, speed_mps, int(stop_raw)


# ==================== 车辆模型 ====================
class VehicleState(object):
    def __init__(self):
        self.x = INIT_X
        self.y = INIT_Y
        self.yaw = INIT_YAW
        self.speed = 0.0
        self.target_speed = 0.0
        self.target_steering = 0.0
        self.stop = 0
        self.last_time = time.time()

    def update(self, dt):
        if self.stop:
            self.target_speed = 0.0

        if self.speed < self.target_speed:
            self.speed = min(self.speed + ACCEL * dt, self.target_speed)
        else:
            self.speed = max(self.speed - DECEL * dt, self.target_speed)
        self.speed = max(0.0, min(MAX_SPEED, self.speed))

        # 物理执行上限
        exec_steer = max(-PHYSICAL_MAX_STEER,
                         min(PHYSICAL_MAX_STEER, self.target_steering))

        # 自行车模型: tan 保护
        model_steer = max(-STEER_MODEL_LIMIT,
                          min(STEER_MODEL_LIMIT, exec_steer))

        if self.speed > 0.01:
            yaw_rate = self.speed * math.tan(model_steer) / WHEELBASE
            # yaw_rate 上限 (防爆)
            yaw_rate = max(-MAX_YAW_RATE, min(MAX_YAW_RATE, yaw_rate))
        else:
            yaw_rate = 0.0

        self.yaw += yaw_rate * dt
        self.yaw = math.atan2(math.sin(self.yaw), math.cos(self.yaw))
        self.x += self.speed * math.cos(self.yaw) * dt
        self.y += self.speed * math.sin(self.yaw) * dt


# ==================== VCU ====================
class VirtualVCU(object):
    def __init__(self):
        rospy.init_node('virtual_vcu', anonymous=True)

        self.vehicle = VehicleState()
        self.running = True

        self.gps_pub = rospy.Publisher('/GPS_data', NavSatFix, queue_size=10)
        self.imu_pub = rospy.Publisher('/imu_data', Imu, queue_size=50)
        self.speed_pub = rospy.Publisher('/vehicle_speed', Float64, queue_size=10)
        self.gt_pub = rospy.Publisher('/vehicle_pose_gt', PoseStamped, queue_size=10)

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind((LISTEN_IP, LISTEN_PORT))
        self.sock.settimeout(0.1)

        rospy.loginfo("[VCU] 监听 UDP %s:%d", LISTEN_IP, LISTEN_PORT)

        t = threading.Thread(target=self._listen_loop)
        t.daemon = True
        t.start()

        rospy.Timer(rospy.Duration(1.0 / UPDATE_RATE), self._update)

        rospy.loginfo("[VCU] 初始位置 (%.1f, %.1f), yaw=%.1f°",
                      INIT_X, INIT_Y, math.degrees(INIT_YAW))

    def _listen_loop(self):
        while self.running and not rospy.is_shutdown():
            try:
                pkt, _ = self.sock.recvfrom(256)
                can_id, data = parse_usr_packet(pkt)      # 模块级函数
                if can_id == CAN_ID:
                    steer, spd, stop = decode_0x210(data)
                    if steer is not None:
                        self.vehicle.target_steering = steer
                        self.vehicle.target_speed = spd
                        self.vehicle.stop = stop
                        rospy.loginfo_throttle(1.0,
                            "[VCU] RX steer=%.2f°, v=%.2f m/s, stop=%d",
                            math.degrees(steer), spd, stop)
            except socket.timeout:
                continue
            except Exception as e:
                rospy.logwarn("[VCU] 接收错误: %s", e)

    def _update(self, _):
        now = time.time()
        dt = now - self.vehicle.last_time
        self.vehicle.last_time = now
        if dt <= 0 or dt > 0.5:
            dt = 1.0 / UPDATE_RATE

        self.vehicle.update(dt)
        stamp = rospy.Time.now()

        lat = GPS_ORIGIN_LAT + math.degrees(self.vehicle.x / R_EARTH)
        # 地图约定：X=前方，Y=右方
        lon = GPS_ORIGIN_LON + math.degrees(self.vehicle.y / (R_EARTH * math.cos(math.radians(GPS_ORIGIN_LAT))))
        gps = NavSatFix()
        gps.header.stamp = stamp
        gps.header.frame_id = "gps"
        gps.latitude = lat
        gps.longitude = lon
        gps.altitude = 0.0
        gps.status.status = 0
        gps.status.service = 1
        gps.position_covariance = [0.1, 0, 0, 0, 0.1, 0, 0, 0, 0.1]
        gps.position_covariance_type = 1
        self.gps_pub.publish(gps)

        imu = Imu()
        imu.header.stamp = stamp
        imu.header.frame_id = "imu"
        imu.orientation.x = 0.0
        imu.orientation.y = 0.0
        # Jiantu 使用罗盘航向：0=北、顺时针为正。
        heading_deg = -math.degrees(self.vehicle.yaw)
        heading_deg = (heading_deg + 360.0) % 360.0
        imu.orientation.z = heading_deg
        imu.orientation.w = 1.0
        imu.orientation_covariance = [0.01, 0, 0, 0, 0.01, 0, 0, 0, 0.01]
        self.imu_pub.publish(imu)

        pose = PoseStamped()
        pose.header.stamp = stamp
        pose.header.frame_id = "map"
        pose.pose.position.x = self.vehicle.x
        pose.pose.position.y = self.vehicle.y
        q = tf_trans.quaternion_from_euler(0, 0, self.vehicle.yaw)
        pose.pose.orientation.x = q[0]
        pose.pose.orientation.y = q[1]
        pose.pose.orientation.z = q[2]
        pose.pose.orientation.w = q[3]
        self.gt_pub.publish(pose)

        self.speed_pub.publish(Float64(self.vehicle.speed))

    def shutdown(self):
        self.running = False
        self.sock.close()


if __name__ == '__main__':
    vcu = VirtualVCU()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        pass
    finally:
        vcu.shutdown()