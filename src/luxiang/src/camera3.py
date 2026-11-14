#!/usr/bin/env python2
# -*- coding: utf-8 -*-
import subprocess
import time
import sys
import signal
import rospy
import socket
import struct
from sensor_msgs.msg import NavSatFix, Imu, PointCloud2
from std_msgs.msg import String
from threading import Lock, Thread
import os

NET_IF = "eth1"
NET_TIMEOUT_SEC = 2.0


class CANUDPClient:
    def __init__(self):
        self.sock = None
        self.dest_ip = "192.168.0.7"
        self.dest_port = 20005
        self.addr_to = None

    def init(self):
        try:
            print "🔧 初始化UDP: 目标={}:{}，本地端口【系统自动分配】".format(self.dest_ip, self.dest_port)
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.addr_to = (self.dest_ip, self.dest_port)
            print "✅ CAN UDP客户端初始化成功（端口系统自动分配）"
            return True
        except Exception as e:
            print "❌ CAN UDP客户端初始化失败: {}".format(e)
            return False

    def send_can_frame(self, can_id, data):
        if self.sock is None:
            return False
        if len(data) != 8:
            return False
        try:
            packet = bytearray(13)
            frame_info = 0x00
            data_length = len(data)
            frame_info |= (data_length & 0x0F)
            if can_id > 0x7FF:
                frame_info |= 0x80
            else:
                frame_info |= 0x00
            packet[0] = frame_info
            if can_id > 0x7FF:
                packet[1] = (can_id >> 24) & 0xFF
                packet[2] = (can_id >> 16) & 0xFF
                packet[3] = (can_id >> 8) & 0xFF
                packet[4] = can_id & 0xFF
            else:
                packet[1] = 0x00
                packet[2] = 0x00
                packet[3] = (can_id >> 8) & 0xFF
                packet[4] = can_id & 0xFF
            for i in range(8):
                if i < len(data):
                    packet[5 + i] = data[i]
                else:
                    packet[5 + i] = 0x00
            self.sock.sendto(bytes(packet), self.addr_to)
            return True
        except Exception as e:
            return False

    def close(self):
        if self.sock:
            self.sock.close()
            self.sock = None


class SensorMonitor:
    def __init__(self, can_client, can_client_ok):
        # VLP16激光雷达断线检测新增
        self.lidar_last_time = None
        self.lidar_error = False
        self.can_client = can_client
        self.can_client_ok = can_client_ok
        self.gps_last_time = None
        self.imu_last_time = None
        self.lock = Lock()
        self.camera_error = False
        self.gps_error = False
        self.imu_error = False
        self.net_error = False
        self.net_last_up_time = time.time()
        self.error_sending_mode = False
        self.last_error_send_time = 0
        self.error_send_interval = 1.0 / 10
        self.CAN_ID = 0x100
        self.ERROR_DATA = [0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01]
        self.CAN_ID_220 = 0x220
        self.DATA_01010101 = [0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01]
        self.DATA_02020202 = [0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02]
        self.last_220_send_time = 0
        self.send_220_interval = 1.0 / 10
        self.send_220_count = 0
        self.last_check_time = 0
        self.check_interval = 1.0 / 10.0
        self.last_camera_check_time = 0
        self.camera_check_interval = 2.0
        self.record_thread = None
        self.record_dir = os.path.expanduser("~/Desktop/robot_records")

        # ========= 新增：双天线故障检测（05模式间接判断） =========
        self.ant1_fault = False          # 主天线/定位链路故障
        self.ant2_fault = False          # 副天线/航向链路故障（暂不可用，NavSatFix无航向位）
        self.ant_fault_start_time = None # 故障开始时间，用于防抖
        self.ANT_FAULT_DEBOUNCE = 1.0    # 防抖阈值：持续1秒异常才判定故障

    def start_loop_record(self):
        global recording_running
        if not os.path.exists(self.record_dir):
            os.makedirs(self.record_dir)
        if recording_running:
            return
        recording_running = True
        self.record_thread = Thread(target=record_worker, args=(self.record_dir,))
        self.record_thread.daemon = True
        self.record_thread.start()
        print "📂 录像保存目录: {}".format(self.record_dir)

    def stop_loop_record(self):
        global recording_running
        recording_running = False
        if self.record_thread is not None:
            self.record_thread.join(timeout=8)
        print "🛑 录像工作线程已退出"

    def send_error_frame_continuously(self):
        if not self.can_client_ok:
            return
        current_time = time.time()
        if current_time - self.last_error_send_time >= self.error_send_interval:
            self.can_client.send_can_frame(self.CAN_ID, self.ERROR_DATA)
            self.last_error_send_time = current_time

    def send_220_data(self):
        if not self.can_client_ok:
            return False
        current_time = time.time()
        if current_time - self.last_220_send_time >= self.send_220_interval:
            self.send_220_count += 1
            if self.send_220_count % 2 == 1:
                data_to_send = self.DATA_01010101
            else:
                data_to_send = self.DATA_02020202
            if self.send_220_count % 600 == 0:
                data_name = "01010101" if self.send_220_count % 2 == 1 else "02020202"
                print "🔵 已发送{}帧{} - CAN ID 0x220 (600Hz)".format(self.send_220_count, data_name)
            success = self.can_client.send_can_frame(self.CAN_ID_220, data_to_send)
            if success:
                self.last_220_send_time = current_time
            return success
        return True

    def check_ethernet_status(self):
        try:
            ret = subprocess.check_output(["ip", "addr", "show", NET_IF], shell=False)
            if "inet " in ret:
                self.net_last_up_time = time.time()
                if self.net_error:
                    print "✅ 网线连接恢复"
                    self.net_error = False
        except Exception:
            pass
        current = time.time()
        if current - self.net_last_up_time > NET_TIMEOUT_SEC:
            if not self.net_error:
                print "⚠️ 网线网口 {} 断开超时！".format(NET_IF)
            self.net_error = True
        else:
            self.net_error = False

    def check_sensor_status(self):
        current_time = time.time()
        if current_time - self.last_check_time >= self.check_interval:
            self.check_sensor_timeout()
            self.check_ethernet_status()
            self.last_check_time = current_time

    def gps_callback(self, msg):
        with self.lock:
            # 原有逻辑：刷新时间戳 + 通讯恢复提示
            self.gps_last_time = time.time()
            if self.gps_error:
                print "✅ GPS数据恢复"
                self.gps_error = False

            # ========= 新增：主天线故障间接判断 =========
            now = time.time()
            # NavSatFix.status.status != 0 代表定位解算有效
            pos_ok = (msg.status.status != 0)

            if not pos_ok:
                # 定位失效，开始防抖计时
                if self.ant_fault_start_time is None:
                    self.ant_fault_start_time = now
                else:
                    fault_duration = now - self.ant_fault_start_time
                    if fault_duration >= self.ANT_FAULT_DEBOUNCE:
                        if not self.ant1_fault:
                            print "⚠️【天线故障】主天线定位异常（防抖1s触发）"
                        self.ant1_fault = True
            else:
                # 定位恢复，清除故障和计时
                if self.ant1_fault:
                    print "✅【天线恢复】主天线定位恢复正常"
                self.ant1_fault = False
                self.ant_fault_start_time = None

    def lidar_callback(self, msg):
        # 雷达点云回调，更新最后接收时间
        with self.lock:
            self.lidar_last_time = time.time()
            if self.lidar_error:
                print "✅ VLP16雷达数据恢复"
                self.lidar_error = False

    def imu_callback(self, msg):
        with self.lock:
            self.imu_last_time = time.time()
            if self.imu_error:
                print "✅ IMU数据恢复"
                self.imu_error = False

    def check_sensor_timeout(self):
        current_time = time.time()
        with self.lock:
            # GPS通讯超时（惯导断电/串口断连）
            if self.gps_last_time is None:
                if not self.gps_error:
                    print "⚠️ 未接收到GPS数据"
                self.gps_error = True
            else:
                time_diff = current_time - self.gps_last_time
                if time_diff > 2.0:
                    if not self.gps_error:
                        print "⚠️ GPS超时 (%.2fs未更新)" % time_diff
                    self.gps_error = True
                else:
                    self.gps_error = False

            # IMU超时（完全未改动）
            if self.imu_last_time is None:
                if not self.imu_error:
                    print "⚠️ 未接收到IMU数据"
                self.imu_error = True
            else:
                time_diff = current_time - self.imu_last_time
                if time_diff > 2.0:
                    if not self.imu_error:
                        print "⚠️ IMU超时 (%.2fs未更新)" % time_diff
                    self.imu_error = True
                else:
                    self.imu_error = False

            # ========== VLP16雷达超时检测【完全未改动】 ==========
            if self.lidar_last_time is None:
                if not self.lidar_error:
                    print "⚠️ 未接收到VLP16雷达点云数据"
                self.lidar_error = True
            else:
                time_diff = current_time - self.lidar_last_time
                if time_diff > 2.0:
                    if not self.lidar_error:
                        print "⚠️ VLP16雷达超时 (%.2fs未更新)" % time_diff
                    self.lidar_error = True
                else:
                    self.lidar_error = False


def accurate_sensor_monitor():
    print "🎥 摄像头和传感器监控启动"
    print "📍 使用CAN1端口: 192.168.0.7:2001"
    rospy.init_node('camera_sensor_monitor', anonymous=True)
    can_dest_ip = rospy.get_param("~can_dest_ip", "192.168.0.7")
    can_dest_port = rospy.get_param("~can_dest_port", 20005)
    can_client = CANUDPClient()
    can_client.dest_ip = can_dest_ip
    can_client.dest_port = can_dest_port
    if not can_client.init():
        print "⚠️ CAN UDP客户端初始化失败，将继续监控但不发送CAN帧"
        can_client_ok = False
    else:
        can_client_ok = True

    def stop(sig, frame):
        print "\n🛑 手动停止监控"
        monitor.stop_loop_record()
        can_client.close()
        rospy.signal_shutdown("Manual stop")
        sys.exit(0)

    signal.signal(signal.SIGINT, stop)
    monitor = SensorMonitor(can_client, can_client_ok)
    print "🔵 CAN ID 0x220发送频率: 600Hz (奇偶次交替01010101/02020202)"
    print "🚨 CAN ID 0x100错误发送频率: 300Hz"
    print "🔍 状态检测频率: 10Hz"
    print "📷 摄像头检测频率: 0.5Hz"
    print "🌐 网线eth0超时:2s"
    print "📡 主天线故障防抖: 1s"
    print "等待传感器数据..."

    rospy.Subscriber('GPS_data', NavSatFix, monitor.gps_callback, queue_size=10)
    rospy.Subscriber('imu_data', Imu, monitor.imu_callback, queue_size=10)
    rospy.Subscriber('/velodyne_points', PointCloud2, monitor.lidar_callback, queue_size=10)
    print "✅ 已订阅VLP16雷达话题 /velodyne_points"
    print "✅ 已订阅GPS_data和imu_data话题"

    rospy.sleep(5.0)
    last_status_print_time = 0

    while not rospy.is_shutdown():
        monitor.send_220_data()
        monitor.check_sensor_status()
        current_time = time.time()

        # 任意传感器/天线故障，持续发送错误CAN帧
        if monitor.gps_error or monitor.lidar_error or monitor.imu_error or monitor.ant1_fault:
            monitor.send_error_frame_continuously()

        # 5秒打印一次状态
        if current_time - last_status_print_time >= 5.0:
            print "📊 状态 - GPS通讯: {}, IMU: {}, 雷达: {}, 网线: {}, 主天线: {}".format(
                "正常" if not monitor.gps_error else "错误",
                "正常" if not monitor.imu_error else "错误",
                "正常" if not monitor.lidar_error else "错误",
                "正常" if not monitor.net_error else "断开",
                "故障" if monitor.ant1_fault else "正常"
            )
            last_status_print_time = current_time

    monitor.stop_loop_record()
    print "\n📞 监控已停止"
    can_client.close()


if __name__ == "__main__":
    accurate_sensor_monitor()

