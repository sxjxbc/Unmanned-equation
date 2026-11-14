#!/usr/bin/env python2
# -*- coding: utf-8 -*-
import subprocess
import time
import sys
import signal
import rospy
import socket
from sensor_msgs.msg import NavSatFix, Imu, PointCloud2
from threading import Lock

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
            print "UDP init: target={}:{}".format(self.dest_ip, self.dest_port)
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.addr_to = (self.dest_ip, self.dest_port)
            print "UDP client ready"
            return True
        except Exception as e:
            print "UDP init failed: {}".format(e)
            return False

    def send_can_frame(self, can_id, data):
        if self.sock is None:
            return False
        if len(data) != 8:
            return False
        try:
            packet = bytearray(13)
            frame_info = len(data) & 0x0F
            if can_id > 0x7FF:
                frame_info |= 0x80
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
                packet[5 + i] = data[i] if i < len(data) else 0x00
            self.sock.sendto(bytes(packet), self.addr_to)
            return True
        except Exception:
            return False

    def close(self):
        if self.sock:
            self.sock.close()
            self.sock = None


class SensorMonitor:
    def __init__(self, can_client, can_client_ok):
        self.can_client = can_client
        self.can_client_ok = can_client_ok

        self.gps_last_time = None
        self.imu_last_time = None
        self.lidar_last_time = None
        self.lock = Lock()

        self.gps_error = False
        self.imu_error = False
        self.lidar_error = False
        self.net_error = False

        self.net_last_up_time = time.time()

        self.error_send_interval = 1.0 / 10.0
        self.CAN_ID = 0x100
        self.ERROR_DATA = [0x01] * 8

        self.CAN_ID_220 = 0x220
        self.DATA_01010101 = [0x01] * 8
        self.DATA_02020202 = [0x02] * 8
        self.last_220_send_time = 0
        self.send_220_interval = 1.0 / 10.0
        self.send_220_count = 0

        self.last_check_time = 0
        self.check_interval = 1.0 / 10.0

    def send_error_frame(self):
        if not self.can_client_ok:
            return
        now = time.time()
        if now - getattr(self, "_last_err", 0) >= self.error_send_interval:
            self.can_client.send_can_frame(self.CAN_ID, self.ERROR_DATA)
            self._last_err = now

    def send_220_data(self):
        if not self.can_client_ok:
            return
        now = time.time()
        if now - self.last_220_send_time >= self.send_220_interval:
            self.send_220_count += 1
            if self.send_220_count % 2 == 1:
                data = self.DATA_01010101
            else:
                data = self.DATA_02020202
            if self.can_client.send_can_frame(self.CAN_ID_220, data):
                self.last_220_send_time = now

    def check_ethernet(self):
        try:
            out = subprocess.check_output(["ip", "addr", "show", NET_IF])
            if "inet " in out:
                self.net_last_up_time = time.time()
                if self.net_error:
                    print "net recovered"
                    self.net_error = False
        except Exception:
            pass
        if time.time() - self.net_last_up_time > NET_TIMEOUT_SEC:
            if not self.net_error:
                print "net {} down".format(NET_IF)
            self.net_error = True

    def gps_callback(self, msg):
        with self.lock:
            self.gps_last_time = time.time()
            if self.gps_error:
                print "GPS recovered"
                self.gps_error = False

    def imu_callback(self, msg):
        with self.lock:
            self.imu_last_time = time.time()
            if self.imu_error:
                print "IMU recovered"
                self.imu_error = False

    def lidar_callback(self, msg):
        with self.lock:
            self.lidar_last_time = time.time()
            if self.lidar_error:
                print "Lidar recovered"
                self.lidar_error = False

    def _check_timeout(self, last, label, cur):
        if last is None:
            print "{} no data".format(label)
            return True
        if cur - last > 2.0:
            print "{} timeout {:.2f}s".format(label, cur - last)
            return True
        return False

    def check_all_sensors(self):
        now = time.time()
        if now - self.last_check_time < self.check_interval:
            return
        self.last_check_time = now
        with self.lock:
            self.gps_error = self._check_timeout(self.gps_last_time, "GPS", now)
            self.imu_error = self._check_timeout(self.imu_last_time, "IMU", now)
            self.lidar_error = self._check_timeout(self.lidar_last_time, "Lidar", now)
        self.check_ethernet()


def main():
    print "sensor monitor start, CAN {}:{}".format("192.168.0.7", 20005)
    rospy.init_node('camera_sensor_monitor', anonymous=True)
    dest_ip = rospy.get_param("~can_dest_ip", "192.168.0.7")
    dest_port = rospy.get_param("~can_dest_port", 20005)

    can_client = CANUDPClient()
    can_client.dest_ip = dest_ip
    can_client.dest_port = dest_port
    ok = can_client.init()
    if not ok:
        print "CAN init failed, no frames will be sent"

    def stop(sig, frame):
        print "manual stop"
        can_client.close()
        rospy.signal_shutdown("stop")
        sys.exit(0)
    signal.signal(signal.SIGINT, stop)

    mon = SensorMonitor(can_client, ok)
    rospy.Subscriber('GPS_data', NavSatFix, mon.gps_callback, queue_size=10)
    rospy.Subscriber('imu_data', Imu, mon.imu_callback, queue_size=10)
    rospy.Subscriber('/velodyne_points', PointCloud2, mon.lidar_callback, queue_size=10)
    print "subscribed GPS/IMU/velodyne"

    rospy.sleep(5.0)
    last_print = 0
    while not rospy.is_shutdown():
        mon.send_220_data()
        mon.check_all_sensors()
        if mon.gps_error or mon.imu_error or mon.lidar_error or mon.net_error:
            mon.send_error_frame()
        now = time.time()
        if now - last_print >= 5.0:
            print "status GPS={} IMU={} Lidar={} net={}".format(
                "ERR" if mon.gps_error else "OK",
                "ERR" if mon.imu_error else "OK",
                "ERR" if mon.lidar_error else "OK",
                "DOWN" if mon.net_error else "OK")
            last_print = now
        time.sleep(0.001)

    can_client.close()
    print "monitor stopped"


if __name__ == "__main__":
    main()



