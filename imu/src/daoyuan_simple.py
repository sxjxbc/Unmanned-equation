#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    Function: ROS parsing INS570D.
    Version: V1.0.5 - 简化版，直接解析
"""

import sys
import os
import rospy
import serial
import struct
from sensor_msgs.msg import Imu, NavSatFix

print("=== daoyuan.py 启动 ===")
print("Python 版本: {}".format(sys.version))

def print_hex(data, length=20):
    """打印十六进制"""
    hex_str = ' '.join(['%02X' % ord(c) if isinstance(c, str) else '%02X' % c for c in data[:length]])
    return hex_str

def byte_value(value):
    return ord(value) if isinstance(value, str) else value


def main():
    rospy.init_node("serial_port", anonymous=True)
    loop_rate = rospy.Rate(10)

    GPS_pub = rospy.Publisher("GPS_data", NavSatFix, queue_size=10)
    imu_pub = rospy.Publisher("imu_data", Imu, queue_size=50)

    imu_msg = Imu()
    gps_msg = NavSatFix()
    
    gps_msg.header.frame_id = "gps"
    imu_msg.header.frame_id = "imu"
    # No documented device fix-quality decoder exists in this driver.
    # Never fabricate covariance or advertise an unverified fix by default.
    assume_fix = rospy.get_param('~assume_gps_fix', False)
    gps_msg.status.status = 0 if assume_fix else -1
    gps_msg.position_covariance_type = 0
    if assume_fix:
        rospy.logwarn('GPS fix is manually assumed; device quality is NOT verified')
    gps_msg.status.service = 1

    # 打开串口
    try:
        ser = serial.Serial(
            port='/dev/imu',
            baudrate=230400,
            bytesize=serial.EIGHTBITS,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
            timeout=0.5,
	    rtscts=False
        )
        rospy.loginfo("串口打开成功: /dev/imu, 230400")
    except Exception as e:
        rospy.logerr("串口打开失败: %s", e)
        return -1

    # 创建目录
    dir_path = os.path.expanduser(rospy.get_param("~log_dir", "~/nvidia/nrt_ws/"))
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    
    out_file = open(dir_path + "/True_lon_lat.txt", "w")
    rospy.loginfo("文件打开成功")

    buffer = b""
    packet_count = 0
    
    rospy.loginfo("等待串口数据...")

    while not rospy.is_shutdown():
        try:
            if ser.in_waiting > 0:
                # 读取数据
                data = ser.read(ser.in_waiting)
                buffer += data
                
                # 查找 BD DB 0B
                while True:
                    # 查找帧头位置
                    pos = buffer.find(b'\xBD\xDB\x0B')
                    if pos == -1:
                        # 没有找到帧头，保留最后10字节防止截断
                        if len(buffer) > 100:
                            buffer = buffer[-10:]
                        break
                    
                    # 检查是否有足够数据
                    if len(buffer) < pos + 64:
                        break
                    
                    # 找到数据包
                    packet_count += 1
                    rospy.loginfo("=== 找到第 %d 个数据包, 位置: %d ===", packet_count, pos)
                    
                    # 提取64字节数据包
                    packet = buffer[pos:pos+64]
                    
                    # 打印前20字节
                    rospy.loginfo("数据包: %s", print_hex(packet[:20]))
                    
                    try:
                        # 解析数据 (使用ord处理Python2的字符串)
                        # 横滚角 (偏移3-4)
                        roll_raw = (byte_value(packet[4]) << 8) | byte_value(packet[3])
                        if roll_raw & 0x8000:
                            roll_raw = roll_raw - 0x10000
                        roll = roll_raw * 360.0 / 32768.0
                        
                        # 俯仰角 (偏移5-6)
                        pitch_raw = (byte_value(packet[6]) << 8) | byte_value(packet[5])
                        if pitch_raw & 0x8000:
                            pitch_raw = pitch_raw - 0x10000
                        pitch = pitch_raw * 360.0 / 32768.0
                        
                        # 航向角 (偏移7-8)
                        heading_raw = (byte_value(packet[8]) << 8) | byte_value(packet[7])
                        if heading_raw & 0x8000:
                            heading_raw = heading_raw - 0x10000
                        heading = heading_raw * 360.0 / 32768.0
                        
                        # 纬度 (偏移21-24, 小端32位)
                        lat_raw = (byte_value(packet[24]) << 24) | (byte_value(packet[23]) << 16) | (byte_value(packet[22]) << 8) | byte_value(packet[21])
                        if lat_raw & 0x80000000:
                            lat_raw = lat_raw - 0x100000000
                        latitude = lat_raw * 0.0000001
                        
                        # 经度 (偏移25-28, 小端32位)
                        lon_raw = (byte_value(packet[28]) << 24) | (byte_value(packet[27]) << 16) | (byte_value(packet[26]) << 8) | byte_value(packet[25])
                        if lon_raw & 0x80000000:
                            lon_raw = lon_raw - 0x100000000
                        longitude = lon_raw * 0.0000001
                        
                        # 高度 (偏移29-32, 小端32位)
                        alt_raw = (byte_value(packet[32]) << 24) | (byte_value(packet[31]) << 16) | (byte_value(packet[30]) << 8) | byte_value(packet[29])
                        if alt_raw & 0x80000000:
                            alt_raw = alt_raw - 0x100000000
                        altitude = alt_raw * 0.001
                        
                        rospy.loginfo("航向: %.2f°, 纬度: %.7f, 经度: %.7f, 高度: %.2f", 
                                     heading, latitude, longitude, altitude)
                        
                        # 发布IMU
                        packet_stamp = rospy.Time.now()
                        imu_msg.header.stamp = packet_stamp
                        imu_msg.orientation.z = heading
                        imu_msg.orientation_covariance = [0.01, 0, 0, 0, 0.01, 0, 0, 0, 0.01]
                        imu_pub.publish(imu_msg)
                        
                        # 发布GPS
                        gps_msg.header.stamp = packet_stamp
                        gps_msg.latitude = latitude
                        gps_msg.longitude = longitude
                        gps_msg.altitude = altitude
                        gps_msg.position_covariance = [0.0] * 9
                        gps_msg.position_covariance_type = 0
                        GPS_pub.publish(gps_msg)
                        
                        # 写入文件
                        out_file.write("%.7f\t%.7f\t%.2f\n" % (latitude, longitude, heading))
                        out_file.flush()
                        
                    except Exception as e:
                        rospy.logerr("解析错误: %s", e)
                    
                    # 移除已处理的数据包
                    buffer = buffer[pos+64:]
                    
            else:
                rospy.sleep(0.01)
                
        except Exception as e:
            rospy.logerr("主循环错误: %s", e)
            rospy.sleep(0.1)

    out_file.close()
    ser.close()
    return 0

if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        rospy.loginfo("节点被中断")
    except Exception as e:
        rospy.logerr("未预期错误: %s", e)
