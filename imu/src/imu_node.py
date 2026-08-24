#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
    Function: ROS parsing GPS.
    Author: GuanShuai
    Date: 2021/7/31
    Version: V1.0.3
"""

import sys
import os
import rospy
import serial
import struct
import numpy as np
from sensor_msgs.msg import Imu, NavSatFix

print("=== daoyuan.py 启动 ===")
print("Python 版本: {}".format(sys.version))

def zhengshu(example):
    """十六进制转换十进制"""
    return int(example)

def zhengshu2(example):
    return int(example)

def convert(a):
    """将16位无符号整数转换为二进制字符串"""
    return format(a, '016b')

def convert2(a):
    """将32位无符号整数转换为二进制字符串"""
    return format(a, '032b')

def hextode(hexstr, example):
    """十六进制转十进制（处理负数）"""
    if hexstr[0] > '0':
        value = example
        if value & 0x8000:  # 检查符号位
            value = value - 0x10000  # 转换为有符号数
        return value
    else:
        return zhengshu(example)

def hextode2(hexstr, example):
    """32位十六进制转十进制（处理负数）"""
    if hexstr[0] > '0':
        value = example
        if value & 0x80000000:  # 检查符号位
            value = value - 0x100000000  # 转换为有符号数
        return value
    else:
        return zhengshu2(example)

def print_hex_data(data, start_idx, length=20, description=""):
    """打印十六进制数据用于调试 - 修复版本"""
    hex_list = []
    for i in range(min(length, len(data) - start_idx)):
        # 处理各种可能的数据类型
        byte_val = data[start_idx + i]
        if isinstance(byte_val, str):  # 如果是字符串
            hex_list.append("%02X" % ord(byte_val))  # 转换为ASCII码
        else:  # 如果是数字
            hex_list.append("%02X" % byte_val)
    hex_str = " ".join(hex_list)
    rospy.loginfo("%s: [%s]", description, hex_str)

def main():
    rospy.init_node("imu_node", anonymous=True)
    loop_rate = rospy.Rate(10)

    # 修改：增加队列大小，确保消息能发布出去
    GPS_pub = rospy.Publisher("GPS_data", NavSatFix, queue_size=10)
    imu_pub = rospy.Publisher("imu_data", Imu, queue_size=50)

    imu_msg = Imu()
    gps_msg = NavSatFix()
    
    # 修改：完善消息头信息
    gps_msg.header.frame_id = "gps"
    imu_msg.header.frame_id = "imu"
    
    # 修改：设置GPS状态为有效
    gps_msg.status.status = 0  # 0 = 无故障
    gps_msg.status.service = 1  # 1 = GPS

    # 创建串口对象
    try:
        ser = serial.Serial(
            port='/dev/ttyUSB0',
            baudrate=115200,
            timeout=0.1
        )
        rospy.loginfo("串口打开成功: /dev/ttyUSB0")
        
        # 修改：等待串口稳定
        rospy.sleep(1.0)
        
    except serial.SerialException as e:
        rospy.logerr("无法打开串口: %s", e)
        # 修改：尝试其他可能的串口设备
        try:
            ser = serial.Serial(
                port='/dev/ttyUSB1',
                baudrate=115200,
                timeout=0.1
            )
            rospy.loginfo("串口打开成功: /dev/ttyUSB1")
        except serial.SerialException as e2:
            rospy.logerr("也无法打开 /dev/ttyUSB1: %s", e2)
            return -1

    # 确保目录存在
    directory_path = "/home/nvidia/catkin_wss/txtshuju"
    if not os.path.exists(directory_path):
        try:
            os.makedirs(directory_path)
            rospy.loginfo("创建目录: %s", directory_path)
        except OSError as e:
            rospy.logerr("无法创建目录 %s: %s", directory_path, e)

    # 文件处理
    out_file = None
    try:
        out_file = open("/home/nvidia/catkin_wss/txtshuju/True_lon_lat.txt", "w")
        rospy.loginfo("文件打开成功")
    except IOError as e:
        rospy.logerr("无法打开文件: %s", e)

    try:
        rospy.loginfo("daoyuan 节点开始运行...")
        rospy.loginfo("等待串口数据...")

        buffer = bytearray()  # 使用bytearray确保统一的数据类型
        packet_size = 66
        debug_count = 0
        packet_count = 0
        
        while not rospy.is_shutdown():
            try:
                if ser.in_waiting > 0:
                    # 直接读取为bytes对象并扩展到bytearray
                    data = ser.read(ser.in_waiting)
                    buffer.extend(data)
                    
                    if debug_count < 5:  # 增加调试次数
                        rospy.loginfo("收到 %d 字节数据，缓冲区大小: %d", len(data), len(buffer))
                        # 打印前20字节查看实际数据格式
                        if len(buffer) >= 20:
                            print_hex_data(buffer, 0, 20, "前20字节数据")
                        debug_count += 1
                    
                    # 查找完整的数据包
                    i = 0
                    found_packet = False
                    while i < len(buffer) - packet_size + 1:
                        # 检查数据包头尾
                        header_ok = (buffer[i] == 0xbd and buffer[i+1] == 0xdb and buffer[i+2] == 0x0b)
                        footer_ok = (buffer[i+63] == 0xbd and buffer[i+64] == 0xdb and buffer[i+65] == 0x0a)
                        
                        if header_ok and footer_ok:
                            rospy.loginfo("=== 找到第 %d 个有效数据包! 位置: %d ===", packet_count + 1, i)
                            found_packet = True
                            packet_count += 1
                            
                            # 打印完整数据包用于验证
                            print_hex_data(buffer, i, 66, "完整数据包")
                            
                            start_gga = i + 3
                            
                            try:
                                # 航向角
                                hangxiang = (buffer[start_gga + 5] << 8) | buffer[start_gga + 4]
                                final_string = convert(hangxiang)
                                result = hextode(final_string, hangxiang)
                                hangxiang_true = result * 360.0 / 32768.0
                                
                                # 设置时间戳
                                current_time = rospy.Time.now()
                                imu_msg.header.stamp = current_time
                                imu_msg.orientation.z = hangxiang_true
                                
                                # 修改：完善IMU消息的其他字段
                                imu_msg.orientation_covariance = [0.01, 0, 0, 0, 0.01, 0, 0, 0, 0.01]
                                
                                imu_pub.publish(imu_msg)
                                rospy.loginfo("发布IMU数据 - 航向角: %f", hangxiang_true)

                                # 纬度
                                weidu = ((buffer[start_gga + 21] << 24) | 
                                        (buffer[start_gga + 20] << 16) | 
                                        (buffer[start_gga + 19] << 8) | 
                                        buffer[start_gga + 18])
                                weidu_string = convert2(weidu)
                                result2 = hextode2(weidu_string, weidu)
                                weidu_true = result2 * 0.0000001
                                rospy.loginfo("纬度: %.11f", weidu_true)

                                # 经度
                                jingdu = ((buffer[start_gga + 25] << 24) | 
                                         (buffer[start_gga + 24] << 16) | 
                                         (buffer[start_gga + 23] << 8) | 
                                         buffer[start_gga + 22])
                                jingdu_string = convert2(jingdu)
                                result3 = hextode2(jingdu_string, jingdu)
                                jingdu_true = result3 * 0.0000001
                                rospy.loginfo("经度: %.10f", jingdu_true)

                                # 写入文件
                                if out_file is not None:
                                    out_file.write("{:.7f}\t{:.7f}\t{:.7f}\n".format(
                                        weidu_true, jingdu_true, hangxiang_true))
                                    out_file.flush()

                                # 发布GPS数据
                                gps_msg.header.stamp = current_time
                                gps_msg.latitude = weidu_true
                                gps_msg.longitude = jingdu_true
                                gps_msg.altitude = 0.0
                                
                                # 修改：设置位置协方差（表示数据质量）
                                gps_msg.position_covariance = [0.1, 0, 0, 0, 0.1, 0, 0, 0, 0.1]
                                gps_msg.position_covariance_type = 1  # 近似方差已知
                                
                                GPS_pub.publish(gps_msg)
                                rospy.loginfo("发布GPS数据 - 纬度: %.7f, 经度: %.7f", weidu_true, jingdu_true)

                            except Exception as parse_error:
                                rospy.logerr("数据解析错误: %s", parse_error)
                                import traceback
                                rospy.logerr(traceback.format_exc())

                            # 移除已处理的数据包
                            del buffer[:i + packet_size]
                            break
                        else:
                            i += 1
                    
                    if not found_packet:
                        if debug_count < 10:  # 增加调试显示次数
                            rospy.loginfo("未找到有效数据包，继续搜索...缓冲区大小: %d", len(buffer))
                        # 如果一直没有找到数据包，尝试其他常见的数据包头
                        if len(buffer) > 200 and debug_count < 15:
                            rospy.logwarn("缓冲区已累积 %d 字节但未找到数据包，检查数据格式", len(buffer))
                            # 查找可能的数据包头
                            header_found = False
                            for j in range(min(100, len(buffer)-5)):
                                if buffer[j] == 0xbd or buffer[j] == 0xaa or buffer[j] == 0x55 or buffer[j] == 0x24:
                                    if not header_found and j < 20:  # 只显示前几个可能的位置
                                        print_hex_data(buffer, j, 10, "可能的数据包头位置 %d" % j)
                                        header_found = True
                    
                    # 如果缓冲区太大，清理旧数据但保留足够空间
                    if len(buffer) > 5000:
                        rospy.logwarn("缓冲区过大(%d字节)，清理数据", len(buffer))
                        # 保留最后1000字节继续搜索
                        buffer = buffer[-1000:]
                        
                else:
                    # 修改：如果没有数据，稍微等待
                    rospy.sleep(0.01)

            except Exception as e:
                rospy.logerr("数据处理错误: %s", e)
                import traceback
                rospy.logerr(traceback.format_exc())
                # 发生错误时清理缓冲区但保留一些数据
                if len(buffer) > 100:
                    buffer = buffer[-100:]
                rospy.sleep(0.1)

            loop_rate.sleep()

    except Exception as e:
        rospy.logerr("主循环错误: %s", e)
        import traceback
        rospy.logerr(traceback.format_exc())
    finally:
        if out_file is not None:
            out_file.close()
            rospy.loginfo("文件已关闭")
        if ser.is_open:
            ser.close()
            rospy.loginfo("串口已关闭")

    return 0

if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        rospy.loginfo("节点被用户中断")
    except Exception as e:
        rospy.logerr("未预期的错误: {}".format(e))
        import traceback
        rospy.logerr(traceback.format_exc())
