#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Stationary synthetic sensors for isolated validation, never vehicle commands."""
import math
import rospy
from sensor_msgs.msg import NavSatFix, Imu
from geometry_msgs.msg import PoseArray, Pose


def main():
    rospy.init_node('validation_inputs')
    gps_pub = rospy.Publisher('/GPS_data', NavSatFix, queue_size=10)
    imu_pub = rospy.Publisher('/imu_data', Imu, queue_size=10)
    lidar_pub = rospy.Publisher('/clustered_points', PoseArray, queue_size=10)
    rate = rospy.Rate(10)
    ox = rospy.get_param('~lidar_offset_x', 1.76618)
    oy = rospy.get_param('~lidar_offset_y', -.11304)
    yaw = math.radians(rospy.get_param('~lidar_yaw_offset_deg', .5))
    sign = rospy.get_param('~lidar_y_sign', -1.)
    if sign not in (-1., 1.):
        raise ValueError('lidar_y_sign must be +1 or -1')
    c, s = math.cos(yaw), math.sin(yaw)
    rospy.logwarn('Synthetic stationary sensors: isolated ROS master only; never mix with live sensors')
    while not rospy.is_shutdown():
        stamp = rospy.Time.now()
        imu = Imu()
        imu.header.stamp = stamp
        imu.orientation.z = 0.0
        gps = NavSatFix()
        gps.header.stamp = stamp
        gps.latitude, gps.longitude = 39.9260648, 116.1992637
        gps.status.status = 0
        cones = PoseArray()
        cones.header.stamp, cones.header.frame_id = stamp, 'velodyne'
        for x in (5., 10.):
            for y in (-1.8, 1.2):
                pose = Pose()
                dx, dy = x-ox, y-oy
                pose.position.x = c*dx+s*dy
                pose.position.y = (-s*dx+c*dy)/sign
                pose.position.z = -.4
                cones.poses.append(pose)
        imu_pub.publish(imu)
        gps_pub.publish(gps)
        lidar_pub.publish(cones)
        rate.sleep()


if __name__ == '__main__':
    main()
