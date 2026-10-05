#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import rospy
import ros_numpy
from sensor_msgs.msg import PointCloud2


def filter_points(cloud, fov_degrees=270.0, max_range=10.0, max_z=0.8):
    """Horizontal sector centred on lidar +X; preserve point fields."""
    cloud = cloud.reshape(-1)
    x, y, z = cloud['x'], cloud['y'], cloud['z']
    finite = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    angle = np.abs(np.arctan2(y, x))
    radius_squared = x*x + y*y
    mask = (finite & (radius_squared > 0.0) &
            (radius_squared <= max_range*max_range) &
            (angle <= np.deg2rad(fov_degrees / 2.0)) & (z < max_z))
    return cloud[mask]


def prefilter(msg):
    cloud = ros_numpy.point_cloud2.pointcloud2_to_array(msg)
    selected = filter_points(cloud, fov_degrees, max_range, max_z)
    rospy.loginfo_throttle(1.0, 'Lidar sector points: %d -> %d', cloud.size, selected.size)
    output = ros_numpy.point_cloud2.array_to_pointcloud2(
        selected, stamp=msg.header.stamp, frame_id=msg.header.frame_id)
    bag_publisher.publish(output)


if __name__ == '__main__':
    rospy.init_node('prefiltering')
    fov_degrees = float(rospy.get_param('~fov_degrees', 270.0))
    max_range = float(rospy.get_param('~max_range', 5.0))
    max_z = float(rospy.get_param('~max_z', 0.8))
    if (not all(np.isfinite(v) for v in (fov_degrees, max_range, max_z)) or
            not 0.0 < fov_degrees <= 360.0 or max_range <= 0.0):
        raise ValueError('Invalid lidar sector parameters')
    bag_publisher = rospy.Publisher('/prefiltered_points', PointCloud2, queue_size=1)
    bag_subscriber = rospy.Subscriber('/velodyne_points', PointCloud2,
                                      prefilter, queue_size=1, buff_size=2**24)
    rospy.spin()
