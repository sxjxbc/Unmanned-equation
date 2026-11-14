#!/usr/bin/env python
"""Original point-cloud conversion with rear/side support for curved corridors."""
import rospy
import ros_numpy
import numpy as np
from sensor_msgs.msg import PointCloud2


def callback(message):
    points=ros_numpy.point_cloud2.pointcloud2_to_array(message).reshape(-1)
    x,y,z=points['x'],points['y'],points['z']
    keep=np.isfinite(x)&np.isfinite(y)&np.isfinite(z)&(x>-6.)&(x<12.)&(y>-6.)&(y<6.)&(z<.8)
    publisher.publish(ros_numpy.point_cloud2.array_to_pointcloud2(points[keep],
                      stamp=message.header.stamp,frame_id='velodyne'))


if __name__=='__main__':
    rospy.init_node('trackdrive_prefilter')
    publisher=rospy.Publisher('/prefiltered_points',PointCloud2,queue_size=1)
    rospy.Subscriber('/velodyne_points',PointCloud2,callback,queue_size=1)
    rospy.spin()
