cmake_minimum_required(VERSION 3.0.2)
project(lidar_nodes)
find_package(catkin REQUIRED COMPONENTS mapping rospy roscpp std_msgs sensor_msgs geometry_msgs visualization_msgs)
catkin_package(CATKIN_DEPENDS mapping rospy roscpp std_msgs sensor_msgs geometry_msgs visualization_msgs)
catkin_install_python(PROGRAMS src/clustering.py src/input_and_prefilter.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})
install(DIRECTORY launch/ DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/launch)
