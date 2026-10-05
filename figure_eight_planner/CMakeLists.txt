cmake_minimum_required(VERSION 3.0.2)
project(figure_eight_planner)

find_package(catkin REQUIRED COMPONENTS
  mapping
  rospy
  std_msgs
  geometry_msgs
  nav_msgs
  sensor_msgs
  visualization_msgs
  tf
)

catkin_package(
  CATKIN_DEPENDS
    mapping
    rospy
    std_msgs
    geometry_msgs
    nav_msgs
    sensor_msgs
    visualization_msgs
    tf
)

catkin_install_python(PROGRAMS
  scripts/virtual_vcu_udp3.0.4.py
  scripts/bazi_radar_simulator.py
  scripts/jiantu_lidar1.0.3.py
  scripts/bazi_planner_bazi_v4.2.py
  scripts/tracking_udp_v1.2.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
