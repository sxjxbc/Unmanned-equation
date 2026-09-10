cmake_minimum_required(VERSION 3.0.2)
project(bazi)

find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
  geometry_msgs
  nav_msgs
  message_generation
  msgs
)



catkin_package(
  CATKIN_DEPENDS
    rospy
    std_msgs
    geometry_msgs
    nav_msgs
    message_runtime
)


catkin_install_python(PROGRAMS
  src/conedetection_to_conearray_node.py
  src/figure_eight_planner_node.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)


install(DIRECTORY launch/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/launch
)


