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

###################################
## Python 节点安装（roslaunch type= 依赖）
###################################
catkin_install_python(PROGRAMS
  src/conedetection_to_conearray_node.py
  src/figure_eight_planner_node.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

###################################
## launch 文件安装（$(find bazi)/launch 可用）
###################################
install(DIRECTORY launch/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/launch
)

