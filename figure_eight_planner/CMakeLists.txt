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

