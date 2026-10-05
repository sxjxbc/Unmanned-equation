cmake_minimum_required(VERSION 3.0.2)
project(acceleration_event)
find_package(catkin REQUIRED COMPONENTS rospy std_msgs geometry_msgs nav_msgs sensor_msgs tf mapping)
catkin_package()
catkin_install_python(PROGRAMS
  src/jiantu.py
  src/zhixian_realtime.py
  src/genzong_realtime.py
  src/data_quality.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
install(DIRECTORY launch/ DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/launch)
