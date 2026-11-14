cmake_minimum_required(VERSION 3.0.2)
project(ebs_test)

find_package(catkin REQUIRED)

catkin_package()

# Install the python script so rosrun ebs_test ebs_output.py works
install(PROGRAMS scripts/ebs_output.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
