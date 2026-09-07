cmake_minimum_required(VERSION 3.1)
project(cone_traditional_perception)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_FLAGS "-std=c++11 ${CMAKE_CXX_FLAGS}")

find_package(catkin REQUIRED COMPONENTS
  roscpp
  cv_bridge
  image_transport
  msgs
)

find_package(OpenCV 3 REQUIRED)

catkin_package(
  CATKIN_DEPENDS roscpp cv_bridge msgs
)

include_directories(
  ${catkin_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
)

add_executable(cone_perception_node src/stereo_traditional.cpp)
target_link_libraries(cone_perception_node
  ${catkin_LIBRARIES}
  ${OpenCV_LIBS}
)
add_dependencies(cone_perception_node ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

add_executable(test_image_node src/test_image.cpp)
target_link_libraries(test_image_node
  ${catkin_LIBRARIES}
  ${OpenCV_LIBS}
)
add_dependencies(test_image_node ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
