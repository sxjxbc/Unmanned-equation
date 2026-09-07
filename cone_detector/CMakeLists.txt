cmake_minimum_required(VERSION 3.0.2)
project(cone_detector)

# ============================================
# 编译选项
# ============================================
if(CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
endif()

# ============================================
# Catkin 宏（必须）
# ============================================
find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
  sensor_msgs
  cv_bridge
  image_transport
  message_generation
)

# ============================================
# 消息定义
# ============================================
add_message_files(
  FILES
  ConeDetection.msg
  ConeDetectionArray.msg
)

# ============================================
# 服务定义（可选，按需启用）
# ============================================
# add_service_files(
#   FILES
#   GetConeStatus.srv
# )

# ============================================
# 动作定义（可选，按需启用）
# ============================================
# add_action_files(
#   FILES
#   DetectCone.action
# )

# ============================================
# 消息/服务/动作 生成（必须）
# ============================================
generate_messages(
  DEPENDENCIES
  std_msgs
  sensor_msgs
)

# ============================================
# Catkin 包声明（必须）
# ============================================
catkin_package(
  CATKIN_DEPENDS
    rospy
    std_msgs
    sensor_msgs
    cv_bridge
    image_transport
    message_runtime
)

# ============================================
# Python 节点安装
# ============================================
catkin_install_python(
  PROGRAMS
    scripts/cone_detector_node.py
    scripts/cone_decision_consumer.py
    scripts/benchmark_detector.py
    scripts/export_to_onnx.py
    scripts/convert_to_tensorrt.py
    scripts/check_trt_env.py
    scripts/verify_int8.py
    scripts/train/train_cone.py
    scripts/train/prepare_dataset.py
    scripts/train/dataset_checker.py
  DESTINATION
    ${CATKIN_PACKAGE_BIN_DESTINATION}
)

# ============================================
# 共享推理模块（NRT_WS camera 包跨包使用）
# camera/yolo_camera.py 通过源目录 sys.path 注入直接引用；
# 此处安装一份到 share 目录，供 rosrun/roslaunch 场景导入。
# ============================================
install(
  DIRECTORY inference/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/python
  OPTIONAL
)

# ============================================
# Launch 文件安装
# ============================================
install(
  DIRECTORY launch/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/launch
)

# ============================================
# 配置文件安装
# ============================================
install(
  DIRECTORY config/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/config
)

# ============================================
# 模型目录（运行时自动创建占位）
# ============================================
install(
  DIRECTORY models/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/models
  OPTIONAL
)

# ============================================
# 文档与指南（轻量化、集成说明）
# ============================================
install(
  DIRECTORY docs/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/docs
  OPTIONAL
)

# ============================================
# 消息头文件（供 C++ 节点使用）
# ============================================
# install(FILES ...)
