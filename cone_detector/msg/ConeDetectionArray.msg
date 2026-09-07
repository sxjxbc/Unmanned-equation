# 一帧图像中所有锥桶检测结果的数组
# 每帧发布一次，作为 /cone_detector/detections 话题的消息类型

Header  header         # ROS 标准头（含时间戳）

ConeDetection[] detections  # 本帧所有检测结果列表

# 帧级统计与状态
uint32          num_detections   # 本帧检测数量
float32         inference_time   # 本次推理耗时（秒）
bool            has_cones       # 是否有检测结果
string          dominant_decision  # 本帧主导决策（置信度最高的决策）

# 决策优先级说明:
#   红色锥桶 → 决策 A（最高优先级）
#   黄色锥桶 → 决策 B（中优先级）
#   蓝色锥桶 → 决策 C（低优先级）
# 无检测时 dominant_decision = "default"
