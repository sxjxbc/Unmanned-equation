# 单个锥桶检测结果
# 对应一个检测框

string  color          # 锥桶颜色: "red" / "blue" / "yellow"
string  decision       # 决策信号: "TURN_LEFT" / "TURN_RIGHT" / "STOP"
                       #   red    → TURN_LEFT  （红色锥桶 → 左转）
                       #   blue   → TURN_RIGHT （蓝色锥桶 → 右转）
                       #   yellow → STOP       （黄色锥桶 → 停止）
float32 confidence     # 置信度 [0.0, 1.0]

# 检测框坐标（原图分辨率，单位：像素）
uint32  x_min          # 左上角 X
uint32  y_min          # 左上角 Y
uint32  x_max          # 右下角 X
uint32  y_max          # 右下角 Y

# 中心点位置（原图分辨率）
float32 center_x       # 归一化中心 X [0.0, 1.0]（相对图像宽度）
float32 center_y       # 归一化中心 Y [0.0, 1.0]（相对图像高度）
