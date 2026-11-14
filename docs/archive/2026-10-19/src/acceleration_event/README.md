# 直线加速：雷达持续纠偏融合版

更新日期：2026-10-13。

本模块用于无人赛车直线加速。摄像头不可用时，直接利用当前雷达锥桶估计车辆相对道路的偏差，生成前瞻目标，再通过现有 PD 和 CAN 接口控制转向。车辆配置沿用本项目版本，借鉴北理双边界检测思路，没有整套移植北理工程。

**推荐入口：`acceleration_lidar.launch`。默认是影子模式，不发送车辆指令。**

当前状态：源码实现和自检已完成，34 项离线测试通过；尚未完成车机 ROS/Python 2.7 构建、真实录包回放和实车验收。测试通过不代表已具备实车运行验收结论。

## 1. 工作原理

```text
当前雷达锥桶 /clustered_points
          ↓
原外参转换、车辆本体回波过滤
          ↓
双边界估计 → 当前车身系中心线
          ↑
IMU + 速度信息：时间对齐、短时运动补偿
          ↓
原前瞻距离 → 原目标点误差 → 原PD → 原转向限幅 → 原CAN

现有定位 /vehicle_pose → 速度估计、固定赛道方向上的纵向进度
历史锥桶地图          → 显示和诊断，不参与新横向控制
```

- 起步通过连续多个有效雷达帧确认赛道参考，重复消息不会增加确认次数。
- 两排有效锥桶用于更新道路方向和横向位置。
- 只有一组左右锥桶时，主要更新横向位置，方向由已有估计和IMU运动补偿维持。
- 单侧锥桶仅在边界身份可信且最后双侧观测仍有效时短时使用。
- 短暂缺帧可以预测，但不会刷新旧观测的原始时间。
- 固定的是任务参考，车辆相对道路的误差在行驶中持续更新。

新链路仍使用“前瞻目标点的横向位置与方位角”作为PD输入，没有直接改成另一种道路航向误差定义。GPS横向位置不直接决定雷达中心线，但GPS仍间接影响速度估计；当前系统也仍要求有效定位来计算纵向进度。

## 2. 主要文件

| 文件 | 用途 |
|---|---|
| `launch/acceleration_lidar.launch` | 新版启动入口，默认影子模式 |
| `src/genzong_lidar.py` | 当前帧雷达控制节点、时序处理、停车和诊断 |
| `src/lidar_road.py` | 双边界几何、运动历史插值、纵向进度 |
| `src/genzong_realtime.py` | 新版复用其中的PD、滤波、前瞻和CAN实现 |
| `src/jiantu.py` | 现有定位与地图节点，为新版提供车辆位姿 |
| `tests/test_lidar_fusion.py` | 几何、运动、终点和配置一致性测试 |
| `tests/test_lidar_fusion_node.py` | 回调、控制流程、输出隔离及停车测试 |

保留的 `acceleration_event.launch` 用于原配置对照与兼容，`zhixian_realtime.py` 是现有构建安装规则引用的兼容脚本。新入口不会启动该规划器，也不订阅它的 `/planned_path`。不要把保留的兼容入口误当作新版本启动方式。

## 3. 环境与依赖

沿用车机原环境：Ubuntu 18.04、ROS Melodic、Python 2.7；已有记录中的NumPy为1.13.3、OpenCV为3.2.0。记录值不是此次实测环境结果，不要求为本次融合升级系统库。

需要：

- ROS：rospy、tf、geometry_msgs、sensor_msgs、nav_msgs、std_msgs、visualization_msgs，以及本项目 `mapping` 生成的消息。
- 既有驱动/感知：`imu`、`lidar_nodes`、地面分割节点、Velodyne驱动。
- Python：NumPy、OpenCV、pyserial、ros_numpy、scikit-learn及其对应依赖，按原车机已验证版本使用。

新几何核心只用Python标准库，但完整节点和既有感知仍需要上述依赖。交接源码包不包含ROS、Velodyne驱动、Python依赖安装包或车机设备配置。

## 4. 构建

以下命令在目标Linux车机执行。假设工作区为 `~/nrt_ws`，请替换为实际路径。

```bash
source /opt/ros/melodic/setup.bash
cd ~/nrt_ws
catkin_make -j2
source devel/setup.bash
rospack find acceleration_event
```

若从压缩包建立全新工作区，应将其中的 `src` 放到工作区下，然后先初始化catkin：

```bash
source /opt/ros/melodic/setup.bash
cd ~/nrt_ws/src
catkin_init_workspace
cd ..
catkin_make -j2
source devel/setup.bash
```

交接包不包含Windows同步目录中的 `build/devel`，也不包含写死 `/opt/ros/...` 路径的伪CMake文件。已有工作区不需要重复初始化。确认 `rospack find` 指向本次源码，避免启动到另一份同名包。

## 5. 启动

### 5.1 传感器

先按车队已经跑通的方式启动雷达和GPS/IMU。新版启动文件不会自动启动这些驱动。

检查输入：

```bash
rostopic hz /clustered_points
rostopic hz /imu_data
rostopic echo -n 1 /clustered_points/header
```

### 5.2 默认影子模式

尚未运行建图定位节点时：

```bash
roslaunch acceleration_event acceleration_lidar.launch
```

已有 `/vehicle_pose` 发布者时，避免重复启动建图：

```bash
roslaunch acceleration_event acceleration_lidar.launch start_mapping:=false
```

影子模式计算目标和候选转角、记录日志，不打开CAN socket。它不会停止其他已经运行的控制器。原建图节点保留原可视化行为，启动它可能需要桌面环境。

### 5.3 实车控制入口

完成影子数据对照、转向方向及停车编码验证后，停止其他直线/八字控制器，确保只有一个车辆输出控制器。通过原VCU有效方式限制测试车速，然后使用：

```bash
roslaunch acceleration_event acceleration_lidar.launch \
  tracking_mode:=control enable_vehicle_output:=true
```

已有建图运行时追加 `start_mapping:=false`。`tracking_mode=control` 与 `enable_vehicle_output=true` 必须同时满足才输出车辆指令；只有其中一个不能开启实车输出。

## 6. 配置约定

原车辆参数沿用当前项目的实际生效值，不采用北理参数：

| 项目 | 当前启动配置 |
|---|---|
| 车道宽度 | 3.0 m |
| 终点距离 | 75.0 m |
| 前瞻范围 | 2.5～4.0 m |
| 前瞻速度系数 | 0.35 |
| 横向PD | Kp=0.60，Kd=0.05 |
| 目标方位角PD | Kp=0.20，Kd=0.01 |
| 转角限幅 | 0.50 rad |
| 单周期转角变化限制 | 0.012 rad |
| 雷达有效观测保留期 | 0.35 s |
| CAN网关 | 192.168.0.7:20005，本地端口11312 |
| CAN帧 | 0x210；零转角raw=500，停车raw=2000 |

原launch有重复赋值，上表取后面的有效值。新版保留这些值，测试对照原配置检查一致性。

雷达外参：x=1.76618 m、y=-0.11304 m、Y符号=-1、安装偏航0.5°。若重新标定，需要同步修改新控制器和建图节点的参数；两者私有参数不会自动同步。

新增融合参数：

| 参数 | 默认值 | 含义 |
|---|---|---|
| `fusion_init_frames` | 3 | 起步有效帧确认数 |
| `fusion_angle_gate` | 0.35 rad | 运动补偿后的方向一致性门限 |
| `fusion_offset_gate` | 0.5 m | 中心偏移一致性门限 |
| `fusion_fit_residual` | 0.25 m | 边界拟合残差门限 |
| `fusion_pose_timeout` | 0.5 s | 定位有效期 |
| `fusion_imu_timeout` | 0.15 s | IMU有效期和控制外推时间上限 |
| `fusion_lidar_frame` | velodyne | 雷达输入坐标系 |
| `fusion_log_enabled` | true | JSONL文件记录 |

这些新门限需要用真实消息频率和延迟验证。保留的旧地图混合权重、地图置信度和局部斜率上限不参与新版横向估计。`fixed_speed` 不代表已经向VCU下发对应实车速度。

## 7. 消息与坐标

| 话题 | 类型 | 用途 |
|---|---|---|
| `/clustered_points` | geometry_msgs/PoseArray | 雷达当前帧锥桶 |
| `/imu_data` | sensor_msgs/Imu | 原协议航向数据 |
| `/vehicle_pose` | geometry_msgs/PoseStamped | 地图系位姿 |
| `/control/lidar_status` | std_msgs/String | JSON诊断状态 |
| `/control/lidar_candidate_steering` | std_msgs/Float64 | 候选转角，弧度；影子模式也发布 |
| `/control/steering` | std_msgs/Float64 | 仅车辆输出开启时发布 |

雷达输入为 `velodyne` 或 `/velodyne`，X前Y左；本项目车辆内部约定X前Y右。通过原外参完成转换，不能仅改frame_id标签代替坐标变换。

**当前IMU驱动把航向角度数放在 `orientation.z`，不是标准四元数。** 新版保留这一协议。`/vehicle_pose` 则必须使用有效四元数，frame_id为map。

所有输入要求非零、有效的ROS时间戳。检测按采集时间匹配运动历史；消息重复发布不能延长观测寿命。

## 8. 日志与排查

```bash
rostopic echo /control/lidar_status
rostopic hz /vehicle_pose
```

默认文件日志：`/tmp/path_tracking_logs/lidar_fusion_*.jsonl`。建议试车时指定持久化目录：

```bash
roslaunch acceleration_event acceleration_lidar.launch \
  log_dir:=$HOME/nrt_logs/lidar_fusion
```

重点字段：`started`、`input_reason`、`observation_age`、`pose_stamp`、`imu_stamp`、`pairs`、`estimate_kind`、`target`、`steering`、`progress`、`stop_reason`、`udp_sent`。

| 状态 | 含义与处理 |
|---|---|
| `waiting_time_alignment` | IMU历史尚未覆盖检测时间；核对时钟、延迟和输入频率 |
| `geometry_rejected` | 空帧、支持不足或配对/边界不一致；查看原始锥桶 |
| `lidar_frame` / `lidar_stamp` | 输入坐标系或时间戳不符合约定 |
| `road_stale` | 最后有效道路观测过期 |
| `pose_stale` / `imu_stale` | 对应状态输入过期 |
| `motion_gap` | 运动历史中断，旧预测坐标原点不可继续使用 |
| `clock_rewind` | ROS时间倒退 |
| `can_send_failed` | 发送失败，锁存停车并持续尝试发送停车编码 |
| `finish` | 固定轴纵向进度达到终点 |

`started=true`表示算法开始闭环计算，不代表影子模式向车辆发指令。停车触发后锁存到节点重启，不会因为数据恢复就自行继续行驶。终点判断先于道路可用性判断；缺锥桶不能跳过已触发的停车。

## 9. 录包、回放和验收

记录：

```bash
rosbag record -O lidar_fusion_check.bag /GPS_data /imu_data \
  /clustered_points /vehicle_pose /control/lidar_status \
  /control/lidar_candidate_steering /control/steering
```

回放使用独立ROS会话，不运行实车输出控制器。使用记录的车辆位姿，关闭重复建图：

```bash
rosparam set /use_sim_time true
roslaunch acceleration_event acceleration_lidar.launch start_mapping:=false
# 另一终端
rosbag play --clock lidar_fusion_check.bag --topics /imu_data /clustered_points /vehicle_pose
```

倒带或重新播放前重启节点；在线运行前停止回放，将 `/use_sim_time` 恢复为false。不要回放历史控制输出话题。

验证顺序：

1. 原地检查居中、偏左、偏右和小角度偏航时的雷达目标方向。
2. 旁路观察真实行驶中目标是否持续更新。
3. 低速接入新控制，检查目标、转向命令和实际车辆响应是否一致。
4. 检查短缺帧、持续失效、时间异常及停车锁存。
5. 测量纵向终点与实际停车位置，再逐步提高速度。

内部Y右约定下，车辆偏左应看到道路中心目标在右侧，即目标横向值为正；执行器转向正负仍须按本车标定核对。

## 10. 离线测试与限制

在工作区根目录执行：

```bash
python -m unittest discover -s src/acceleration_event/tests -p 'test_lidar_fusion*.py' -v
```

最近一次自检：34项通过，Python 3语法检查通过，运行环境为本机Python 3.14。测试覆盖几何、连续纠偏、时间插值、输入异常、影子无socket、原PD/CAN复用、启动时间差、关闭后的回调、发送失败和停车锁存。

测试使用ROS传输替身，未启动真实ROS或发送车辆指令。代码按Python 2.7语法编写，但尚未在目标Python 2.7环境验证。

已知限制：

- 原历史地图显示问题未修复，新版横向控制通过绕过历史地图降低其影响。
- 定位来源仍是现有链路，GPS漂移和错误质量信息不能仅靠本模块完全解决。
- 起点在控制器就绪时确定，不是独立发车信号；需与实车起跑位置一致。
- 当前近场感知可能经常只有一排锥桶，方向可靠性和配对误差必须实测。
- 尚未接入真实轮角和停稳反馈；发送停车编码不等于确认已经制动停稳。
- 新入口不要与另一个实车控制器同时输出。

后续任何实车调参，应记录源码版本、完整参数、rosbag和诊断日志，保证同事能够复现结果。
