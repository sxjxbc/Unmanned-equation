# 直线加速数据链路说明

当前已处理位姿、时间、观测有效性、正常停车保持、中心线几何、配对/质量来源统一及纵向终点进度。
摄像头融合、定位质量校准和AS Finished尚未实现。

## 数据与时间

- 控制每周期从固定来源读取完整位姿快照；launch默认 localization_mode=slam。
- 显式选择 fssim 可用于仿真，不会自动跨来源切换。仿真State必须提供非零header时间戳。
- 原始ROS时间用于测量间隔、雷达对齐及消息年龄；单调时钟用于接收活跃性。
- 零时间戳、非有限数值、倒序/重复位姿、位置及航向跳变被拒绝。
- 速度来自测量时间差，不按配置最高车速裁剪真实记录；前瞻计算仍限幅。
- GPS发布位姿使用对应时刻的IMU航向。当前INS驱动仍以orientation.z承载角度，不是标准四元数。
- 雷达按位姿历史插值或容差内最近样本变换，一帧完成后只发布一次地图。
- 地图保留历史用于关联，但输出仅包含当前帧更新的锥桶；空帧清空可用观测。
- 规划路径有独立 /planning/status：path_stamp、observation_stamp、pose_stamp、valid。
  控制器要求状态与路径时间戳匹配且原始观测未过期，不以新发布时间掩盖旧观测。

## 暂定参数（需真实频率/延迟日志校准）

| 参数 | 默认值 | 含义 |
|---|---|---|
| sensor_timeout / pose_timeout | 0.5秒 | 输入年龄及位姿接收活跃性上限 |
| alignment_tolerance | 0.15秒 | 雷达/GPS与历史样本对齐最大偏差 |
| path_timeout | 0.5秒 | 路径及质量状态接收期限 |
| path_observation_timeout / observation_hold_timeout | 0.5秒 | 原始锥桶观测期限 |
| pose_max_speed | 12m/s | 位姿跳变检测用的宽松运动上界，不是实际限速 |

PoseBuffer另有0.5米位置跳变余量；航向跳变界为0.3弧度+3rad/s×测量间隔。
这些阈值仅用于检测异常，不代表已验证的比赛安全边界。GPS若低频或抖动大，当前默认可能抑制跟踪。
ROS时钟倒退需重新初始化节点；不把回放时钟重置误作车辆正常运动。

## 失效输出约定与限制

输入无效时暂停新的跟踪输出，发布 /control/input_status 并记录CSV。
没有新增自动故障停车；人工独立RES仍是用户选择的异常中止途径。
暂停输出不等于车辆停车：实车VCU断联行为尚未确认，虚拟VCU可能继续执行上一指令。
不得将本次离线检查当作实车停车能力或比赛合规验证。

已触发的正常终点停车优先持续输出raw=2000，输入失效也不能跳过。
仿真同分支输出零驱动，而不是继续固定油门；零驱动不等于已验证的主动制动。
仍需VCU速度、制动及状态反馈完成停稳/AS Finished确认。

后续仍需处理：感知质量/空帧贯通、定位质量和滤波调参。

## 第一步：中心线几何

- 规划器centerline_k/b、控制器local_centerline_map_k/b均存储固定map直线Y=k*X+b。
  控制器原local_centerline_k/b仅为当前车辆坐标下的派生值，CSV含义保持为局部参数。
- 历史线和新测量由规划器在map中滤波；控制器仅转换同一条规划线，不再独立滤波。
- straight_slope_limit约束map道路斜率，不约束车辆偏航。
  规划器拒绝超界拟合并保留原历史线及观测时间，不拼接旧斜率和异常新截距。
- 同一观测时间戳不会反复推动中心线滤波，也不会反复刷新控制器历史保持计时。
- 正常左右配对优先；启动仍需完整左右对。启动后仅在所有候选明确处于历史线同侧时
  才允许单侧兜底，支持单个锥桶。左右分类遵循map Y向右的定义。
- 单侧保留原side_correction_offset=0.6m参数（向未观测侧额外偏移），尚未做实车标定。
  /planning/status新增centerline_source；single_side时pairs=0，不伪装成完整配对。
- 移除会因车辆偏航扭曲道路的局部Y点位裁剪。max_center_offset=1.5m现在约束车辆到
  map中心线的垂直距离：超出时不发布有效路径，而不是把道路裁剪到车辆旁。
  接入路径前6m的渐变、舵角幅值/变化限制、输出使能和观测超时继续保留。
- 道路在车辆前向无法表示（垂直或反向）时不生成该路径/局部目标。

## 第二步：唯一中心线与质量权重

- 仅规划器订阅锥桶并执行配对、拟合和历史保持。控制器删除独立/更宽松的配对及拟合，
  不再订阅/cone_map。
- 正常配对采用固定map中相对历史中心线的左右关系、2.3~3.7m宽度、1.5m纵向差，
  锥桶不复用，同一排不重复计数。候选保留20个再选最多两组，避免控制端先截成4点。
- 配对质量=宽度一致性×同排一致性×两锥桶置信度较小值；优先选择更高质量的候选。
  两组使用选中组中较低的质量，一组额外乘0.6；单侧质量上限0.3。
  该值是保守质量评分，不是标定概率；上游建图置信度仍需在后续步骤完善。
- 已有中心线更新系数=原centerline_alpha×当前质量，低质量观测缓慢修改规划目标。
- /planning/status新增centerline对象：map_k、map_b、pairs、source、confidence、
  observation_stamp，与该条/planned_path的path_stamp和顶层观测时间匹配。
  confidence为原始质量，控制器每周期乘max(0,1-观测年龄/path_observation_timeout)。
- 控制器局部纠偏权重=配置上限×当前有效质量：两组上限0.65，一组0.45，单侧0.20。
  例如两组有效质量0.4时权重0.26；新单侧质量0.3时权重0.06。
  路径渐变与中心线目标仍是不同跟踪目标，但共享同一道路估计和原始观测。
- 重新发布路径不刷新旧观测年龄；质量为零、过期、缺字段、非有限值、类型错误、
  配对数量/来源矛盾或路径时间不匹配均不能用于新的跟踪输出。
- CSV追加centerline_source、centerline_confidence（年龄衰减后）、centerline_blend_weight、
  observation_age；原有列次序保留。
- launch删除控制器的cone_confidence_threshold、local_cone_*、max_local_cones、pair_*、
  local_max_slope及local_timeout，新增local_blend_single_side。配对参数仅在规划器配置。
- 规划器和控制器必须同步部署。旧版status没有centerline字段时，新控制器会抑制跟踪；
  不能只升级控制器或回放旧规划状态作为新有效输入。

## 第三步：独立纵向终点

- 进度为相对固定起点、沿固定map轴的有符号投影：
  s=(x-start_x)*cos(finish_axis_yaw)+(y-start_y)*sin(finish_axis_yaw)。
  不累加相邻点距离；横向摆动不增程，倒退减程，往返不会重复计入75米。
- 控制周期先取得有效定位并更新终点，再检查路径/观测；路径尚未建立或失效时仍更新进度。
  达到终点后在本周期进入停车保持，跳过路径查询、目标选择和PD计算。
- finish_start_mode默认first_valid_pose：首次有效控制定位快照锁定起点，与路径启动无关，
  后续不会重新定基准。此模式不是物理起跑信号；控制器若在车辆行驶中才启动，会从当时位置起算。
- configured模式使用finish_start_x/y明确指定map中的起跑线参考点，首次定位不会覆盖它。
  参考点必须与/vehicle_pose使用同一map、同一车辆定位参考点；地图重建后需重新确认坐标。
  finish_axis_yaw为固定赛道方向，单位弧度，默认0表示map +X；不随车辆当前航向或中心线滤波改变。
- 每个有效测量时间戳仅更新一次。过期、缺失、跳变拒绝的定位不推进终点，也不重设起点。
- finish_distance默认75m。finish_confirm_samples默认1，首次有效越线测量即锁定正常停车；
  可配置为2或更大以要求多次新测量越线。重复控制周期不算新测量，回到线前或定位失效清空待确认计数。
  增加确认次数会增加停车触发延迟，需结合实际定位频率和停车距离验证；默认未增加此延迟。
- 停车锁定后即使位置退回、定位失效或路径失效，也持续发送正常停车raw=2000。
  FSSIM零驱动按固定localization_mode选择输出，不因control_mode变为none而中断。
  enable_vehicle_output=false仍禁止车辆输出；停车状态和进度可离线检查。
- /control/input_status及CSV新增finish_progress、finish_reference_mode、finish_reference_x/y、
  finish_axis_yaw、finish_pose_stamp、finish_confirm_count、finish_reference_stamp。
  未锁定的参考坐标为null；configured参考不由传感器建立，reference_stamp为null。
  CSV旧total_distance列保留位置，但含义已改为有符号纵向进度，需同步调整旧日志分析程序。
- 普通launch和validation.launch均支持上述finish_*参数。回放配置示例（车辆输出强制禁用）：

```text
roslaunch acceleration_event validation.launch finish_start_mode:=configured finish_start_x:=0.0 finish_start_y:=0.0 finish_axis_yaw:=0.0
```

以上参数是几何触发逻辑，不提供定位精度保证、主动制动距离或停稳反馈；仍不能宣称已完成AS Finished。

## 自检

在工作区运行：

```text
python -B -m unittest discover -s src/acceleration_event/tests -v
```

测试提取实际节点方法并使用内存替身，不初始化ROS、不创建socket、不发送车辆指令。
当前主机不能执行ROS/catkin集成；已新增包声明和安装规则，但其编译/部署需在目标ROS环境验证。

## 原版本环境验证入口

旧构建记录为ROS Melodic、/usr/bin/python2；car历史系统记录为Ubuntu18.04.6。
当前未升级或安装任何依赖。实时版本须运行scripts/check_original_environment.sh核对。
该脚本只读库存、编译语法并执行离线函数测试；不启动ROS节点、不构建、不发指令。
tests/test_buffer_python2.py、test_centerline_geometry.py、test_finish_progress.py面向原始Python2兼容编写；
后两者需要已有NumPy，已接入环境检查脚本。test_data_quality.py是主机Python3的扩展测试。
本次在主机Python3执行78项离线测试通过；原始Python2和ROS/catkin运行尚未验证。
Python2的接收计时使用librt.so.1中的CLOCK_MONOTONIC，无pip依赖。

正常launch现默认enable_vehicle_output=false；禁用时不初始化UDP socket，不发布/control/steering或/fssim/cmd。
质量状态、规划及CSV仍保留。需要运行车辆时必须显式选择enable_vehicle_output:=true。
回放专用launch/validation.launch强制禁用车辆输出与OpenCV窗口，默认use_sim_time=true。
回放时只启动原始输入话题，使用rosbag play --clock，避免同时回放旧的规划/控制输出话题。
验证开关只能约束本包，不能禁用其他车辆控制节点。


## 感知输出质量修复

聚类同时输出筛选后的 `/clustered_points` (PoseArray) 和 `/lidar/cones` (mapping/ConeArray)。后者坐标为雷达 velodyne，保留输入观测时间，confidence 为当前点数/距离启发式得分，不是标定概率。直线建图默认 `cone_input_mode=quality`，只订阅后者；旧录包可显式设 `legacy`，此模式没有检测置信度。同步部署 lidar_nodes 与 acceleration_event，并构建 mapping 消息依赖。八字代码未修改，但共用旧话题收到的簇现在也经过筛选。

默认筛选点数至少5、可见高度跨度至多0.60 m、XY包围盒等效半径至多0.30 m、距离至多30 m、置信度至少0.25。高度/半径无下限，以保留稀疏或遮挡观测；这些条件不能保证聚类一定是锥筒，阈值需要真实数据标定。范围仍由原预处理0至6 m、横向正负2 m限制，未扩大。

空障碍物点云和全部簇被筛除均输出原时间戳的空消息。非有限点在聚类前移除。非法非空质量消息不会被当成空场景清图。当前检测置信度对地图已有置信度取上限限制，不替代后续待修复的观测计数和同帧关联。

发布器在订阅器前初始化，订阅队列为1；移除逐簇大量日志，每秒输出帧处理耗时，超过100 ms告警。耗时包括算法和发布调用，不等于端到端延迟；Python2时钟回退为time.time。未替换二次方距离矩阵，未验证实测吞吐量。

离线测试现为83项；新增覆盖尺寸/非有限值筛选、空点云输出、消息时间戳与置信度，以及建图质量入口。ROS Melodic/Python2构建、真实点云回放和车辆输出未在本机验证。


## 定位质量、稳定初始化与地图关联修复

GPS接收默认要求NavSat状态为0/1/2，有已知的水平协方差，两个水平对角方差均大于0且不超过0.25平方米；水平协方差须对称且满足半正定约束，全部元素有限。未知协方差默认拒绝，旧录包可显式设置 `gps_allow_unknown_covariance:=true`；该开关不会接收NO_FIX或超限的已知协方差。拒绝后不刷新车辆位姿，下游沿原有0.5秒时效约束停止接受旧位姿，不新增车辆自动故障停车。

初始化仅由有效GPS及时间对齐IMU样本推进，连续覆盖3秒、至少10个样本、间隔不超过0.5秒；所有点相对均值的水平距离不超过0.30米，航向相对圆周均值偏差不超过3度。无效/中断/运动或偏航变化重置窗口。使用平均经纬度与圆周平均航向建立原点，初始化完成当帧计算相对于平均原点的实际位姿，不强制置零。需停车完成初始化；阈值是初始配置，须按实际定位噪声验证。

串口驱动无可信质量字段协议，现默认NO_FIX与UNKNOWN协方差，不再发布伪造有效状态/固定方差。因此默认严格链路无法用此驱动完成初始化，必须接入真实质量来源。仅为兼容旧数据，可在驱动显式设 `assume_gps_fix:=true`（告警提示人为假设），并在建图显式设 `gps_allow_unknown_covariance:=true`；这不是生产环境质量验证。未猜测帧校验或设备状态字节。驱动同时修正Python2/3字节读取、展开日志目录，并为同一串口包的IMU/GPS使用同一发布时间戳（仍非设备采样时间）。

地图先清理过期对象，再基于帧前地图构造距离候选并作确定性最近距离优先的一对一关联，每帧每对象最多更新一次。未匹配但靠近已有对象的碎片被抑制，新簇优先保留高质量观测；阈值0.8米内真实不同锥筒仍可能被抑制，需回放评估。该策略不是全局最优分配算法。地图置信度=当前检测质量乘独立帧支持，首帧支持0.4、每新帧增加0.15、最大1；同质量首帧到第二帧不再下降，同时间戳不增加支持。低质量首帧可能暂未达到规划阈值，符合保守启动行为。

修复PoseBuffer精确时间戳命中先于插值检查；修复validation.launch对cone_input_mode转发缺失。新增原ROS/Python2环境脚本对感知契约及定位地图离线测试的执行。当前主机99项离线测试通过，未执行原环境构建或实车验证。
