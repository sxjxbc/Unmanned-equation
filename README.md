# nrt_ws 无人赛车 ROS 1 工作区

本项目包含直线加速、八字绕环，以及雷达感知、GPS/IMU输入和诊断工具。当前维护重点是 **`acceleration_event` 的旧版直线链路**：地图锥桶 → 固定赛道中心线 → 平滑路径 → 前瞻目标 → PD转向。另有独立的当前帧雷达融合链路，两者不能混用启动方式或验证结论。

文档核对日期：2026-10-19。本文负责安装、运行和排查；算法与状态机见 [项目逻辑分析](docs/项目逻辑分析.md)。

## 1. 当前进度与使用边界

| 检查 | 已保存的结果 | 不代表什么 |
|---|---|---|
| 选定主机回归 | 81项通过 | 不是整个历史测试目录全部通过 |
| car虚拟机真实构建 | `mapping`、`acceleration_event`通过 | 未覆盖整个雷达驱动工作区 |
| 真实ROS无输出启动 | 建图、规划、控制可通信并进入running | 不是实车执行验收 |
| 真实ROS持续通信/故障注入 | 7项通过，候选输出约20Hz | 未测实际轮角、VCU制动与停稳 |
| 离线完整闭环 | 简单偏移/偏航等场景通过；组合偏差及高速延迟存在失败 | 尚不能保证稳定跑完赛道 |

上述真实ROS集成结果针对旧版链路。融合链路有独立离线测试，不能直接套用旧版的7项ROS集成结论。轴距、软件转角到真实前轮转角的对应关系、执行延迟尚未标定。

**优先使用无输出入口验证。旧版实车入口默认启用车辆输出；`running`只说明软件进入控制，不证明车辆实际执行正确。**

## 2. 目录与入口

| 路径 | 用途 |
|---|---|
| [src/acceleration_event](src/acceleration_event) | 两种直线链路、诊断、验证工具 |
| [src/mapping](src/mapping) | 生成Cone/ConeArray等消息；不是独立定位算法库 |
| [src/imu](src/imu) | 串口GPS/IMU驱动，项目专用航向格式 |
| [src/lidar](src/lidar) | 点云预处理、地面分割、聚类及第三方依赖源码 |
| [src/figure_eight_planner](src/figure_eight_planner) | 八字建图、规划、追踪与虚拟VCU工具 |
| [docs/项目逻辑分析.md](docs/项目逻辑分析.md) | 当前实现、参数、话题、状态机和限制 |
| [exports](exports) | 原始验证报告、控制CSV、模型试验及源码归档，不参与运行 |
| [backups](backups) | 各轮修改前备份，不应加入ROS包搜索路径 |
| `build/`、`devel/` | 旧构建产物；跨机器/系统不要直接复用 |
| `tests/`、`src/acceleration_event/tests/` | 测试；后者包含不同历史阶段的测试，需选用验证入口 |
| `tmp/`、`.test_dependencies/` | 历史分析/主机辅助文件，不是车机依赖安装方案 |

`src/introduce.docx`及第三方README/许可证保留，雷达改造方案作为历史说明归档。技能目录和编辑器配置不属于车辆节点，未清理。

## 3. 环境与依赖

最近一次真实验证：Ubuntu 18.04.6 x86_64、ROS Melodic 1.14.13、Python 2.7.17、NumPy 1.13.3、OpenCV 3.2.0。它们是已测组合，不应盲目升级或把Windows Python 3替换为车机解释器。

- 核心ROS包：catkin、rospy、tf、geometry_msgs、nav_msgs、sensor_msgs、std_msgs、visualization_msgs、消息生成/运行时。
- Python：NumPy、OpenCV、PyYAML；串口驱动另外需要pyserial。实际雷达感知还需要其源码声明的ros_numpy、SciPy/scikit-learn等依赖及Velodyne驱动，按目标机检查。
- `track_frame.py`、`legacy_run_guard.py`、`lidar_road.py`的几何核心使用标准库，但完整节点仍有ROS/NumPy等导入。
- `fssim_common`是可选仿真消息依赖；旧控制器会优先使用活跃FSSIM状态，测试/实车不要接入意外的仿真状态发布者。
- `mapping`只生成消息；当前CMake不再导出未构建的`mapping`库或占位`system_lib`。

## 4. Linux构建与源码同步

### 已有验证工作区

car虚拟机已验证副本为 **`/home/car/nrt_validation_20261019`**。原`~/nrt_ws`未被覆盖。宿主源码为`D:\bh_work\nrt_ws`；上次验证时VMware共享目录未挂载，源码经SSH同步，宿主修改不会自动更新该副本。

以下均为Ubuntu终端命令，使用前确认目录是当前源码副本：

```bash
source /opt/ros/melodic/setup.bash
cd ~/nrt_validation_20261019
python src/acceleration_event/scripts/prepare_linux_sources.py
catkin_make -j2
source devel/setup.bash
rospack find acceleration_event
rospack find mapping
rosmsg show mapping/ConeArray
```

`prepare_linux_sources.py`会修改该Linux包内Python文件的CRLF与执行权限，应在复制出的Linux工作区执行；保留传输归档，不直接作用于需要保持原字节的共享源。

### 新建最小业务验证工作区

只复制两个包，避免把未配置的硬件驱动一起带入首次检查。以下目标目录须为新目录，源路径按实际挂载/传输位置替换：

```bash
source /opt/ros/melodic/setup.bash
mkdir -p ~/nrt_validation_new/src
cp -a /实际源码位置/nrt_ws/src/mapping ~/nrt_validation_new/src/
cp -a /实际源码位置/nrt_ws/src/acceleration_event ~/nrt_validation_new/src/
cd ~/nrt_validation_new
ln -s /opt/ros/melodic/share/catkin/cmake/toplevel.cmake src/CMakeLists.txt
python src/acceleration_event/scripts/prepare_linux_sources.py
catkin_make -j2
source devel/setup.bash
```

不要复制Windows的`build/devel`，也不要复制本仓库`src/CMakeLists.txt`：该文件当前是Linux路径文本，不是有效catkin顶层符号链接。全工作区构建还需独立解决雷达/八字包的外部依赖；两包构建通过不能替代它。

helper通过CMake复制到`devel/lib/acceleration_event`。源码改动后重新构建；用`rospack find`与Python模块`__file__`核对路径，防止source到旧工作区。仅复制文件但不构建可能继续导入旧helper。

## 5. 首选：旧版无输出验证

在已构建并source的Linux工作区执行，脚本自建独立本地master并清理测试进程，无需另起roscore：

```bash
python src/acceleration_event/scripts/smoke_ros_validation.py
```

预期：打印`PASS: synthetic stationary startup reached running; vehicle command topics absent`和日志目录。它执行真实建图/规划/控制，合成输入静止，输出与显示均关闭。

持续通信与故障检查（结果目录必须新建，不能重用已有场景目录）：

```bash
python src/acceleration_event/scripts/ros_fault_validation.py --output ~/nrt_ros_fault_new
```

七项包括：20秒持续运行、启动前缺GPS后恢复、0.2秒空锥桶帧、持续道路/GPS/IMU中断、过期IMU时间戳。查看输出目录的`report.json`及逐场景日志。

也可手动启动：

```bash
roslaunch acceleration_event validation.launch synthetic_inputs:=true
```

该手动入口不会为你建立独立隔离master；仅在测试ROS环境使用，不与真实传感器混用。`synthetic_inputs`默认false，`use_sim_time`默认false；回放若启用仿真时间，需同时提供`/clock`。

## 6. 三种业务入口：不要混用

| 入口 | 实际启动内容 | 输出默认值与注意事项 |
|---|---|---|
| `validation.launch` | 旧版建图+规划+控制；合成源可选 | 强制关闭输出与显示，首选集成检查 |
| `live_mapping_check.launch` | 旧版链路，真实驱动可选 | 关闭输出/显示；`start_sensors`默认false |
| `acceleration_event.launch` | 旧版地图/固定赛道路径链路 | **输出默认true**、显示默认true；不自动启动传感器 |
| `acceleration_lidar.launch` | 当前帧雷达融合控制器，建图可选 | 默认shadow且输出false；默认建图仍有显示 |
| `figure_eight_planner/bazi_realtime.launch` | 八字建图、规划、UDP追踪 | 独立赛项，不能与直线控制器同时输出 |

在已有测试传感器/回放输入时观察旧版，不开启车辆输出：

```bash
roslaunch acceleration_event acceleration_event.launch enable_vehicle_output:=false enable_visualization:=false
```

观察融合链路：

```bash
roslaunch acceleration_event acceleration_lidar.launch
# 已有唯一的/vehicle_pose发布者时，使用start_mapping:=false
```

融合版只有`tracking_mode=control`且`enable_vehicle_output=true`才启用车辆输出；这不是本README的首次验证步骤。旧版无输出时只发布候选转角且不创建CAN客户端，退出/故障分支也不发送CAN。两种模式均不会关闭其他控制器。

传感器入口是 [lidar.launch](src/lidar/src/lidar_nodes/launch/lidar.launch)和 [daoyuan_simple.py](src/imu/src/daoyuan_simple.py)，实际串口、网络及驱动配置需另行确认。`start_mapping.sh`会检查固定NumPy/OpenCV版本，默认请求启动硬件驱动；其旧提示含GPS质量旁路描述，但当前建图没有对应参数开关，不应据此认定质量校验已完善。

八字`bazi_sim.launch`引用`bazi_closed_loop`包，而当前目录包名为`figure_eight_planner`，不能把它当作已通过的仿真入口。八字及完整雷达链路未纳入这轮真实ROS验收。

## 7. 状态、日志与自动诊断

```bash
rostopic echo /control/legacy_status
rostopic echo /planning/legacy_road_status
# 融合版：rostopic echo /control/lidar_status
```

旧版`waiting`可恢复，查看`waiting_for`；`running`表示已进入控制；`stopped`锁存故障，查看`stop_reason`。CAN初始化失败时等待并重试，`output_ready`只说明输出被禁用或本地客户端已初始化，**不是VCU回执**。已进入控制后，即使车仍在起点，也开始检查超时。

旁路记录器（只监听，不接管、不停车）：

```bash
roslaunch acceleration_event run_diagnostics.launch mode:=legacy
# 融合版改为mode:=lidar
# 发车前开启报警判断，停车后把true改成false
rostopic pub -1 /diagnostics/run_active std_msgs/Bool 'data: true'
```

诊断一开始就记录数据，默认不报警。`/diagnostics/run_active`只控制诊断判断，不控制车辆发车。默认记录`/control/steering`；旧版无输出或融合shadow没有该话题，开启诊断可能出现`MISSING_STEERING`，不应把它当成节点启动失败。

| 记录 | 默认位置/接口 |
|---|---|
| 旧控制CSV | `/tmp/path_tracking_logs/control_log_*.csv` |
| 融合JSONL | `/tmp/path_tracking_logs/lidar_fusion_*.jsonl` |
| 旁路证据/摘要 | `/tmp/run_diagnostics/日期时间/evidence.jsonl`、`report.json` |
| 无输出候选转角 | `/control/legacy_candidate_steering`或`/control/lidar_candidate_steering` |

`/tmp`可能随重启清理，及时复制整目录，或通过各入口支持的`log_dir`改为持久目录。定位静止检测是疑点，不是独立运动证据；正常停车也会触发。

## 8. 离线工具与已有证据

Windows/主机Python3，在仓库根目录执行：

```powershell
python src/acceleration_event/scripts/validate_track_frame.py --output exports/check_new
python src/acceleration_event/scripts/legacy_loop_validation.py --output exports/loop_new
python src/acceleration_event/scripts/analyze_loop_response.py --output exports/response_new
```

第一个入口是选定测试集合，默认比较基线依赖`backups/track_frame_20261015_221841/acceleration_event`；移走备份后需传入有效`--baseline`。闭环工具使用未标定的简单车辆模型，当前组合偏差有失败，返回非零退出码是保留的失败证据。对照参数仅用于内存中的离线试验，不改变实车默认参数。不要把Python3专用工具当作Melodic节点运行。

| 证据 | 链接 |
|---|---|
| 最近选定回归81项 | [regressions/report.json](exports/ros_fault_validation_20261019/regressions/report.json) |
| 真实构建、启动和31周期控制 | [vm_validation/report.json](exports/vm_validation_20261019/report.json) |
| 七项真实ROS故障检查 | [guest_results/report.json](exports/ros_fault_validation_20261019/guest_results/report.json) |
| 14项闭环场景，包含未通过项 | [closed_loop/report.json](exports/closed_loop_20261019/report.json) |
| 滤波/变化率对照与延迟扫描 | [response/report.json](exports/loop_response_20261019/report.json)、[delayed_sweep.json](exports/loop_response_20261019/delayed_sweep.json) |

验证报告各有作用域：主机回归中的“未执行ROS”是该测试套件自身的限制，不否定另存的真实ROS检查。旧文档提到的34/73/103项等统计不作为当前统一验收结论。

## 9. 排查与后续工作

1. 构建失败：确认ROS/Python、消息生成和包依赖，勿先升级系统。
2. 节点找不到：核对Linux执行权限、CRLF、source路径和helper重建。
3. 一直等待：查看具体缺失输入、观测年龄和CAN就绪状态，不直接放宽门限。
4. 启动后停车：保留首次故障时间与原因；输入恢复不会自动解除锁存。
5. 有转角却偏离：同时检查真实轮角、方向、定位和目标误差；不要仅凭软件输出调大PID。

后续优先项是经车辆参数标定的动态稳定性、真实传感器质量、VCU反馈及低速实车验收。旧版的历史地图观测时间仍是整图级别，杂点刷新旧地图、GPS静态或漂移等问题不能由超时保护完全识别。

## 10. 文档归档与回退

当前统一入口为本README与[逻辑分析](docs/项目逻辑分析.md)。14份旧说明按原路径结构移至 [历史索引](docs/archive/2026-10-19/README.md)，保留原字节及SHA256；不再作为当前操作指南。

文档整理不修改源码、测试、参数或验证数据。程序回退应从对应`backups/…/acceleration_event`整包恢复并重新构建，不混用不同时期的规划器、控制器和helper。文档归档的[manifest.json](docs/archive/2026-10-19/manifest.json)列出原路径，可恢复旧文档，但不会回退程序。
