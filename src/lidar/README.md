# lidar 雷达驱动 —— 独立嵌套工作区（重要说明）

## ⚠️ 这个目录不是普通 ROS 包，是一个"嵌套工作区"

`lidar/` 顶层**没有** `package.xml`，它不是一个可直接被主工作区 `catkin_make` 识别的包。
它内部 `lidar/src/` 下才是真正的东西，是一个用 **catkin_tools + catkin_simple** 构建的独立工作区。

## 内部真正的 6 个包（在 lidar/src/ 下）

| 包 | 作用 | 构建方式 |
|---|---|---|
| `catkin_simple` | 构建工具库（提供 cs_* 宏） | 依赖 |
| `glog_catkin` | 日志库 catkin 封装 | 依赖 |
| `lidar_nodes` | 雷达节点：`clustering.py`（聚类）、`input_and_prefilter.py`（预过滤） | catkin_make |
| `linefit_ground_segmentation/linefit_ground_segmentation` | C++ 地面分割核心库 | **catkin_simple**（cs_add_library） |
| `linefit_ground_segmentation/linefit_ground_segmentation_ros` | 地面分割 ROS 节点 | **catkin_simple**（cs_install） |

注意：`linefit_ground_segmentation` 是"包套包"，里面嵌套了上面两个真包。

## 为什么不能摊平进主工作区

`linefit_ground_segmentation` 系列用了 `find_package(catkin_simple)` + `cs_add_library`/`cs_install`
这些 **catkin_simple 特有的宏**，且依赖 **PCL 点云库**。主工作区用标准 `catkin_make`，
没有 `catkin_simple`，直接摊平会导致 `cs_*` 宏未定义、编译报错。

## 正确的构建方式（在车机 Linux 上）

这个 lidar 工作区需要**单独**用 catkin_tools 构建，不能和主 nrt_ws 混在一起：

```bash
# 1. 先装 catkin_tools（如果没有）
sudo apt install python-catkin-tools

# 2. 进入 lidar 工作区，单独构建
cd /home/nvidia/lidar_ws    # 或你放置 lidar 的位置
catkin build

# 3. source 后，雷达驱动就能用
source devel/setup.bash
roslaunch lidar_nodes lidar.launch
```

## 依赖的第三方驱动（车机需预装，zip 里没有）

`lidar.launch` 里 include 了 Velodyne 官方驱动：

```xml
<include file="$(find velodyne_pointcloud)/launch/VLP16_points.launch"/>
```

这个 `velodyne_pointcloud` 包**不在本仓库里**，需要车机系统预装（`sudo apt install ros-melodic-velodyne`）。

## 雷达数据接口（来自 LIDAR_SECTOR_PLAN.md）

- `/prefiltered_points`：270° 扇区点云
- `/ground_segmentation/obstacle_cloud`：非地面点
- `/lidar/cones`：锥桶候选（带置信度）
- `/clustered_points`：兼容旧消费者的 PoseArray
- `/cone_map`：世界坐标锥桶地图
