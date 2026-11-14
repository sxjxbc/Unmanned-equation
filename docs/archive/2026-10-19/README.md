# 文档整理记录与历史索引

整理日期：2026-10-19。当前文档以[整体README](../../../README.md)和[项目逻辑分析](../../项目逻辑分析.md)为准。此处归档保留历史原文，不代表现行操作步骤或已经完成的功能。

## 新增与合并

| 当前文档 | 内容来源与用途 |
|---|---|
| 根README | 合并运行、构建、诊断、验证与回退说明，核对当前启动文件 |
| docs/项目逻辑分析.md | 重新依据源码整理地图/固定赛道/车体坐标、估计、目标、PD、保护与终点，不直接搬用旧说明 |
| src/acceleration_event/README.md | 缩为包级导航，避免同时维护两份总说明 |
| src/acceleration_event/DATA_QUALITY.md | 保留旧链接入口，指向当前规范并提示原文已归档 |

## 归档清单与原因

以下14份原文先复制并校验SHA256，再从当前目录移除；包README和DATA_QUALITY随后写入短导航。目录结构保留原工作区相对路径，文件内容未改写。

| 原文/归档链接 | 归档原因与替代位置 |
|---|---|
| [docs/北理直线加速融合方案_2026-10-12.md](docs/北理直线加速融合方案_2026-10-12.md) | 实施前方案，保留思路；当前链路对照见逻辑分析 |
| [docs/雷达融合版运行与验证.md](docs/雷达融合版运行与验证.md) | 独立旧长说明，测试统计和原脚本未改描述已过时；合并到两份主文档 |
| [docs/直线加速模块交付文档.md](docs/直线加速模块交付文档.md) | 与包README/融合说明重复，部分运行前提属当时版本；统一README |
| [docs/旧版失效保护修正_2026-10-15.md](docs/旧版失效保护修正_2026-10-15.md) | 阶段性修正记录；现行状态机与启动恢复已合并 |
| [docs/固定赛道坐标修正_2026-10-15.md](docs/固定赛道坐标修正_2026-10-15.md) | 阶段性实现记录；现行参考系与局部目标已合并 |
| [docs/启动可靠性与无输出验证_2026-10-15.md](docs/启动可靠性与无输出验证_2026-10-15.md) | 当时尚未验证ROS，已被真实构建与集成证据更新 |
| [docs/实车自动诊断使用说明.md](docs/实车自动诊断使用说明.md) | 操作合并到README；原文“旧版不发布内部状态”不再适用 |
| [docs/启动与完整闭环检查_2026-10-19.md](docs/启动与完整闭环检查_2026-10-19.md) | 保留最初闭环失败记录，环境状态随后已有更新 |
| [docs/虚拟机ROS无输出验收_2026-10-19.md](docs/虚拟机ROS无输出验收_2026-10-19.md) | 合并步骤与结论，原始build/ROS日志仍在exports，未移动 |
| [docs/ROS持续通信与故障注入_2026-10-19.md](docs/ROS持续通信与故障注入_2026-10-19.md) | 合并七项结果和摆动限制，原始逐消息证据保留 |
| [src/acceleration_event/README.md](src/acceleration_event/README.md) | 包长说明只维护融合版，统计较旧；原位改为整体文档导航 |
| [src/acceleration_event/DATA_QUALITY.md](src/acceleration_event/DATA_QUALITY.md) | 描述了当前没有采用的planning/status、定位策略和终点方案，不能作为现行规范 |
| [src/acceleration_event/docs/straight_logic_first_three.md](src/acceleration_event/docs/straight_logic_first_three.md) | 描述map斜率、单目标及103/84项检查，与当前固定T/混合目标实现不一致，保留为其他阶段记录 |
| [src/lidar/LIDAR_SECTOR_PLAN.md](src/lidar/LIDAR_SECTOR_PLAN.md) | 雷达/八字阶段性改造方案，含当前直线聚类不提供的接口描述；第三方README未动 |

## 归档完整性与恢复

[manifest.json](manifest.json)记录每份原文的原始路径、归档路径、SHA256。归档时逐文件比较原件与副本后才移除原件；整理结束再核对14份归档内容。源代码、测试、launch、参数、脚本等118个运行文件的整理前指纹在 [runtime_before.json](runtime_before.json)，整理后校验结果在 [verification.json](verification.json)。

要查旧说明，直接阅读归档；要恢复到历史目录布局，依据manifest将对应文件复制到新的恢复目录或明确的原路径。恢复旧README/DATA_QUALITY会覆盖当前导航，恢复旧文档不会回退程序。恢复程序应另外使用backups下相应整包并重建。

历史原文的相对源码链接和命令按当时布局保留，可能指向已改变的路径；没有改写原文来伪造现行有效性。当前主文档及本索引的链接单独校验。

## 未清理的内容与仍待确认事项

- 源码、测试、构建配置、运行数据、exports证据、backups、第三方README/许可证全部保留。
- `src/introduce.docx`作为原始资料保留，本次未审校其内部叙述；技能/编辑器文件不属于本轮文档清理范围。
- 真实整套雷达/八字驱动环境、融合版集成效果、车辆标定与动态稳定性仍待确认。旧版两包真实ROS通过不覆盖这些范围。
- 构建命令按当前已测Linux工作区整理；Windows共享未挂载的事实保留，不假定共享已经可用。八字模拟包名不匹配和start_mapping.sh旧提示在README中明确列为限制，未擅自修改代码。
