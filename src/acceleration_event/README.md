# acceleration_event

本包提供旧版地图/路径直线链路、当前帧雷达融合链路，以及旁路诊断和无输出验证工具。

- 构建、启动、日志、验证进度：[项目整体README](../../README.md)。
- 当前算法、参数与状态机：[项目逻辑分析](../../docs/项目逻辑分析.md)。
- 本包原长说明已归档：[历史README](../../docs/archive/2026-10-19/src/acceleration_event/README.md)。

`acceleration_event.launch`默认启用车辆输出；首次检查使用`validation.launch`或独立无输出测试脚本。融合版的`acceleration_lidar.launch`默认shadow，不能与旧版启动方式或验收结果混用。
