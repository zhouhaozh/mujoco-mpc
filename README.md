# 网球机器人仿真与MPC

2026-10-26：以GitLab最新main `868a9d9` 建立 `codex/x4-inertia-integration-20261026`，合入X4接口/五轴30°工具 `b1943e0`，无文本冲突。相关编译、6组CTest和77项Python检查通过；上游惯量数据属于仿真辨识，未直接替换物理参数，未运行真机。来源、J2重力下垂量级分析和复现见[惯量整合报告](docs/X4_INERTIA_INTEGRATION_20261026.md)。

代码来源：Ubuntu /home/ubuntu/Zhaohui/sim；本次没有使用Mac项目副本。

2026-10-20恢复原团队73个提交及分支关系，并将Ubuntu当前已保存的预测、同步模块化、时间窗口实验工具与文档提交到 codex/pipeline-modularization。
GitLab首次源码快照单独保留，默认main通过一次合并同时保留原团队历史和该快照。见[分支协作说明](BRANCH_COLLABORATION.md)。

## 入口

- sckd_sim：MuJoCo团队仿真、pfly/nmpc6/wbmpc、同步runtime模块、机器人描述和配置。
- prediction/tennis_calibration：预测源码及测试。
- tools、tests：实验编排、几何评估、缓存生产、时间窗口搜索和审计。
- [当前状态](docs/MPC_PROJECT_STATUS.md)、[模块化实施](docs/PIPELINE_IMPLEMENTATION_20261019.md)、[时间窗口搜索](docs/TIME_WINDOW_SEARCH_20261019.md)。
- USAGE.md、PREDICTOR_INTEGRATION.md、sckd_sim/README.md：既有使用说明。

## 资源与运行边界

Ubuntu为唯一计算和结果存储端。必要URDF、MJCF/XML、STL、静态配置及求解器生成源码随代码保留。
本次新增提交不加入本机环境、deps、编译输出、原始数据、模型训练结果、实验日志或视频。
原历史已经跟踪的少量库、报告、模型与历史日志保留原blob，不改写历史，也不提交它们的新运行改动。
必要预测模型与资源仍在Ubuntu完整工程；选定模型包和运行环境位于 /home/ubuntu/Zhaohui/mpc_sim。
.gitignore不整体排除artifacts或NPZ/JSON模型，后续必要资源须按明确清单审查。

部分构建入口仍依赖Ubuntu既有路径，克隆后需要配置依赖及外部模型资源。
本次只做Git历史、内容和合并关系验证，未重新运行仿真、训练或控制性能测试；300Hz完整MPC未验收，默认算法未因上传切换。

## 来源审计

SOURCE_SNAPSHOT.json记录本次选中工作树源码及文档的相对路径、字节数和SHA256。
初次独立快照及其原始清单保留在 archive/source-snapshot-20261020 分支。
