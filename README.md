# 网球机器人仿真与MPC

本仓库于2026-10-20从Ubuntu当前项目源码创建，只以 /home/ubuntu/Zhaohui/sim 为来源，没有使用Mac项目副本。
导出包含原分支 codex/pipeline-modularization 的当前已保存源码以及未提交改动；这是独立源码快照，不改写原仓库历史。

## 入口

- sckd_sim：MuJoCo团队仿真、pfly/nmpc6/wbmpc、同步runtime模块、机器人描述和配置。
- prediction/tennis_calibration：分支上的预测源码及测试。
- tools、tests：实验编排、几何评估、缓存生产、时间窗口搜索和审计。
- docs/MPC_PROJECT_STATUS.md：项目状态；docs/PIPELINE_IMPLEMENTATION_20261019.md：模块化说明。
- docs/TIME_WINDOW_SEARCH_20261019.md：时间窗口搜索，含未改善的碰撞结果及实时性边界。
- USAGE.md、PREDICTOR_INTEGRATION.md、sckd_sim/README.md：既有使用说明。

## 资源和运行边界

机器人URDF、MJCF/XML、必要STL网格、静态配置、求解器生成C/JSON源码随项目保留。
Python环境、MuJoCo/CasADi/OSQP依赖、编译产物、训练权重、预测模型包、原始数据、实验日志和视频没有上传。
完整资源仍保存在Ubuntu；预测运行环境及选定模型位于 /home/ubuntu/Zhaohui/mpc_sim。
部分构建入口仍使用Ubuntu既有依赖路径，克隆本仓库后需按文档配置依赖及外部模型资源，不能直接认定为可运行验收。

本次仅执行源码内容核对与Git传输检查，没有重新运行仿真、训练或性能测试。
默认控制策略未因上传而切换；300Hz完整MPC尚未验收。

## 来源审计

SOURCE_SNAPSHOT.json保存每个原始文件的相对路径、字节数和SHA256，及原分支/HEAD。
根.gitignore在保留原规则后追加了资源排除规则；根README和SOURCE_SNAPSHOT是本次导出说明。
