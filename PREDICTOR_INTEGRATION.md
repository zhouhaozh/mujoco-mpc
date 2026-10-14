# 独立网球预测服务接入（2026-10-14）

当前分支 `codex/tennis-predictor-pfly-nmpc6`。旧控制说明是导入时快照。最新完整报告位于主项目 `demo/mujoco_motor_lab/docs/tennis_controller_integration.md`；Ubuntu 副本为 `/home/ubuntu/Zhaohui/mpc_sim/docs/tennis_controller_integration.md`。

构建入口：`python3 tools/setup_predictor_ubuntu.py --archive /home/ubuntu/Zhaohui/sim.tar.xz`。Python 环境在本目录 `.venv`，新构建在 `sckd_sim/build-predictor`。Linux x86_64 的 OSQP 仍依赖原团队归档中经哈希核对的静态库。

`BALL_PREDICTOR_SOCKET` 开启外部服务，两种模式为 `pfly` / `nmpc6`；不设置时仍使用旧 EKF，`BALL_DEDUP=1` 可禁止重复融合。仿真 nmpc6 从 MJCF 读取 J2/J3 角窗，`NMPC6_LEGACY_LIMITS=1` 保留旧实机界对照。不要修改 XML 去迁就旧角窗。

真实数据的所选配方为 GP＋自适应/Huber＋恢复保护＋MLP，首次触地后停更 100 ms，最长预测 1.5 s，残差最长 0.7 s。合成球上的命中率未全面胜过原 EKF；预测展开平均 33.56 ms，当前仅研究仿真，不是硬实时或实机验收。

复现实验：`tools/evaluate_predictor_control.py --team . --prediction ../mpc_sim --out ../mpc_sim/artifacts/tennis_controller/<新的结果目录>`。默认 3 种预测配置×2 控制器×3 发球族×3 种子×10 球，保持未命中、不可规划和回退记录。每次使用新的结果目录，避免混合耗时日志。

本轮已加入 wbmpc；数据来源与频率报告为主线 `docs/tennis_wbmpc_and_frequency.md`。评估脚本默认覆盖三控制器，`--modes` 可指定；`--profile` 写出分阶段耗时，单进程测频使用 `--workers 1`。MPC 失败/拒收和 pfly 回退保留计数。

对侧发球录像：新增 `tools/record_prediction_demo.py`，`opposite` 发球配置从 `(16,-0.5,2)` 越网。预测曲线只读取当前控制器缓存，不增加预测调用；每10ms记录真实状态，离线 MuJoCo 渲染0.25×慢放。Ubuntu 视频位于 `sckd_sim/video/prediction_opposite_20261014_v2/`，本组3方法均未命中，如实保留。详细说明见统一仓库 `demo/mujoco_motor_lab/docs/tennis_prediction_video.md`。
