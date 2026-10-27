# 网球机器人 MuJoCo 仿真与 MPC

本仓库包含网球机器人整机 MuJoCo 仿真、`pfly` / `nmpc6` / `wbmpc`
控制策略、网球轨迹预测、同步运行时、机器人描述和实验工具。

> 新机器请从 [中文环境安装与部署手册](docs/环境安装与部署.md) 开始；手册包含一键安装、
> 无头验收、GUI/SSH 运行和常见故障处理。

## Ubuntu 快速开始

一键安装支持 Ubuntu 24.04 x86_64。在仓库根目录执行：

```bash
bash tools/install_ubuntu.sh
```

脚本准备系统依赖，创建项目环境，安装固定版本的 MuJoCo、CasADi 等依赖，
编译工程、运行 CTest 并执行三球无头仿真。系统包已准备好时可加
`--skip-system-deps`；低内存机器可加 `--jobs 1`。

无图形环境可直接运行三球 `pfly` 仿真：

```bash
cd sckd_sim
./build/m2_intercept models/generated/x4_tennis_demo.mjcf 3 qp 0.25 2026 pfly
```

桌面环境下从仓库根目录启动 MuJoCo viewer：

```bash
tools/run_mujoco_demo.sh
# 或指定策略和球数
tools/run_mujoco_demo.sh nmpc6 3
```

更多命令见 [操作手册](USAGE.md) 和 [sckd_sim 说明](sckd_sim/README.md)。

## 目录

- `sckd_sim`：C++17 仿真、控制、运行时、MJCF/URDF/STL 与配置。
- `prediction/tennis_calibration`：预测与校准源码及测试。
- `tools`、`tests`：实验编排、评估、缓存和时间窗口工具。
- `docs`：设计、接口、阶段结论和真机交接资料。

## 资源边界

- 当前构建保留了 Linux x86_64 所需的 `libosqp.a` 和 `nlp.so`；它们不可直接用于其他平台。
- MuJoCo、CasADi、Python 虚拟环境和编译输出由安装脚本在本地生成，不纳入仓库。
- 大型实验报告、视频和原始运行数据不在当前分支快照；旧报告仍可从保留的历史提交查看。
- 外部 GP/MLP 预测模型和实测数据不随仓库分发；默认仿真可使用内置预测链运行。

## 历史

本仓库保留原团队完整提交历史及六个分支，当前 `main` 在历史之上整理为可独立克隆的快照。
旧 GitLab 地址、历史报告和旧机器路径可能出现在过去的提交中；当前分支的构建和运行不依赖这些地址。
分支关系见 [历史与分支说明](BRANCH_COLLABORATION.md)。

本次整理范围和验证结果见 [GitHub 整理记录](GITHUB_PREPARATION.md)。
