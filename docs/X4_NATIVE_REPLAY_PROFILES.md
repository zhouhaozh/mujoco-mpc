# X4 原生 MPC 回放配置目录

在仓库根目录运行：

```bash
python3 tools/run_x4_native_replay.py \
  --config-dir sckd_sim/models/config/measured_dynamics/current \
  --view --view-speed 0.1
```

`--config-dir` 必须指定，没有默认值。相对路径按当前 shell 工作目录解析，也可以传入绝对路径。仓库自带三套配置：

- `sckd_sim/models/config/measured_dynamics/current/`：迁移时的当前机械参数。
- `sckd_sim/models/config/measured_dynamics/baseline/`：固定原始参数。
- `sckd_sim/models/config/measured_dynamics/fitted/`：固定拟合参数。

目录名称没有特殊含义。每个目录必须包含 `robot.json`、`motors.json`、`chassis.json`、`scene.mjcf`、`replay_initialization.json` 和 `meshes/`。原生 MPC 回放读取其中的机械配置、场景和网格；`replay_initialization.json` 只供实测轨迹回放使用，不改变原生 MPC 的启动过程。

每次运行在输出目录的 `config/` 下保存完整源配置、网格和独立回放模型。查看器、预测服务和原生仿真共用该次 `config/runtime.json`；重新打开回放时，查看器从 `summary.json` 加载该次模型。运行不会覆盖源配置或全局生成模型，修改全局 `x4_950.json` 也不会影响这些独立配置目录。

`--config` 中的轨迹、预测模型、MPC 库、初始位置、控制限制和运行时间等设置继续生效；`--config-dir` 选择机械参数、场景及网格。MPC 库内部模型不会随目录切换而重新拟合。

MPC 求解采用**冻结仿真的同步执行语义**：读取 `t` 时刻状态后，仿真时钟、机器人状态、球状态和仿真中的击球剩余时间均保持不变，直到求解返回；随后在同一 `t` 时刻完成控制输出，再推进 **1 ms 仿真时间**。因此，MPC 求解在仿真时间中视为瞬间完成，实际墙钟求解耗时仍被记录，但不计入仿真中的计算延迟。

`mpc_period_ms=0` 表示每推进 1 ms 后，只要目标有效且满足最小击球剩余时间条件，就立即进入下一轮求解；正值表示按指定的仿真周期触发求解，求解期间同样冻结仿真。这不代表实际达到 1000 Hz，墙钟调用频率由求解和循环开销决定。

显示限速沿用现有行为：`mpc_period_ms=0` 且本轮发生求解时，跳过 `--view-speed` 的慢放等待；其余循环按从启动时刻计算的全局时间表等待。因此 `--view-speed 0.1` 不保证求解阶段严格按 0.1 倍速显示，恢复等待时也可能补等或追赶；这些墙钟节奏变化不改变上述仿真时间语义。

输出目录默认带时间戳，也可用 `--output` 指定空目录：

```text
run/
  config/
    scene.mjcf             # 机械参数源场景快照
    replay.mjcf            # 本次实际运行模型，含参考球
    replay.xml
    replay_manifest.json
    robot.json
    motors.json
    chassis.json
    replay_initialization.json
    meshes/
    runtime.json           # 本次所有实际配置路径
  profile_manifest.json    # 配置目录、源运行配置及快照 SHA-256
  resolved_config.json
  frames.jsonl
  events.jsonl
  summary.json
```

`--no-build` 只跳过 C++ 编译，仍会创建新的完整配置快照和回放模型。球拍薄盒保持隐藏，完整预测轨迹和击球时刻位置偏差继续显示。

创建自定义机械参数时复制整个目录并修改副本，然后把副本路径传给 `--config-dir`：

```bash
mkdir -p my_configs
cp -a sckd_sim/models/config/measured_dynamics/current my_configs/native_test
python3 tools/run_x4_native_replay.py \
  --config-dir my_configs/native_test \
  --view --view-speed 0.1
```
