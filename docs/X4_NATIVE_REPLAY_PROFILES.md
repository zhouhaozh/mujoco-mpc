# X4 原生 MPC 回放参数选择

在仓库根目录运行：

```bash
python3 tools/run_x4_native_replay.py --profile baseline --view --view-speed 0.1
python3 tools/run_x4_native_replay.py --profile fitted --view --view-speed 0.1
python3 tools/run_x4_native_replay.py --profile current --view --view-speed 0.1
```

省略 `--profile` 等同于 `current`。

- `baseline`：读取 `sckd_sim/models/config/measured_dynamics/baseline/` 的固定原始参数。
- `fitted`：读取 `sckd_sim/models/config/measured_dynamics/fitted/` 的固定拟合参数。
- `current`：读取运行配置指定的当前机械参数。默认场景从 `models/generated/x4_950_scene.mjcf` 生成，与原入口一致；自定义 `--config` 指定的其他场景路径直接作为源，支持普通机器人场景或已有回放场景。

每次运行在输出目录的 `config/` 下保存机器人、电机、底盘配置和独立回放模型。查看器、预测服务和原生仿真共用该次 `config/runtime.json`；重新打开回放时，查看器从 `summary.json` 加载该次模型。参数切换不会覆盖全局配置、全局生成模型或两套固定参数。

`--config` 中的轨迹、预测模型、MPC 库、初始位置、控制限制和运行时间等设置继续生效；选择 `baseline` 或 `fitted` 只替换机械参数及场景。MPC 库内部模型不随此选项重新拟合。

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
    runtime.json           # 本次所有实际配置路径
  profile_manifest.json    # 参数选择、源运行配置及快照 SHA-256
  resolved_config.json
  frames.jsonl
  events.jsonl
  summary.json
```

`--no-build` 只跳过 C++ 编译，仍会创建新的配置快照和回放模型。网格资源共用 `models/source/x4_950`，不会复制大型网格文件。球拍薄盒保持隐藏，完整预测轨迹和击球时刻位置偏差继续显示。
