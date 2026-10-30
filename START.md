# 显示模型

```bash
cd /home/ubuntu/works/mojuco-mpc/mpc-github

# 用最新 URDF 和配置重新生成模型
.venv/bin/python sckd_sim/models/build_model.py --variant x4_950

# 打开可视化窗口
.venv/bin/python -m mujoco.viewer --mjcf=sckd_sim/models/generated/x4_950_scene.mjcf
```

## 启动

```bash
# --view-speed 0.1：十分之一速度；--view-speed 1：正常速度
python3 tools/run_x4_native_replay.py \
  --config-dir sckd_sim/models/config/measured_dynamics/current \
  --view --view-speed 0.1
```

`--config-dir` 必须指定。可使用仓库自带的 `current/`、`baseline/`、`fitted/`，也可以指定自行复制并修改的配置目录。原生 MPC 回放只读取其中的机械参数和模型；目录里的 `replay_initialization.json` 只供下文的实测轨迹回放使用。

迁移提示：`--profile`、`--compare-profiles` 已移除，分别改用 `--config-dir`、`--compare-config-dirs`；`--legacy-initialization` 已移除，需要旧起始规则时将所选目录中 `replay_initialization.json` 的 `enabled` 设为 `false`。本次还修改了 C++ 配置加载以支持目录内同级的 `motors.json`，拉取改动后的第一次运行请去掉 `--no-build` 完成一次构建。

## 生成回放

`current/` 配置目录默认启用起始控制状态处理：按源 CSV 的 SHA-256 恢复已保存的底盘前缀估计初态，并逐轴保持初始姿态直到新指令生效。设置保存在该目录的 `replay_initialization.json`，包含现有十条杀球/和平球记录；未知记录明确标记为未缓存，底盘采用冷启动。

每次运行会完整快照配置目录，输出原始命令、实际应用目标和激活时刻。`baseline/`、`fitted/` 的 `replay_initialization.json` 默认关闭起始处理；需要复现冷启动时，将自定义目录中该文件的 `enabled` 设为 `false`。行为由文件内容决定，与目录名称无关。`--view-run` 播放已有结果，不会重新应用新设置，需要重新生成回放才能看到新行为。

在项目根目录执行以下命令。`--config-dir` 没有默认值，必须明确指定；相对路径按当前 shell 工作目录解析，也可以使用绝对路径。输出目录必须不存在或为空；使用时间戳可以为每次生成的回放创建新目录。

生成杀球回放：

```bash
.venv/bin/python tools/run_measured_dynamics.py \
  --config-dir sckd_sim/models/config/measured_dynamics/current \
  --no-build \
  --data-dir "/home/ubuntu/Downloads/20261020_5组扣杀球_15条轨迹/下位机轨迹" \
  --output "outputs/measured_dynamics/smash_$(date +%Y%m%d_%H%M%S)" \
  --view
```

生成和平球回放：

```bash
.venv/bin/python tools/run_measured_dynamics.py \
  --config-dir sckd_sim/models/config/measured_dynamics/current \
  --no-build \
  --data-dir "/home/ubuntu/Downloads/20261020_5组扣杀球_15条轨迹/和平球轨迹" \
  --output "outputs/measured_dynamics/peace_$(date +%Y%m%d_%H%M%S)" \
  --view
```

使用 `--data-dir` 时，每次处理对应目录中的全部 5 条轨迹。只生成指定一条轨迹时，使用 `--trajectory-file`：

```bash
.venv/bin/python tools/run_measured_dynamics.py \
  --config-dir sckd_sim/models/config/measured_dynamics/current \
  --no-build \
  --trajectory-file "/home/ubuntu/Downloads/20261020_5组扣杀球_15条轨迹/和平球轨迹/r20261020_135751_498000.csv" \
  --output "outputs/measured_dynamics/single_$(date +%Y%m%d_%H%M%S)" \
  --view
```

替换 `--trajectory-file` 后的 CSV 路径即可选择其他轨迹。单条回放输出为 `group_1`，起始控制规则取自所选目录的 `replay_initialization.json`。`--trajectory-file` 与 `--data-dir` 不能同时使用。`--group` 只指定打开已有结果时首先显示哪一组，不能筛选本次生成的轨迹。

生成完成后会自动打开对照窗口。批量回放按 **1–5** 切换轨迹，按 **空格** 暂停或继续，按 **R** 从头播放；实心机器人表示仿真，青色透明机器人表示真机轨迹。可追加 `--speed 0.25` 以四分之一速度播放。

`--no-build` 复用已有原生回放程序；首次运行或修改 C++ 后，去掉此参数以重新构建。输出目录中包含 `report.md`、逐组对照图和数值数据；以后可用 `--view-run 输出目录` 再次打开。

每套配置目录必须完整包含 `robot.json`、`motors.json`、`chassis.json`、`scene.mjcf`、`replay_initialization.json` 和 `meshes/`。运行时不会回退读取全局 `x4_950.json`；因此修改全局文件不会改变这些目录。要新建一套配置，复制整个目录后修改副本，再把新路径传给 `--config-dir`。

对比两套配置：

```bash
.venv/bin/python tools/run_measured_dynamics.py \
  --compare-config-dirs \
    sckd_sim/models/config/measured_dynamics/baseline \
    sckd_sim/models/config/measured_dynamics/fitted \
  --output "outputs/measured_dynamics/compare_$(date +%Y%m%d_%H%M%S)"
```

完整目录结构、初始化差异和对比输出说明见 [独立动力学配置目录](sckd_sim/models/config/measured_dynamics/README.md)。

## 回放

```bash
python tools/run_measured_dynamics.py \
  --view-run outputs/measured_dynamics/baseline
```


## 辨识后回放
```bash
python tools/run_measured_dynamics.py \
  --view-run outputs/measured_dynamics/parameter_fit_final
```
