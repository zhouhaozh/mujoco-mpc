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
# --view-speed 0.1：更慢，十分之一速度,--view-speed 1：正常速度。
# --profile baseline|fitted|current
python3 tools/run_x4_native_replay.py --view --view-speed 0.1
```

## 生成回放

当前 `current` 实测回放默认启用起始控制状态处理：按源 CSV 的 SHA-256 恢复已保存的底盘前缀估计初态，并逐轴保持初始姿态直到新指令生效。设置保存在 `sckd_sim/models/config/measured_replay_initialization.json`，包含现有十条杀球/和平球记录；未知记录明确标记为未缓存，底盘采用冷启动。Kp/Kd 继续使用当前参数。

每次运行会快照保存起始设置，输出原始命令、实际应用目标和激活时刻。`--legacy-initialization` 可复现原起始规则；历史 `baseline/fitted` 配置保持原规则。`--view-run` 播放已有结果，不会重新应用新设置，需要重新生成回放才能看到新行为。

在项目根目录执行以下命令。输出目录必须不存在或为空；使用时间戳可以为每次生成的回放创建新目录。

生成杀球回放：

```bash
.venv/bin/python tools/run_measured_dynamics.py \
  --no-build \
  --data-dir "/home/ubuntu/Downloads/20261020_5组扣杀球_15条轨迹/下位机轨迹" \
  --output "outputs/measured_dynamics/smash_$(date +%Y%m%d_%H%M%S)" \
  --view
```

生成和平球回放：

```bash
.venv/bin/python tools/run_measured_dynamics.py \
  --no-build \
  --data-dir "/home/ubuntu/Downloads/20261020_5组扣杀球_15条轨迹/和平球轨迹" \
  --output "outputs/measured_dynamics/peace_$(date +%Y%m%d_%H%M%S)" \
  --view
```

使用 `--data-dir` 时，每次处理对应目录中的全部 5 条轨迹。只生成指定一条轨迹时，使用 `--trajectory-file`：

```bash
.venv/bin/python tools/run_measured_dynamics.py \
  --no-build \
  --trajectory-file "/home/ubuntu/Downloads/20261020_5组扣杀球_15条轨迹/和平球轨迹/r20261020_135751_498000.csv" \
  --output "outputs/measured_dynamics/single_$(date +%Y%m%d_%H%M%S)" \
  --view
```

替换 `--trajectory-file` 后的 CSV 路径即可选择其他轨迹。单条回放输出为 `group_1`，并沿用 `current` 的起始控制状态处理。`--trajectory-file` 与 `--data-dir` 不能同时使用。`--group` 只指定打开已有结果时首先显示哪一组，不能筛选本次生成的轨迹。

生成完成后会自动打开对照窗口。批量回放按 **1–5** 切换轨迹，按 **空格** 暂停或继续，按 **R** 从头播放；实心机器人表示仿真，青色透明机器人表示真机轨迹。可追加 `--speed 0.25` 以四分之一速度播放。

`--no-build` 复用已有原生回放程序；首次运行或修改 C++ 后，去掉此参数以重新构建。输出目录中包含 `report.md`、逐组对照图和数值数据；以后可用 `--view-run 输出目录` 再次打开。

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
