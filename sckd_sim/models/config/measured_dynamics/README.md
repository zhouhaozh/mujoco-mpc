# 独立动力学配置目录

实测轨迹回放通过 `--config-dir` 直接选择一套完整配置。命令不再根据 `current`、`baseline`、`fitted` 这些名称切换参数，也不会回退读取全局 `x4_950.json`。仓库提供三个可直接使用的目录：

- `current/`：保存迁移时的当前参数，启用已辨识的起始控制状态处理。
- `baseline/`：保存原始参数，起始控制状态处理关闭。
- `fitted/`：保存五条杀球轨迹拟合参数，起始控制状态处理关闭。

目录名称没有特殊含义。运行行为完全由目录内的文件决定；复制并重命名目录后仍会得到相同结果。

迁移提示：`--profile`、`--compare-profiles` 已移除，分别改用 `--config-dir`、`--compare-config-dirs`。`--legacy-initialization` 也已移除，需要旧起始规则时将所选目录中 `replay_initialization.json` 的 `enabled` 设为 `false`。C++ 配置加载已调整为支持目录内同级的 `motors.json`；拉取本次改动后的第一次运行必须去掉 `--no-build` 构建一次，之后才可继续复用已有二进制。

## 目录内容

每套配置必须独立包含：

```text
my_robot/
  robot.json
  motors.json
  chassis.json
  scene.mjcf
  replay_initialization.json
  meshes/
```

| 文件 | 作用 |
|---|---|
| `robot.json` | `arm_servo.kp/kd`、机构限矩和关节映射 |
| `motors.json` | 电机限矩、T-N 曲线等 |
| `chassis.json` | 底盘斜坡、舵轮伺服参数 |
| `scene.mjcf` | 质量、惯量、关节 `armature/damping/frictionloss`、静态 `ctrlrange` 和网格引用 |
| `replay_initialization.json` | 实测轨迹回放的起始控制策略和逐轨迹底盘初态 |
| `meshes/` | `scene.mjcf` 使用的全部模型网格 |

`provenance.json` 可以随目录保留参数来源和哈希，但不是运行所需文件。运行前会检查配置和模型资源是否完整；缺少必需文件或引用目录外资源会直接报错，不会悄悄使用全局文件。

修改限矩时，`scene.mjcf` 的执行器 `ctrlrange` 必须同步匹配电机和机构限矩，否则原生加载器会拒绝错配。实际质量、惯量和关节被动动力学由 `scene.mjcf` 加载；`robot.json.fitted_joint_dynamics` 只是拟合记录。

## 单套配置回放

从仓库根目录执行：

```bash
.venv/bin/python tools/run_measured_dynamics.py \
  --config-dir sckd_sim/models/config/measured_dynamics/current \
  --no-build \
  --view
```

`--config-dir` 必须指定，没有默认目录。相对路径按当前 shell 工作目录解析，绝对路径也可以使用。省略 `--output` 会自动创建带时间戳的新目录；显式指定的输出目录必须不存在或为空。

每次运行都会把所选配置及网格完整复制到输出的 `config/`，并记录文件哈希。之后修改源配置不会改变已经生成的结果，`--view-run` 会继续使用回放中的快照：

```bash
.venv/bin/python tools/run_measured_dynamics.py \
  --view-run outputs/measured_dynamics/previous_run
```

## 对比两套配置

一次解析实测初态和命令，再分别运行两个目录：

```bash
.venv/bin/python tools/run_measured_dynamics.py \
  --compare-config-dirs \
    sckd_sim/models/config/measured_dynamics/baseline \
    sckd_sim/models/config/measured_dynamics/fitted \
  --output outputs/measured_dynamics/my_comparison
```

对比输出固定使用 `config_1/`、`config_2/`，因此两个源目录同名时也不会冲突：

```text
my_comparison/
  comparison.md/json
  config_1/
    config/                  # 第一套完整配置快照和网格
    manifest.json            # 配置、初态、命令和运行选项哈希
    group_N_input.json
    group_N/
  config_2/
    ...
```

两轮共用一次解析得到的原始实测状态、命令、时间和控制选项。`comparison.json` 分别记录原始输入是否一致，以及应用各自 `replay_initialization.json` 后的有效输入是否一致。若两套目录的起始策略不同，有效输入会不同，此时结果同时包含动力学参数和起始策略的影响，不能当作只改变机械参数的对照。

分别查看两轮结果：

```bash
.venv/bin/python tools/run_measured_dynamics.py \
  --view-run outputs/measured_dynamics/my_comparison/config_1
.venv/bin/python tools/run_measured_dynamics.py \
  --view-run outputs/measured_dynamics/my_comparison/config_2
```

## 创建和修改配置

复制整个目录，再修改副本：

```bash
mkdir -p my_configs
cp -a sckd_sim/models/config/measured_dynamics/current my_configs/test_a
.venv/bin/python tools/run_measured_dynamics.py \
  --config-dir my_configs/test_a \
  --no-build --view
```

直接修改全局 `sckd_sim/models/config/x4_950.json` 不会影响这些独立目录。需要冷启动和立即应用机械臂命令时，将副本中 `replay_initialization.json` 的 `enabled` 设为 `false`；需要保存的前置控制状态时，将其设为 `true` 并维护对应轨迹哈希。这里没有基于目录名称的特殊逻辑。

`fitted/` 中 RS04 的 120 N·m 是不足 0.6 秒轨迹得到的短时有效限矩，不是持续额定能力；模型没有热或持续时间限制。
