# 显式选择新旧动力学参数

`baseline/` 固定原始参数，`fitted/` 固定本轮拟合参数。两套均包含完整配置及场景，运行时不读取旧 `outputs/` 结果，也不覆盖全局参数。

| 文件 | 调整内容 |
|---|---|
| `robot.json` | `arm_servo.kp/kd`、机构限矩和关节映射 |
| `motors.json` | 电机限矩、T-N 曲线等 |
| `chassis.json` | 底盘斜坡、舵轮伺服参数 |
| `scene.mjcf` | 关节 `armature/damping/frictionloss` 和静态 `ctrlrange` |
| `provenance.json` | 建立配置时的原始文件哈希，仅用于溯源 |

修改限矩时，`scene.mjcf` 的执行器 `ctrlrange` 必须同步匹配电机/机构限矩，否则原生加载器会拒绝错配。实际关节惯量/阻尼由 `scene.mjcf` 加载；`robot.json.fitted_joint_dynamics` 只是拟合记录。

从仓库根目录执行：

```bash
# 一次解析五组实测初态和命令，分别重新运行两套动力学
.venv/bin/python tools/run_measured_dynamics.py --compare-profiles \
  --output outputs/measured_dynamics/my_comparison

# 分别选择原参数/新参数，重新计算后打开查看器
.venv/bin/python tools/run_measured_dynamics.py --profile baseline --view
.venv/bin/python tools/run_measured_dynamics.py --profile fitted --view
```

省略 `--output` 会自动创建带时间戳的新目录。指定目录必须不存在或为空。已有原生构建可加 `--no-build`。`--profile current`（也是省略配置时的默认行为）使用全局 `models/config/*x4_950.json` 和 `models/generated/x4_950_scene.mjcf`，不会改变两份固定配置。

`--compare-profiles` 生成：

```text
my_comparison/
  comparison.md/json          # 新旧误差及输入一致性
  baseline/
    config/                  # 本次实际使用的参数和模型快照
    manifest.json            # 配置哈希、初态/命令哈希和运行选项
    group_N_input.json       # 原生动力学实际输入
    group_N/                 # 新算出的 frames、指标、实测叠加图
  fitted/
    ...
```

两套运行共用一次解析结果，保留同样的初始状态、命令、时间、坐标解释、控制模式和重力前馈开关；仅配置路径不同。`comparison.json.same_experiment_inputs` 确认输入哈希一致。新增功能只修改配置选择、运行封装和查看器，C++ 控制器、动力学求解及命令消费逻辑不变。

`--view` 是本次计算完成后显示；`--view-run` 才是只看已保存结果：

```bash
.venv/bin/python tools/run_measured_dynamics.py --view-run outputs/measured_dynamics/my_comparison/baseline
.venv/bin/python tools/run_measured_dynamics.py --view-run outputs/measured_dynamics/my_comparison/fitted
```

新结果的查看器读取各自 `config/scene.mjcf`，不再用全局当前模型替代。旧结果没有模型快照时仍按旧方式加载当前模型。网格文件继续共用 `models/source/x4_950`；快照不包含网格副本。

`fitted` 的 RS04 120 N·m 是不足 0.6 s 轨迹的有效限矩，不是持续额定能力；原有模型没有热/持续时间限制。两套仿真都从同样的可观测初态及初始化规则启动，真机记录前的控制器历史仍未知。
