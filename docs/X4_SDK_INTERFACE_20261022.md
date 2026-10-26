# X4 SDK 接口第一阶段：离线命令链与验收

后续机械臂工具：用户指定本SDK为准，已新增[逐轴测试程序](X4_ARM_AXIS_TEST_20261022.md)，Ubuntu编译、状态机/CLI检查与完整SDK机械臂仿真通过。默认不连接设备，硬件入口本轮未运行；仍未完成真机方向、零位或物理对齐。

后续进展：2026-10-22 已 SSH 登录 `.38` 真机主机做只读环境审计；用户当前要求暂停控制指令。真机有多套部署包，旧 arm_test fk 会初始化并使能，不能作为被动读取入口。详见[真机环境与控制入口](../../docs/X4_ROBOT_ENVIRONMENT_20261022.md)。下文“未连接机器人”描述本次离线接口验收阶段；目前仍未运行任何真机控制或完成物理对齐。

日期：2026-10-22，北京时间。按用户要求，本阶段不连接机器人。全部编译、测试、仿真、比较和运行证据在 Ubuntu 执行与保存；Mac 只编辑源码、中文文档并核对部署文件。

已完成统一命令／状态契约、MuJoCo 与 X4 SDK 适配器、实际下发命令录制、离线回放和比较工具。当前链路是 **原 pipeline 生成实际执行命令 → JSONL → MuJoCo 或 SDK 设备仿真回放 → 比较报告**。这是接口准备阶段，尚未完成实时 MPC 真机闭环或物理对齐。

## 工作区与版本

| 项目 | 本次位置／版本 |
|---|---|
| 计算端 | `ubuntu@172.20.0.11`，主机 `ubuntu-11`，Ubuntu Linux x86_64，g++ 13.3.0 |
| 独立开发目录 | `/home/ubuntu/Zhaohui/sim-x4-interface` |
| 分支／起点 | `codex/x4-interface-20261022`，基线 `9d38dd0`；本阶段修改未提交、未推送 |
| 原 MPC 主线 | `/home/ubuntu/Zhaohui/sim`；未部署覆盖 |
| SDK | `/home/ubuntu/Zhaohui/x4-sdk`，`main/f261527a432dc143bc281ea0aa764dd660518457`；未改发布包 |
| 构建 | 独立目录内 `sckd_sim/build-x4`；依赖原主线 `deps/mujoco`、`deps/casadi` |
| 运行证据 | 独立目录内 `sckd_sim/artifacts/x4_io_20261022`；不加入 Git、不复制到 Mac |

## 接口与实现

以下源码路径相对团队仓库。

| 文件 | 行为 |
|---|---|
| `sckd_sim/cpp/runtime/robot_io.h` | `Command`、`Observation`、`RobotBackend`、时钟／坐标／能力校验，无 SDK 或 MuJoCo 依赖 |
| `runtime/robot_io_log.h` | `X4_ROBOT_IO/1` JSONL，manifest＋实际命令／反馈／下发结果；拒绝覆盖旧日志 |
| `runtime/robot_io_mapping.h` | 显式五轴方向／零位映射，包含来源／目标坐标名、版本和标定状态 |
| `runtime/mujoco_robot_backend.*` | 车体系 Twist → 既有 ChassisServo；五轴命令 → 既有 MotorModel；状态读取不消耗测量噪声序列 |
| `runtime/x4_sdk_backend.*` | SDK 头隔离；借用已初始化 Robot，或单独建立纯设备仿真；统一预检及状态来源声明 |
| `runtime/trajectory_executor.cpp`、`apps/m2_intercept.cpp` | 记录最终实际下发值，不二次发送；可显式建立零前馈基线 |
| `apps/robot_io_replay.cpp` | `validate`、`mujoco`、`sdk-sim` 三种模式；不提供硬件运行模式 |
| `tools/compare_robot_io.py` | 完整序号集合、指令差、相对调度误差和有效反馈误差；固定声明物理对齐未验收 |

表中省略前缀的 `runtime/` 与 `apps/` 位于 `sckd_sim/cpp/`。

命令使用 SI 单位：底盘 `vx/vy/wz` 为车体系 m/s、m/s、rad/s；关节位置 rad、速度 rad/s、力矩 Nm。每条命令有 sequence、plan_id、issued、expires 和 frame_id。时钟区分 simulation 与 monotonic，不混用数值。当前录制的 `plan_id=0`，manifest 标为 `unavailable_zero`，尚未接通计划版本跟踪。

反馈区分接收时间与采样时间，并标明 simulation_truth、measured、estimated、command_integrated 或 unavailable。SDK facade 不暴露底盘采样时间，保持未知；底盘里程计按当前审计标为命令积分，不能当独立实测位置。逐关节／底盘读取不是整机同步快照。MuJoCo 未建模温度，日志值为 null。

`WriteResult.submitted` 仅表示已交给后端，不代表电机确认执行。SDK 的底盘与臂分两次提交，非整机原子事务；内部手柄仲裁竞态仍由 SDK 维护方确认。借用接口不调用 Init、Shutdown、ArmHome 或 Enable，也不代替硬件生命周期与独立看门狗。

### 前馈与坐标

SDK `SetJoints` 仅公开 pos/vel/kp/kd，不能表达现有 MuJoCo 的 `tau_ff`。默认 `M2_ARM_FEEDFORWARD=preserve` 保留既有控制行为；显式 `zero` 才在最终下发前清零。SDK 适配器拒绝任何非零前馈，不自动删除。

两层坐标变换必须区分：

1. MPC 模型系 → SDK 模型系：由 `X4_JOINT_FRAME_MAP/1` 显式提供，`q_to=sign*q_from+offset`，速度与力矩乘 sign。没有隐式单位映射。
2. SDK 模型系 → 电机编码系：沿用 SDK 的 `motor=joint_sign*(model-joint_offset)`；适配器从同一 ini 读取上下文，仅用于检查协议编码范围，实际命令仍交 SDK 转换，避免重复变换。

SDK 预检在任何派发之前检查时间、序号、frame、finite、前馈、正增益、float 转换及位置／速度／增益编码范围；越界拒绝，不静默钳位。这不是机械运动限位标定。ready、arm_degraded、manual_takeover 不满足时拒绝下发。

此次 SDK 回放使用 artifacts 中的 identity 映射，版本 `offline_identity_software_test_only_20261022`、`calibrated=false`，仅验证软件转换与传递。**该映射未标定，不得作为真机配置。** 仓库未加入默认硬件映射。

## Ubuntu 使用方法

以下命令只在 Ubuntu 独立工作区执行。每次使用新的日志前缀、输出文件和 SDK 日志目录，工具拒绝覆盖。

```bash
cd /home/ubuntu/Zhaohui/sim-x4-interface
cmake -S sckd_sim -B sckd_sim/build-x4 \
  -DMUJOCO_DIR=/home/ubuntu/Zhaohui/sim/deps/mujoco \
  -DCASADI_DIR=/home/ubuntu/Zhaohui/sim/deps/casadi \
  -DX4_SDK_ROOT=/home/ubuntu/Zhaohui/x4-sdk \
  -DCMAKE_BUILD_TYPE=Release
cmake --build sckd_sim/build-x4 --target m2_intercept robot_io_replay \
  test_robot_io test_mujoco_robot_io test_x4_sdk_backend \
  test_runtime_config test_runtime_supervision -j4
```

录制例子（将 `next_run` 改成新的实验名）：

```bash
cd /home/ubuntu/Zhaohui/sim-x4-interface/sckd_sim
mkdir -p artifacts/next_run
M2_ROBOT_IO_LOG_PREFIX=artifacts/next_run/pfly_zero \
M2_ARM_FEEDFORWARD=zero \
./build-x4/m2_intercept models/generated/x4_tennis_demo.mjcf 1 qp 0.25 2026 pfly fixed

./build-x4/robot_io_replay --backend validate \
  --input artifacts/next_run/pfly_zero_trial_0.jsonl
./build-x4/robot_io_replay --backend mujoco \
  --input artifacts/next_run/pfly_zero_trial_0.jsonl \
  --model-root . --output artifacts/next_run/pfly_zero_replay.jsonl
python3 ../tools/compare_robot_io.py \
  --left artifacts/next_run/pfly_zero_trial_0.jsonl \
  --right artifacts/next_run/pfly_zero_replay.jsonl \
  --output artifacts/next_run/comparison.json
```

纯 SDK 设备仿真回放需要额外显式提供 `--sdk-config`、`--sdk-log-dir` 和跨模型 `--joint-map`。工厂要求 `[Sim] enabled=true`，只实例化 ChassisServerSim／ArmServerSim，不调用 `Robot::Init`，不启动 CAN、IMU 或手柄服务。配置的公开 Sim 参数及来源写入日志目录的 `sdk_sim_config_manifest.txt`。回放转换到 SDK CLOCK_MONOTONIC 后按相对时间发送，不放宽原命令到期时间。

```bash
./build-x4/robot_io_replay --backend sdk-sim \
  --input artifacts/next_run/pfly_zero_trial_0.jsonl \
  --sdk-config /home/ubuntu/Zhaohui/x4-sdk/config/robot_sim.ini \
  --joint-map /绝对路径/明确标识用途的模型映射.json \
  --sdk-log-dir artifacts/next_run/sdk_plant_logs \
  --output artifacts/next_run/pfly_zero_sdk_sim.jsonl
```

这两个新选项已进入 ResolvedConfig 的 JSON/env/source/roundtrip。现有 ablation runner 会清除继承的 `M2_*` 环境，尚未新增对应 CLI；本阶段使用直接 m2 入口或显式 JSON，不宣称旧 runner 自动传递新开关。

MuJoCo 回放仅支持当前 1 ms 命令与物理步长，恢复初态、初始保持增益和底盘限率模式。它回放执行器命令，不重放原 pipeline 的球气动力及解析碰撞覆盖；不能据此宣称整个世界状态完全一致。`measured_replay` 会修改球的碰撞属性并外源覆盖球状态，当前回放提前拒绝该模式。本次三份样例都未使用该模式。

## 本次验证结果

Ubuntu 已构建 m2_intercept、robot_io_replay 与相关测试。5/5 CTest 通过：runtime_config、runtime_supervision、robot_io、mujoco_robot_io、x4_sdk_backend。MuJoCo 适配器测试包括 80 ms 变化命令和非零前馈，比较新旧路径全部 17 轴命令及本体状态；读取状态不改变后续测量 RNG。SDK 测试使用 fake endpoint，无设备初始化。

日志比较器最终 **10/10 Python 回归测试通过**，包括混合时钟、坐标错配、非法映射、重复序号、维度／有限值／有效期以及未知来源／采样时间。未知来源不进入误差统计，未知时间不伪造；全部四组回放用最终规则重新生成比较报告，均通过完整序号配对。统计按命令序号配对，没有做反馈延迟补偿，也不证明采样时刻同步。

固定 seed=2026、单回合、baseline，直接入口的底盘限率为 `both_unlimited`，没有改变电机限矩。它们是接口样例，不是控制效果配对统计。

| 用例 | 完整命令数 | 录制末时刻 | m2 原评分 | MuJoCo 回放命令最大差 | q／dq 状态最大差 |
|---|---:|---:|---|---:|---|
| pfly，preserve | 1076 | 1.076 s | 1/1，退出 0 | 0 | 0／0 |
| pfly，zero | 1077 | 1.077 s | 1/1，退出 0 | 0 | 0／0 |
| nmpc6，zero | 2376 | 2.376 s | **0/1，退出 1** | 0 | **0.089292679 rad／1.525158302 rad/s** |

三份完整日志的 validate、MuJoCo 回放均退出 0。nmpc6 单回合有 49 次求解尝试、25 次接纳、24 次拒收，均值 8.155 ms、最大 28.319 ms；这些墙钟样本不是实时性验收，也不能用该单回合推断去掉前馈的性能影响。

nmpc6 的前 1116 条记录状态（0–1.115 s）精确一致；第 1117 条（issued=1.117 s，状态时间约 1.116 s）首次分叉，Joint3 的 q/dq 差为 1.4554e-6 rad／0.0014554 rad/s，此拍臂已施加力矩仍一致，下一拍力矩才分叉。最大 q 差位于状态时间 2.015 s 的 Joint1。已确认回放遗漏球气动力及拍／地解析冲量；解析冲量只修改球状态，不直接给机器人反冲。模型仍允许球与车轮接触，这是候选原因，但日志未记录球状态、接触对和底盘 12 轴电机命令，**具体首次触发仍未确定**，不能排除底盘执行差异。本报告保留非等价结果。

最新带编码范围检查的 SDK 仿真完整接收 pfly_zero 的 1077 条命令；映射后指令差 0，输出日志再次 validate 通过。本次短回放最大发送迟到约 **0.459 ms**，不等于长期实时保证。SDK 使用理想底盘与简化臂动力学，不是 MuJoCo plant；报告不输出未标定的跨模型状态对齐结论，`physical_alignment_validated=false`。

5 个负路径均在创建输出和启动模拟器之前拒绝，退出 2：硬件模式、非零前馈、缺少跨模型映射、位置编码越界、速度编码越界。strace 的文件／socket／connect／bind／ioctl 审计未发现 `socket(AF_CAN)` 或 `/dev/tty`、`/dev/serial`、`/dev/can` 访问。未连接机器人。

另一个 MuJoCo 负路径：含 `measured_replay=true` 的日志提前拒绝，退出 2，未创建回放文件。`git diff --check` 通过；明确部署的源码和团队文档两端逐文件 SHA256 核对，输入／二进制／证据哈希保存在 Ubuntu。

主要证据：`build_final.log`、`build_replay_scope.log`、`ctest_final.log`、`python_tests_strict.log`、三组 `*_trial_0.jsonl`／`*_replay.jsonl`、`integration_results.json`；最新比较为 `strict_comparisons/`。最新 SDK 验收为 `final_encoding/results.json`、`sdk_sim.strace`、`sdk_sim.log`、`sdk_comparison.json` 和 `sdk_plant_logs/`；额外拒绝记录为 `final_encoding/measured_replay_negative.json`。部署审计与资源身份为 `deployment_hash_audit.json`、`input_manifest.json`，它们记录验收后的文件快照，而不是日志生成时的原子快照。均位于上述 Ubuntu 证据目录。

## 后续接入顺序与未完成项

1. 沿用用户已完成的底盘接口封装，重点核对五轴顺序、方向、零位、TCP／拍面定义，提供版本化的已标定 MPC→SDK 映射；后续在明确动作授权与现场条件后，用静态姿态和小幅单轴数据确认。
2. 用统一 Observation 替换在线 RobotStateProvider，处理采样时间、延迟、来源、有效性和完整本体估计。当前 m2 主循环仍使用原仿真状态。
3. 将在线命令派发接入统一 backend，隔离预测／求解与固定节拍执行，完成 plan_id、过期计划、掉线／过期命令及保持／停止策略。记录／回放链已通，在线调度尚未实现。
4. 在用户准备连接机器人后，核实 SDK 生命周期、与 robotd/court_run 的互斥、手柄优先级、独立急停／看门狗和停机顺序，再进行低速小幅开环采集。本轮不执行这些硬件动作。
5. 依据真机数据辨识零位、延迟、增益、摩擦、重力、轮胎／舵响应和底盘独立测量；形成仿真与控制预测模型的版本。编码范围通过不能代替机械可行性与动力学标定。
6. 同一观测、同一命令比较 MuJoCo／SDK 仿真／真机，分阶段跑通预测→击球任务→MPC→执行→反馈闭环；再做留出轨迹、端到端延迟和长时间截止率验收。

本阶段未更改控制器数学、模型基线、SDK 发布包或真机配置；未进行真机运动或命中率验收。
