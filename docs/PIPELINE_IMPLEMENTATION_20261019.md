# Pipeline 同步模块化实施记录

后续：2026-10-22 在独立工作区完成[X4统一接口与离线命令回放第一阶段](X4_SDK_INTERFACE_20261022.md)。原本文记录保持历史口径；在线本体反馈替换、计划版本及异步执行仍未完成，未连接机器人。

日期：2026-10-19，北京时间。计算平台：Ubuntu `ubuntu-11 / 172.30.0.10`，Linux `6.14.0-27-generic`、x86_64。Mac只编辑源码和文档。

Ubuntu团队仓库 `/home/ubuntu/Zhaohui/sim` 已建立分支 `codex/pipeline-modularization`，起点 `7a95839fc9fb752e402cd66d4cb241f19695f972`。原工作区已有控制器修改，本次保留原文件、差异和哈希后增量部署，没有reset或覆盖模型。Git作者身份尚未配置，本次变更尚未提交，也未推送。

## 已实施的边界

| 层 | 源码位置（相对团队仓库） | 本轮职责 |
|---|---|---|
| 配置 | `sckd_sim/cpp/runtime/runtime_config.h` | 一次解析配置，记录最终值、派生模式、来源、argv与白名单环境 |
| 契约 | `sckd_sim/cpp/runtime/contracts.h` | 时钟/坐标/观测、本体来源及后续任务/计划/求解事件值类型 |
| 输入 | `runtime/state_inputs.cpp` | 接收已到达观测、维护捕获/接收时间和序号、本体快照 |
| 调度 | `runtime/synchronous_schedule.h` | 1ms仿真tick、20tick规划、5tick控制trace、17tick viewer的原节拍 |
| 规划 | `runtime/planning_session.cpp`、`controller_adapters.*` | 初规划、精化、各控制方法求解及原有计划状态管理 |
| 接纳与回退 | `runtime/plan_supervisor.h` | 抽出nmpc6末100ms守门与nmpc6/wbmpc连续失败回退规则 |
| 执行 | `runtime/trajectory_executor.cpp` | 轨迹采样、随挥、旧时延补偿、Twist/臂混合命令与前馈 |
| 物理 | `runtime/simulation_step.cpp`、`simulation_physics.cpp` | 气动力、解析地面/拍面冲量及物理推进 |
| 日志 | `runtime/control_trace.h`、`control_telemetry.cpp`、`video_trace.cpp` | 运行域拥有日志文件，旧字段保留，追加输入身份与配置旁文件 |
| 模型包 | `prediction/tennis_calibration/model_bundle.py`、`collision_model_bank.py` | 完整资源清单、SHA256、配方/坐标/hold/recovery/forecast设置；模型加载不再依赖播放器 |
| 实验 | `tools/pipeline_experiment.py` | 最终实验身份、资源/源码/二进制哈希、服务/socket/超时/锁、完成与复用检查 |
| 评估 | `tools/interception_metrics.py` | 纯几何指标与明确评分协议，旧入口保留各自端点口径 |

表中C++ `runtime/` 均位于 `sckd_sim/cpp/runtime/`。主程序从1536行降到889行，其中约200行为按字段注释的Context装配；每tick只调用具名模块，保持原计算顺序：

```text
气动力 → 观测/本体快照 → 初规划/精化/求解/接纳
→ 锁止与击球误差评估 → 轨迹执行 → 控制遥测
→ 碰撞与物理推进 → 视频/GUI
```

当前采用显式引用Context来保留热启动、计划缓存与计数器生命周期。`BallObservation/RobotEstimate`和trace身份已接通；`ActuatorState/HitTaskSnapshot/TrajectoryTiming/SolveReport/PlanDecisionEvent/WholeBodyReference`是已定义的迁移契约，尚未替换运行中的全部 `HitPlan/WmPlan` 和旧事件，不能称为完整不可变任务/计划协议已接通。

CMake已拆出contracts、planning、execution、telemetry、simulation目标。execution/telemetry自身不链接CasADi；多策略应用仍链接planning和nmpc6，pfly独立无CasADi插件构建尚未实现，RobotModel与MuJoCo也尚未完全解耦。

## 配置与模型包的使用

配置优先级为 `内置默认 < JSON values < 环境变量 < 位置参数`。`--config PATH`或`--config=PATH`选择配置，优先于 `M2_CONFIG_PATH`。相对资源路径保持相对于进程cwd，旧atoi/atof、变量存在/空字符串含义及默认策略保持不变。

最小配置示例：

```json
{
  "schema": "SCKD_RUNTIME_CONFIG/1",
  "values": {
    "trials": 1,
    "swing_mode": "qp",
    "base_mode": "pfly",
    "control_variant": "baseline",
    "steer_rate_mode": "limited"
  }
}
```

在Ubuntu的 `sim/sckd_sim` 下运行 `./build-modularization/m2_intercept --config /绝对路径/config.json`。stdout的 `[resolved config]` 行记录最终配置；启用控制诊断时另写 `<diagnostics>.config.json`，已存在文件拒绝覆盖。配置JSON接受的字段由 `RuntimeValues` 定义并严格检查；底层求解器的已有调试环境仍由各实现读取，runner统一记录相关白名单环境。

直接运行baseline的pfly/nmpc6原默认仍是 `both_unlimited`，配对消融runner原默认仍为 `limited`。复现必须显式指定并检查最终值。提取没有扩大取消限率的轴范围，也没有改变电机限矩、转矩转速包络、9ms延迟或驱动轮斜坡。

ModelBundle包含selection、collision父配置与`velocity_0/1/3`子权重、identified滤波配置、noise、可选残差head、transform、观察器配方/常量、hold/recovery及forecast域。collision.parameters保持物理参数权威来源，identified只提供filter。相对路径与内容指纹支持资源随清单一同搬迁；不自动复制模型。

本轮冻结清单位于Ubuntu：

```text
/home/ubuntu/Zhaohui/mpc_sim/artifacts/tennis_controller/20261019_model_bundle/model_bundle.json
fingerprint: 9f9473a94fbd113cbde699199bb936181ecd91a178b20b8a0fa20b7d6dad14a7
```

清单在Ubuntu读取原资源生成，未重训或修改权重。服务支持 `--bundle` 和独占创建的 `--resolved-manifest`，原 `--selection` 入口兼容。新分支 `prediction/tennis_calibration` 保存预测源码，模型、数据和环境仍位于原预测工程。

runner支持 `--binary` 指定候选二进制，`--prediction-source` 指定分支源码；`--prediction` 保留解释器和旧资源根。旧 `build-predictor` 默认入口和二进制未替换。推荐在Ubuntu执行：

```bash
cd /home/ubuntu/Zhaohui/sim
.venv/bin/python tools/evaluate_predictor_control.py \
  --team /home/ubuntu/Zhaohui/sim \
  --prediction /home/ubuntu/Zhaohui/mpc_sim \
  --prediction-source /home/ubuntu/Zhaohui/sim/prediction \
  --binary /home/ubuntu/Zhaohui/sim/sckd_sim/build-modularization/m2_intercept \
  --model-bundle /home/ubuntu/Zhaohui/mpc_sim/artifacts/tennis_controller/20261019_model_bundle/model_bundle.json \
  --out /home/ubuntu/Zhaohui/sim/sckd_sim/artifacts/下一次独立运行目录 \
  --variants external --modes pfly,nmpc6,wbmpc --seeds 2026 \
  --trials 1 --workers 1 --steer-rate-mode limited --profile
```

同一目录仅在完整身份与结果哈希都匹配时复用。旧无身份结果或不同配置拒绝复用；失败尝试保留，`--retry-incomplete`显式归档后重试。输出仍保留旧summary字段；Episode汇总仍有集中管理的stdout兼容适配，尚未全面改为C++结构化EpisodeResult。

## Ubuntu验收

候选位于 `sckd_sim/build-modularization/m2_intercept`。构建使用 `/usr/bin/c++`、`deps/mujoco`、`deps/casadi` 和既有OSQP依赖；独立重建的旧源码基线位于下述证据目录 `baseline/build/m2_intercept`。两端CMake原有差异保留，仅增量部署runtime目标。

| 检查 | 实际结果 |
|---|---|
| 原源码独立重建 | 8/8 CTest通过 |
| 候选构建及相关回归 | 10/10 CTest通过，含12项配置、6项调度/守门doctest案例 |
| 模型包/服务/播放器兼容 | 最终33项中32通过、1跳过；另行启用真实C++ IPC探针，该项通过 |
| 共用实验runner | 17/17通过 |
| bundle runner | 最终11/11通过，含源码选择、候选二进制和身份变更拒绝复用 |
| 纯几何评估与两旧入口 | 14/14通过，保留原浮点端点差异 |
| 三方法×实测缓存首/中/末样本 | 首轮9/9等价；最终候选8/9等价，wbmpc test_01跨60ms墙钟门。比较状态、目标、参考、17轴命令和旧求解/采用/回退事件，均显式limited |
| 合成fixed/rand/gs×三方法 | 首轮8/9等价；wbmpc fixed在60ms墙钟门附近出现差异；独立复跑等价，旧二进制自对照等价；最终候选9/9等价 |
| 预测旧入口/新legacy/新bundle | 3条真实原始轨迹、1999条事件；每对229755个数值一致，最大绝对差0，覆盖hold与recovery |
| 分支源码+ModelBundle+候选runner端到端 | 三方法×fixed/rand/gs共9次完成，配置与每次252–258项资源哈希核验通过；116份分支预测源码与资源工程源码一致 |

对照只排除真实耗时字段，不排除由耗时守门引起的采用/命令差异。首轮wbmpc fixed首次差异在仿真 `t=0.520s`：旧解59.970444ms被接纳，新解60.106144ms被拒收。最终实测test_01首次差异在求解时刻 `t=1.080s`：旧解59.149937ms被接纳，新解60.411477ms被拒收。两次均跨越原60ms门，后续差异完整保留；最终18组总计17/18等价，不能记作全部通过。独立复跑一致不抵消失败，也不证明任意系统负载下确定性。

新增输入快照没有增加带噪底盘传感器读取：原SimBackend球观测与关节测量共用RNG，额外读取会改变后续球噪声。当前保留原五臂轴读取次数，底盘轴快照明确标记仿真真值；臂位置标记测量，臂速度仍标记真值。

所有原始证据保留在Ubuntu：

```text
/home/ubuntu/Zhaohui/sim/sckd_sim/artifacts/pipeline_modularization_20261019/
  baseline/、baseline_manifest.json、tracked_diff.txt
  deployment_manifest.json
  synthetic_pairs/、wbmpc_isolated_pair/、wbmpc_baseline_self/
  measured_pairs/、final_measured_pairs/
  predictor_equivalence/
  bundle_runner_e2e/、verification_summary.json
```

这些是重构等价与生命周期检查，使用历史轨迹，不能解释为新盲测精度、命中率提升、墙钟50Hz/1kHz闭环或实机验收。实际相机外参仍未标定，机器人反馈仍含上述真值。

Git初始没有暂存内容；失败的baseline提交准备已校验138份暂存blob等于保存的原文件后撤销，工作文件全部保留。初始未提交改动与本次修改均在新分支工作区，未把历史改动混入提交；源码增量另保留在证据目录的 `task_source.patch`。没有模型、数据、环境或运行日志加入提交。

## 后续独立阶段

1. 将Context中的共享计划状态逐步迁移为已定义的任务/计划/求解事件值类型，独立拥有状态与版本；完整消费预测有效域和失效原因。
2. 抽出RobotModel/StateProvider，继续分离算法库与MuJoCo/CasADi依赖，并完善结构化Episode输出。
3. 在新的独立变更中实现异步预测/求解、单调墙钟、迟到结果和版本拒收、执行周期/截止时间统计；重新设计60ms等墙钟守门的验收方式。

本轮保留同步仿真推进，未引入线程或改变接纳阈值；异步与墙钟实时性需要自己的基线和截止时间验收。
