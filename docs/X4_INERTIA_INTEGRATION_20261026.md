# X4 惯量更新整合与角度误差分析

2026-10-26，北京时间。代码检查、编译、数值复算均在 Ubuntu `ubuntu-11 / 172.20.0.11` 完成。本轮没有连接真机控制主机、使能、初始化或发送运动指令。

## 1. 整合范围

- GitLab MPC：`LZH/mpc`，本次拉取的最新 `main` 为 `868a9d9`。
- 当前接口及五轴方向工具保存为提交 `b1943e0`，来源分支 `codex/x4-interface-20261022`；52个源码、测试和文档文件，运行数据不在该提交中。
- 以 `868a9d9` 新建 `codex/x4-inertia-integration-20261026`，合入上述提交，无文本冲突；`sckd_sim/CMakeLists.txt` 自动合并，双方新增目标均保留。
- Ubuntu新工作区：`/home/ubuntu/Zhaohui/sim-x4-inertia-20261026`。原 `sim`、`sim-x4-interface` 的运行现场及数据保留。
- GitLab SSH 已验证，服务端口为2222。密钥仅保存在Ubuntu用户SSH目录，不进入仓库。
- SDK远端 `origin/main` 更新至 `77a8108`，当前权威SDK工作树仍为 `f261527`；本轮不替换已部署二进制或SDK库。

上游已经跟踪的CSV、图和视频按main原样保留；本轮新生成的日志、图、数值结果只留Ubuntu artifacts，不新增到提交。

## 2. 新数据的真实含义

上游新增的是**五轴MIT前馈辨识的仿真程序、仿真样本和真机采集规程**，不是已经标定完成的真机惯量表。相对旧main，本次没有修改MJCF/URDF中的质量、质心或连杆惯量。

主要入口：

| 路径 | 用途 |
|---|---|
| `sckd_sim/cpp/apps/mit_motor_sim.cpp` | 单电机MIT执行链及前馈误差实验 |
| `sckd_sim/cpp/apps/mit_arm_id_sim.cpp` | 固定基座五轴动力学、量化、延迟和前馈对照 |
| `sckd_sim/python/tools/mit_arm_id.py` | 重力、摩擦、全耦合惯量辨识及前馈生成 |
| `sckd_sim/eval/reports/mitarm/` | 已保存的仿真输入、拟合参数及前馈样本 |
| `docs/真机臂前馈辨识操作手册.md` | 日志字段、实验阶段和现场待核验项 |

`mit_arm_id_sim.cpp` 人工注入黏滞摩擦、库仑摩擦和转子反射惯量；其中 `armature=[0.02,0.02,0.01,0.005,0.005] kg·m²` 是仿真占位值。不能当作实测结果。

`mitarm_id_params.npz` 包含：

| 字段 | 形状与含义 | 使用边界 |
|---|---|---|
| `gtheta` | 5项重力基参数，kg量纲 | 固定几何下的拟合组合，不是五个可直接替换的真实质量 |
| `b_hat` | 5项黏滞摩擦，N·m·s/rad | 部分为负，不能直接作为物理阻尼 |
| `tc_hat` | 5项库仑摩擦，N·m | 仿真辨识值，真机须重采 |
| `itheta` | 45项惯量基参数；质量组合0:5，张量组合5:35，35:40未用，armature组合40:45 | 截断伪逆得到非唯一组合，含负值；不能逐项写入MJCF |

例如 `gtheta=[0,4.3711,-3.0214,3.0037,0.5005]`。负项反映该基参数化下的不可辨识性，不能解释为连杆负质量。原NPZ SHA256为 `225f2640a4dfd849c033865c17b2c9ac9fabe8c5ae1ced2da337287a757726a6`。

`Chain.mmat(q,itheta)` 重建5×5全耦合惯量矩阵；`stage_ff()` 生成：

```text
tau_ff = g_hat(q) + b_hat*dq + tc_hat*tanh(dq/0.05)
         + M_hat(q)*ddq + 选择性科氏项
```

输出 `ff_swing.csv`，供仿真 `swing ff_file=...` 使用。单关节对角惯量不足以覆盖J2/J3等关节的动态耦合。

本次在home及J2/J3各±30°的五个姿态检查中，拟合矩阵均正定，但不能据此认为全姿态有效：J2=-65°时，J1的 `M_hat[0,0]=1.4983`，CAD加仿真armature基准为0.24068 kg·m²，相差约6.2倍。现有数据适合验证辨识流程，尚不适合直接部署全工作空间动力学补偿。

## 3. 能否校正目前的角度误差

### J2：重力下垂的解释得到同步记录支持

使用已有真机记录 `20261026_140215_803308/samples.csv`，取第一次运动前最后稳定帧及之前0.5秒的125行。目标、反馈来自同一记录窗；五轴采样时间差最大0.186ms，仍不是原子硬件快照。

- SDK目标角：`[-1.4,35,-22.5,-88,0]°`。
- SDK反馈均值：`[-0.58787,15.12011,-16.82784,-90.68526,-0.15698]°`。
- 权威配置：`kp=[50,70,10,8,5]`、`kd=[10,30,5,4,1]`。
- 映射到MuJoCo角度域后，计算得到：

| 关节 | 位置P项 | 完整PD项 | 原CAD重力矩 | 新仿真辨识重力矩 |
|---|---:|---:|---:|---:|
| J2 | -24.288 | -24.244 | -24.523 | -24.783 N·m |
| J3 | -0.990 | -1.032 | -0.691 | -0.732 N·m |

J2静差为19.88°；辨识重力矩除以 `Kp=70` 的等效误差约20.29°。这高度支持“位置环需要依靠位置误差产生重力保持力矩，导致下垂”的解释。新旧模型对该姿态的重力估计只差约0.26N·m，**单纯换惯量参数不会让这约20°误差消失**。

以上PD矩是按配置增益重构的估计值，并非实测电机力矩。真实增益、执行饱和、摩擦和力矩刻度尚未独立标定，不能以这次量级吻合宣布物理零位已正确。

### J3与动态幅度偏差：仍需分别标定

J3完整PD估计与辨识重力矩仍差约0.30N·m，静摩擦、几何、零位和执行差异均可能参与，现有数据不能唯一分解。

此前30°方向测试中J2实际正向变化38.62°、J3为21.83°，且回程存在残差。这不能只用一个常量 `joint_offset` 校正。应分别检查独立物理角度、正反回差、稳态下垂和运动中的延迟/耦合。

### 对照界面：惯量不决定显示关节角

双视口将目标或SDK反馈经显示映射写入 `qpos`，再调用 `mj_forward` 重建姿态，没有独立 `mj_step` 动力学闭环。惯量更新不会直接改变它的角度显示。

当前显示关系为 `q_mj=[-q_sdk1,-q_sdk2,q_sdk3,q_sdk4+90°,q_sdk5]`，输入已经是SDK完成电机sign/offset转换后的角度。不要再叠加一遍电机偏移。方向模式左视口是“初始反馈＋命令增量”，底部保留原始SDK目标/反馈；它不是原始home目标的直接显示。

## 4. 如何用于后续校准与MPC

1. 用独立角度基准确认五轴物理零位、方向和几何映射；多姿态、正反趋近记录，区分常量零偏和姿态相关误差。
2. 先核实真实协议、力矩/速度刻度及实际增益。上游手册假设存在标准MIT `tau_ff` 字段；当前SDK的 `JointCmd` 只有pos/vel/kp/kd，新版77a8108也没有该字段。
3. 优先采真机多姿态双向稳态数据，辨识重力；再用恒速动作辨识摩擦，最后用同步动态数据辨识全耦合惯量。上游要求250Hz/4ms逐帧日志和实测延迟；现有方向测试记录没有完整提供逐帧量化命令、tff及参考加速度，不能直接替代专用辨识日志。
4. 若补齐真实力矩前馈接口，可先验证重力补偿，再加入摩擦与惯性项；验证范围逐步扩展。若只保留位置接口，需要另行设计参考位置补偿并离线验证，不能把力矩直接塞进位置字段。
5. MPC应使用经过留出姿态验证的 `M(q)、C(q,dq)、g(q)` 预测，同时保留SDK执行层和时间契约。现有 `SimBackend::arm_gravity()` / `arm_inverse()` 可读取更新后的物理模型，但当前仿真拟合NPZ不等于可直接替换模型的物理参数。

质量和质心决定重力项，转动惯量主要影响加速度项；稳态误差与动态误差需分开评价。公式与惯量坐标要求可参照[MuJoCo动力学](https://mujoco.readthedocs.io/en/stable/computation/index.html)及[惯性参数说明](https://mujoco.readthedocs.io/en/stable/XMLreference.html#body-inertial)。

SDK更新主要增加Vision/UWB串口接口，机械臂头文件、配置及arm/hal/core库未变化。`Robot`对象布局变化而库文件名未变，升级SDK时必须配套头文件与库重新编译，不能只替换旧程序使用的libsdk。

## 5. 本次验证与复现

Ubuntu新工作区已编译 `m2_intercept`、`robot_io_replay`、`mit_motor_sim`、`mit_arm_id_sim`、`m1_step_resp`及相关测试目标。

- 6组CTest通过：runtime_config、runtime_supervision、robot_io、mujoco_robot_io、x4_sdk_backend、arm_axis_check。
- 24项CLI、12项原生入口、7项传输、34项查看器检查通过；仅模拟和测试替身，没有启动硬件。
- 五轴辨识selftest通过，保持1秒后最大速度约8.37e-5rad/s。
- 对上游已保存数据复算：FK重力交叉核对最大差0.023N·m；挥拍J2 RMS由476.5mrad降至辨识前馈的9.9mrad。这是上游仿真数据的复算，不是真机补偿收益。
- 显示测试使用哈希绑定的 `x4_tennis_demo.mjcf`；辨识使用 `x4_tennis_scene.mjcf`。两个模型及原显示映射均未改。
- 首次检查遇到缺少xvfb-run、X11认证和分析环境缺matplotlib，已改用现有X11会话认证、正确demo模型及系统Python完成复验，未为适应测试修改模型或删除哈希检查。

复算上游保存数据的只读命令：

```bash
cd /home/ubuntu/Zhaohui/sim-x4-inertia-20261026/sckd_sim
./build-inertia-integration/mit_arm_id_sim selftest artifacts/inertia_integration_20261026/arm_selftest
/usr/bin/python3 python/tools/mit_arm_id.py eval/reports/mitarm --stage fk --out artifacts/inertia_integration_20261026/data_readback
/usr/bin/python3 python/tools/mit_arm_id.py eval/reports/mitarm --stage report --out artifacts/inertia_integration_20261026/data_readback
```

分析使用Ubuntu系统Python（numpy1.26.4、matplotlib3.6.3）；查看器用原项目venv。`grav/fric/inert/ff`阶段会向CSV目录写参数或前馈，重新辨识应使用独立数据目录，避免覆盖上游标称结果。

证据仅在Ubuntu：新工作区的 `sckd_sim/artifacts/inertia_integration_20261026/` 与 `sckd_sim/artifacts/inertia_read_audit_20261026/`。后者包含参数物理性抽查及 `latest_stable_gravity_comparison.json`。本轮没有完成真机零位、惯量标定或真实MPC闭环验收。
