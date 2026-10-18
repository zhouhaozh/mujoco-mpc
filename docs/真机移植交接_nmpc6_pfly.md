# 真机移植交接:nmpc6 对齐 + pfly 新增(目标仓 sckd-robot)

> 写给执行移植的 agent:主仓 `/home/ubuntu/research/sckd-robot`,以 `/add-dir
> /home/ubuntu/sckd/sim` 双目录挂本仓。本仓 = 仿真验证主线(sckd_sim),下文
> 路径前缀「sim:」相对 `/home/ubuntu/sckd/sim`,「rob:」相对
> `/home/ubuntu/research/sckd-robot`。日期 2026-10-15;数字/结论均带出处,
> 引用前请核对源文件。
>
> **系统职责边界(用户 2026-10-15 定调)**:sckd-robot 是机器人本体控制代码,
> **只接收击球参数、执行动作,没有任何球相关感知**。感知/击球决策在上位机
> (视觉盒),经 30 字节串口包下发。三方分工:
>
> | 方 | 职责 |
> |---|---|
> | 上位机(视觉盒) | 球感知、预测、击球点/意图决策、走位目标 → 30B 包 |
> | sckd-robot(移植目标) | 解包 → 规划执行(本次移植的两控制器在此对拼) |
> | sckd_sim(本仓) | 闭环仿真验证、参数标定先验、对拍基准 |

## 0. 一页结论

1. **nmpc6 侧不是搬运,是「对齐修复」**。rob 已有
   `motion/MpcPlanner`(Phase 5b,2026-09-20):与本仓 `legacy_nmpc` 出自同一
   旧仓 badminton-mpc-6.0 nmpc.cc 的两条独立移植线(rob=运行时符号构建+
   Waypoint 消费+mpc_replay 对拍;sim=nlp.so codegen+精确 FK+显式状态码)。
   rob 缺的正是本仓补掉的:**X4-T1 网球几何 FK** —— 其代码默认 DH
   `dims={0.69,0.706,-0.1705,-0.85}`(rob: motion/include/motion/mpc_planner.h:73;
   实机 ini 部署值=0.685, 0.705, −0.1705, **−0.5** —— 2026-09-21 已换短拍,
   工程师确认 d5=0.5 为现状,0.69/0.706 系 9-08 过期快照值)仍是旧机
   集中参数式(d4=−0.1705 集中腕偏移),对本机分布式腕几何残差 0.1~0.3m
   级,数值验证见 sim: third_party/nmpc6/README.md「旧仓库 FK 差异留档」。
2. **pfly 侧是真新增**。rob 的 IPlanner 只有 mpc/manual/rl;pfly 执行栈
   (底盘参考剖面 + 臂 IK + Chebyshev 轨迹 QP 挥拍 + τ̂ 滞后补偿)整体作为
   新 IPlanner 移入。唯一硬依赖障碍:ArmIk 用 MuJoCo 雅可比
   (sim: cpp/control/arm_ik.h:4),须换解析式(材料现成,§3.2)。
3. **职责重划(因 sckd-robot 无感知)**:sim 里 hit_planner 的"采样候选+
   评分+RefineAt 精化"整段**不进机器人**——上位机已经干了等价的活(30B 包
   即其输出)。机器人侧 pfly = 参数变换器 + 执行栈:从 30B 包构造
   BaseRefGen::Schedule 与 ArmSwing 终值。缺口字段与选项见 §3.4。
4. **A/B 公平性天然成立**:两控制器吃同一 30B 包、同走位目标(包内
   robot_target)。上游决策质量是两者共同上限,真机 A/B 只比执行层。
5. **真机 A/B 的已知先验(诚实数字)**:sim 无误差补偿口径下 nmpc6 全面
   低于 pfly(fixed 65% vs 100%、rand 30-40% vs 70%、gs 60/40/0 vs 60/60/70,
   sim: USAGE.md:50 / PROGRESS.md:452-467)。差距结构性:末段速度-位置权衡
   (挥扫线系统性短左 0.15~0.3m)+ 羽毛球角窗不适配网球几何。旧机靠 ONNX
   误差网吸收前者 —— 误差网按本机几何**重训是必选项**(§2.4)。

## 1. 接口契约盘点(移植前逐条核对)

### 1.1 上游 30B 包(rob: hal/include/hal/vision_protocol.h)

帧 30 字节;字段:robot_current x/y/z(cm)、robot_target x/y(mm)、
hitpos x/y/z(m,场地绝对)、hitspeed、hitpitch、hityaw、hitspeedz、hittype
(int8)、dt(int16,距击球毫秒)。

**与两控制器的输入映射**:

| 包字段 | nmpc6(sim: cpp/control/legacy_nmpc.h:46-63 Request) | pfly 需要的量 |
|---|---|---|
| hitpos x/y/z | target[3] | HitPlan.p_hit |
| dt | dt[](=剩余/3 均分,sim 口径 m2_intercept) | t_hit = now + dt |
| hitspeed | hit_spd(拍末端 vx 目标) | v_out_speed(意图) |
| hityaw / hitpitch | hit_yaw_deg / hit_pitch_deg(法向目标构造) | v_out_yaw/elev → n_face 构造 |
| hitspeedz | hit_speed_z(拍末端 vz 目标) | (v_racket z 分量) |
| robot_target x/y | (旧机由上游走位保证;sim 用 base_anchor 等价) | BaseRefGen 终点(或锚) |
| hittype | (角窗分档:前场/正常,HitProfile 两档) | hit_type |
| **包里没有** | v_in(来球速度@t_hit)、v_base*(fly 速度意图)、连续球轨迹 | ← pfly 缺口,§3.4 |

### 1.2 rob 侧规划接口(rob: motion/include/motion/planner.h)

- `PlanRequest`:robot_x/y **mm**(注意!)/hit_* m 场地绝对/θ1=J1+车体 yaw/
  joint_vel_cmd(ω3..5 用目标速度而非反馈,rob 侧 plan_thread 回填)。
- `WholeBodyPlan::Waypoint`:t=距击球秒数**倒计降序**,消费端锚 (t_hit−now)
  选段(与重解频率解耦);twist+joint_pos[5]/joint_vel[5]。
- 消费端安全窗 `MpcPlanner::UnionJointWindow`(rob: mpc_planner.h:30-32,
  2026-09-24 越窗案:稠密路径重积分无窗校验,J3 cmd −240° vs 窗 [−144,0])
  —— **pfly 移入后同样要接此窗**(sim 侧无对应物,sim 靠 QP 约束内保证)。

### 1.3 单位/坐标差异表(装配时逐行核对)

| 项 | sim(sckd_sim) | rob(sckd-robot) | 动作 |
|---|---|---|---|
| 底盘位置 | m(世界系) | PlanRequest **mm**;vision robot_ cm | 换算 |
| 拍面法向来源 | hit_planner ReboundInverse(v_in,v_out) | 30B 包 hityaw/hitpitch(旧机口径) | pfly 需适配(§3.4) |
| 臂关节 | MJCF q1..q5 恒等映射 | θ=sign·motor+offset(rob: arm/include/arm/types.h:77-80) | 零位偏置激活纪律(§4.1) |
| 待机 | sim J2=J3=−35°(commit 1d914cb,FK 验证) | home_deg=−1.4,35,−22.5,−88,0(rob ini,J2=+35°=抬臂) | 已同构,激活偏置时同批迁移;注意两系 J2 抬臂符号相反(sim=负,rob/旧机=正)——nmpc6 角窗是旧机系,sim 现状 0 解全回退即此(「角度窗待办」) |
| 关节限位(URDF/MJCF) | J2/J3 range [−3.14,0.5] rad;J1/J4/J5 continuous 无限位;velocity 标注=0(占位垃圾) | RS04/RS06 位置无硬限(协议量程在速度域);真机位置约束全在控制层(MPC 角窗+守门+UnionJointWindow),机械靠插销孔/外壳 | URDF J2 上界 +28.6° vs 待机抬臂 +35°(rob 系)=已知矛盾(零位/符号映射未对表,REAL_MACHINE_PARAMS ⚠️);URDF 限位不可作真机保护依据 |
| 臂执行帧(两仓核实 2026-10-16) | MotorCmd 含 tau_ff 通道 | **MIT 帧无力矩槽位**:旧仓 MotionSet=4×16bit(pos/speed/kp/kd,robot_control.cc:3243);rob EncodeMit 同五参无力矩(arm/server.cc:187) | 力矩前馈上真机的**前置问题=固件是否支持带 T 位的标准 MIT 帧**(RobStride 官方 MIT 含 12bit 力矩位,旧仓 4×16 系 72V 定制变体)——供应商确认前,前馈只存在于 sim;速度前馈(speed 字段)已在用 |
| 速度上限 | 72V 口径 T-N 包络 | 实机**已确认 72V**(用户 2026-10-15;RS04 空载 36 rad/s 口径) | sim 包络直接适用;du_max/jv 起步用现值,台架实测后微调 |

## 2. nmpc6 侧:对齐修复 MpcPlanner

### 2.1 FK 换血(核心工作)

推荐路线:**保持 rob 架构,只换几何** —— rob `mpc_solver.cc BuildSolver`
的符号 FK/动力学用 sim: third_party/nmpc6/gen_nlp.cc 逐式替换(它就是
set_my_nmpc_solver 的 X4-T1 几何版:闭式 5 旋转链 + 符号雅可比速度
J·ω+plat·(vx,vy,0),常数在 sim_fk.h 全精度);数值 rollout FK
(`MpcPlanner::EndPos`,旧 GetEndPos2)同步换 sim: third_party/nmpc6/sim_fk.h
的 `SimArmPos`(对 MuJoCo 300 随机位形 max|err|≈4e-4)。

备选路线:rob 曾砍掉的 nlp.so 加载分支复活,直接挂 sim:
third_party/nmpc6/nlp.so(27 参/87 约 codegen,与驱动布局强绑定)。收益=
两仓数值函数逐位同源;代价=维护两套装配链。**不推荐先走**,留作对拍基准用。

**验收对拍口径(重要)**:
- 旧机 MPC_Log 对拍(mpc_replay 原设计)在换几何后**不再逐位一致**(几何
  变了)——先在羽毛球 dims 下跑通对拍回归(证明装配链无移植错),再切
  网球几何;
- 网球几何下以本仓为基准:sim: third_party/nmpc6/README.md 的 ctest 回归
  (实机日志参数用例,严格档 e≈(0,0.014,0.011))与 sim: cpp/tests/
  test_legacy_nmpc.cc 用例参数移植成 rob 侧 replay 用例,rob 求解器 vs
  sim nlp.so 同参数互证(SolveWithProblem 直喂 27 维 p,rob:
  mpc_planner.h:237)。

### 2.2 参数网球化清单(rob Params 默认=旧机羽毛球值)

| 参数组 | 现值(出处) | 动作 |
|---|---|---|
| dims | 代码默认 {0.69,0.706,−0.1705,−0.85};**实机 ini=0.685,0.705,−0.1705,−0.5(短拍已换)** | 换 sim_fk.h 常数(随 §2.1);换拍后 sim 模型拍长真实性待复核(2026-10-06 交叉校核只覆盖底盘/电机映射) |
| end_z_min | **rob ini=0.65**(短拍净空语义,Δ+0.137 重标);sim/旧机=0.35 | 以 rob 0.65 为准(真机实测口径);sim 侧 0.35 未跟 |
| j4_min | **rob ini=−270**(2026-09-21 触地跑:解骑 −360 下限→执行过冲 −422° 整圈卷腕,收紧留 63° 余);sim 仍 −360 | 以 rob −270 为准;sim legacy_nmpc 未跟 |
| j2_min | rob ini=+10(首轮 −30 实证鞭打过冲 43° 插轮区,"限位须按计划限−过冲包络设计") | 同值;sim 同为 +10,一致 |
| 工作空间 x∈[−1,5.7]/y±2 m | 旧机羽毛球场尺度(旧 :1046-1110 原值) | 网球场上按场地+臂可达重推;锚窗 ±0.4 平时掩盖它,全场自由落位工况会暴露 |
| HitProfile 角窗 | j2∈[10,60/70]°、j3∈[−144,0]° 等(rob: mpc_planner.h:36-52) | 羽毛球挥拍风格,sim 已证对网球低近球适配差(PROGRESS.md:458);按网球几何重设(先验:sim 待机 J2=+35° 在窗内是 nmpc6 待机 70% 的原因) |
| ACC/du_max | acc={8,8,30,30,180,400,200},du_max={4,4,5,3,40,50,10} | 电压已确认 72V(用户 2026-10-15),sim 72V 口径包络直接适用:起步用现值,台架实测后微调(sim 侧调参入口同表:gen_nlp.cc) |
| 拍速标定 | 旧机全速=直驱腕+羽毛球几何 | sim 标定 N6_SPD=0.8(1.0/0.9/0.8/0.65→60/60/80/60,PROGRESS.md:445);真机以 0.8 起步扫 |
| vy_offset/jv 等 | 旧机值 | 台架阶段实测后调 |

### 2.3 sim 三处适配的真机取舍(sim 加的,真机怎么办)

| 适配 | sim 原因 | 真机取舍 |
|---|---|---|
| base_anchor ±0.4m(legacy_nmpc.h:55-62) | sim 无上游走位,不加则底盘停 1.3m 外臂全伸展 | **保留,锚=30B 包 robot_target**(等价旧机上游走位);A/B 时 pfly 同用此目标,公平 |
| 目标锁止 t_hit−0.15 | RefineAt 逐拍精化让末段 e 0.005→0.245 解退化 | 上游 hit_* 本身慢更新(旧机"慢策略层天然准静态");若上位机逐帧刷新 hitpos,则需同款锁止门(剩余<0.15s 冻结 target) |
| 末段接受门(<100ms 仅 严格+成功+<10ms 新解) | 旧机 MpcThread 原语义 | rob Params 已有(accept_window_ms/solve_accept_ms),核对数值 |
| 连败 0.75s 回退 B′ | sim 新增 | **真机 A/B 时关闭**(纯对照);或两控制器统一带同一回退 |

### 2.4 误差网(必选项,数据工作量最大)

- rob 已有挂点:[MpcErrorNet] ini + motion/src/error_net.cc(单 xyz ONNX,
  clamp_x=[−0.02,0.16] 等不对称界,rob: mpc_planner.h:158-165)。
- **旧羽毛球网不可用**:其吸收的系统性偏差部分来自旧 FK 几何误差,换
  sim_fk.h 后残差结构变化;且本机臂+网球拍动力学不同。
- 采数协议=发球机 fixed 族逐发记录拍-球有符号误差(§5 阶段 1 一石二鸟)。
- 替代/过渡:sim 量化过的**在线 comp 学习**(拍@t_hit−锁定点 有符号误差
  IIR,z 分双槽,T<0.6 末段渐入)可把 fixed 拉到 60-80%、拍-球误差
  −0.27→±0.1(PROGRESS.md:449-451);无采集周期,可先行上线。
- 留档两坑:补偿符号抄反越补越短(−0.68 发散);全量首解注入推出可行域
  连败回退。clamp 不对称正界 [−0.02,+0.16] 本身就是"挥扫线系统性短左"的
  病征记录,重训后应趋对称。

### 2.5 求解/时序先验(从 sim 带走)

- 解时:冷解 30ms / 热解 3.7ms(sim 实测);重解 50Hz 稳定,100Hz 反退
  (RefineAt 同频抖动+计划置换,执行层跟不上)——rob plan_thread 4ms 节拍
  下注意 `plan_refresh_ms` 节流(rob: mpc_planner.h:145-150,J1 kp50 极限环
  ±1°/±25dps 实证,自参考闭环过紧)。
- IPOPT:rob 默认 linear_solver=MA57(需 HSL);无 HSL 环境切 mumps(sim
  默认,mu_strategy=adaptive 等调参两仓一致)。

## 3. pfly 侧:新增执行栈 IPlanner

### 3.1 模块映射表

| sim 文件 | 职责 | rob 落点 |
|---|---|---|
| planning/hit_planner.{h,cpp} | 候选采样/评分/RefineAt(感知侧规划) | **不移植主体**;仅取 ReboundInverse/可达环带等纯函数供参数变换用(§3.4) |
| control/base_ref_gen.h | 底盘到位/fly-through 参考剖面(header-only,绝对时刻锚定) | 直接拷入;终点=robot_target(30B 包) |
| control/arm_ik.{h,cpp} | 拍心 IK/法向/J(依赖 MuJoCo) | **重写为解析版**(§3.2) |
| control/arm_swing.{h,cpp} | Chebyshev 5×9 轨迹 QP 挥拍+τ̂ 闭环滞后补偿+锁止微调 | 直接拷入(QP 数据无依赖);SwingCmd→Waypoint 适配(§3.3) |
| control/gen/{arm_qp_data.h, arm_nlp.json} | QP codegen 常数 | 直接拷入(无 MuJoCo/casadi 依赖) |
| control/swing_generator.h | SwingCmd/HitPlan 类型 | 随 arm_swing 带 |
| estimation/ball_predictor.{h,cpp} | 球轨迹预测(aero+bounce) | 不进机器人(感知在上位机);上位机改进时参考 |
| sim/analytic_contact.h | 拍-球解析冲量(rebound_a=1.5/b=−0.5,k=0.5 占位) | 不进机器人;k 值影响上位机意图↔拍速映射的标定口径 |

### 3.2 ArmIk 去 MuJoCo(唯一硬重写)

sim ArmIk 用 mj_jacSite/mj_forward。替换材料:
- 位置 FK:sim_fk.h `SimArmPos`(模板函数,可直接编入 rob);
- 雅可比:gen_nlp.cc 已有 SX 符号 J(速度=J·ω+plat·(vx,vy,0))——数值化
  即得 3×5 J。注:旧机手写 J 丢 ω2/ω5 项、plat 漏 vy 的近似在 sim 换 FK
  时已消除(符号精确);rob BuildSolver 照搬 gen_nlp.cc 即得同款精确 J。
  旧近似仅存在于旧机 nlp.so/旧源码——对拍旧 MPC_Log 时留意此差异来源;
- 法向 FK:sim_fk.h 链末 R·e_y(拍面法向=薄轴 y);
- IK 本体(sim: arm_ik.cpp 有界 DLS)算法照搬,只换 FK/J 后端。

### 3.3 执行层映射(pfly 的 Waypoint 生产)

- ArmSwing 播放输出 SwingCmd{q,dq,ddp,增益} → rob Waypoint{t=倒计,
  joint_pos/vel}(twist 由底盘参考剖面 EvalAt 世界系→车体系析出);
- 增益/ddp(惯性前馈)在 rob 的 arm 驱动若有等价前馈通道则映射,无则
  先丢(ddp 影响小于 τ̂,sim 口径 τ̂ 为主滞后);
- **τ̂ 闭环滞后补偿必须带**(sim: arm_swing.h:24-28,τ̂=e·dq/|dq|² 最小
  二乘时间移位,替代常数提前量;kLead=0.020 仅兜底)——sim 首版符号抄反
  致拍提前冲过截面 2/10 回归,注意方向;
- 锁止语义:剩余<0.15s 锁终值(kLockMargin),锁后仅 ≤0.05 rad/关节微调
  (kMicroAdj)——正好消费上游 hit_* 慢更新流(§1.3);
- QP 起挥延迟 kInitDelay=0.25s 是 sim 平台中心(0/0.20/0.25/0.30/0.35 →
  40/86.7/93.3/70/60%,sim: m2_intercept.cpp:218-221)——真机伺服带宽不同,
  台架阶段重扫。

### 3.4 上游参数变换器 + 缺口选项(设计决策点)

30B 包 → pfly 内部量的变换(机器人侧新写的胶水层,替代 sim 的
hit_planner.Plan):

| pfly 需要 | 来源 | 选项 |
|---|---|---|
| p_hit / t_hit | hitpos / now+dt | 直接 |
| base 终点 | robot_target(mm→m) | 直接(与 nmpc6 锚同源,A/B 公平) |
| n_face(拍面法向) | **包内无 v_in**,ReboundInverse 缺输入 | **已定(用户 2026-10-15):选项 A**——法向目标直接由 hitpitch/hityaw 构造(与 nmpc6 的 hit_yaw_deg/hit_pitch_deg→法向同源);B(上位机协议给 v_in)/C(帧间差分估 v_in)留档备查,精度不足时升级 |
| v_racket(拍速目标) | hitspeed/hitspeedz | 模长=hitspeed(经 N6_SPD 类标定折减),法向对齐后定向 |
| v_base*(fly-through) | **包内无** | 起步:关 fly(定点口径,sim M2 基线);后续:上游给或按 robot_target−robot_curr 与 dt 自算(B′ 投影规则可搬) |
| hit_type 角窗 | hittype | 直接 |

### 3.5 pfly 已知坑(带自 sim 留档)

- IK 必须独立 scratch 状态:sim 用独立 mjData,直接写活体会把整车瞬移到
  IK 姿态(2026-10-06);解析版天然免疫,但注意 IK 求解别污染反馈滤波。
- 纯速度模式臂失控:sim 实测 kp=0 时 dq 冲 24 rad/s——**臂必须位置+速度
  hybrid**(rob arm 驱动本就 hybrid,核对增益来源)。
- 首帧 RefineAt 大改终值 → 挥拍中途大改必失稳(kMicroAdj 门即为此)。
- 观察窗污染计数:GUI 判"翻转先于首次落地"才计中(sim commit 9d10886);
  真机统计口径要提前定义,否则 A/B 数字不可比。

## 4. 共性前置(两控制器都要,先行)

1. **零位/符号激活纪律**:rob joint_sign/joint_offset 缺省全 0(旧行为),
   激活须与模型系 ini(home/prep/限位)**同批迁移**,单改映射会变行为
   (rob: arm/types.h:77-79)。sim 侧待机=实机 home 同构已核(commit
   1d914cb:电机零位≠模型零位,ini joint_offset/-sign 待实机对表)。
2. **电压已确认 72V**(用户 2026-10-15,解除 REAL_MACHINE_PARAMS ⚠️):
   RS04 空载 36 rad/s 口径,sim 的 72V T-N 包络直接适用;du_max/jv/ACC/QP
   速度限位按文档现值起步,台架实测后微调。
3. **伺服标定**:τ̂ 逐机辨识(sim 口径=播放时钟最小二乘移位);增益按 rob
   docs/arm-joint-tune-runbook.md;J5 限矩 5.5 N·m。
4. **拍-球回弹 k=0.5 为文献占位**:上位机意图↔拍速映射的标定依赖实测
   (待用户提供回弹数据,sim PROGRESS.md「待用户提供」#1/#2:拍末端质量
   惯量、挥速×拍面位置→出球速度)。
5. **节拍**:plan_thread 4ms 重解、消费端 (t_hit−now) 锚定选段;nmpc6 侧
   plan_refresh_ms 节流先例在案。

## 5. 真机 A/B 协议

### 5.1 阶段(每阶段内两控制器交替、同球组)

| 阶段 | 内容 | 通过判据 |
|---|---|---|
| 0 台架 | 静止/吊位固定球:零位对表、FK 对拍(实测拍心 vs SimArmPos/EndPos 预测)、Plans→伺服跟踪(指令 vs 实测轨迹) | 末端跟踪误差 <5cm 量级;FK 残差 <2cm |
| 1 发球机 fixed | 定点族;恒零补偿;**逐发记录拍-球有符号误差(=误差网采数)** | 机制闭合:解数/拒收/回退统计健康 |
| 2 挂补偿 | 误差网首轮 或 在线 comp 学习 | fixed 族命中率显著抬升(sim 先验:60-80%) |
| 3 rand / gs | 全域随机/平抽族 | 出 A/B 主表 |

### 5.2 指标与判据(与 sim 口径对齐)

- 命中判据(sim 等价:方向改变 + min_dist<0.30 + 落地前):真机=视觉回放
  判出球方向改变 + 落点在场内;拍-球 min_dist 用 Vicon(有
  /home/ubuntu/sckd/vicon-stream)或高速相机。
- 指标:命中率 / 出球速度(意图 8.0 的实现率)/ 拍-球误差三轴 / 底盘
  贡献占比 / 解统计(nmpc6:solves/rejects/ms_max;pfly:solve_ms)。
- 回退语义统一:阶段 3 前关两控制器回退(纯对照)。

## 6. 本仓可拷贝资产清单

| 资产 | 路径(sim:) | 用途 |
|---|---|---|
| 精确 FK(数值+模板) | third_party/nmpc6/sim_fk.h | FK 换血+ArmIk 重写后端 |
| OCP 符号生成器(SX 网球几何版) | third_party/nmpc6/gen_nlp.cc | BuildSolver 逐式替换源 |
| nlp.so codegen + 重生成流程 | third_party/nmpc6/{nlp.c,nlp.so} + README.md | 对拍基准(勿改公式单方) |
| nmpc6 驱动(守门链/显式状态/base_anchor/1ms Plans) | cpp/control/legacy_nmpc.{h,cpp} | 参数装配/适配语义参考 |
| nmpc6 回归用例 | cpp/tests/test_legacy_nmpc.cc(ctest -R legacy_nmpc) | 移植成 rob replay 用例 |
| pfly 执行栈 | cpp/control/{base_ref_gen.h, arm_ik.*, arm_swing.*, swing_generator.h, gen/} | §3 主体 |
| 反弹/冲量模型口径 | cpp/sim/analytic_contact.h | 标定口径对齐(k 值同源) |
| 建模对比审阅稿 | docs/nmpc6_vs_pfly_建模求解.md | 两法公式/权重/约束全景 |
| A/B 口径与数字 | USAGE.md / PROGRESS.md | 判据/指标先验 |

## 7. 坑索引(按主题,两仓留档)

| 主题 | 坑 | 出处 |
|---|---|---|
| 求解装配 | 旧仓残留 nlp.so 是陈旧版(26参/45约),源码才是真源;生成文件名不得含目录斜杠 | sim: third_party/nmpc6/README.md |
| 消费端 | Waypoint 按"计划盖章后流逝时间"插值 → 4ms 重解下采样恒停计划开头(挥拍变爬行);必须 (t_hit−now) 全局倒计时锚定 | rob: planner.h:17-20 |
| 反馈-重解环路 | 225Hz 重解自参考闭环过紧 → J1 极限环;plan_refresh_ms 节流 | rob: mpc_planner.h:145-150 |
| 统计 | SolveStats 每拍重置把成功擦成"0 解"假象;另立回合累计口径 | rob: mpc_planner.h:219-222 |
| 窗校验 | 稠密路径重积分不查窗 → 鞭打外推越窗;消费端 UnionJointWindow | rob: mpc_planner.h:30-32 |
| 执行 | 纯速度模式臂失稳(dq 24 rad/s);τ̂ 符号反=提前冲过截面;IK 污染活体状态 | sim: PROGRESS.md:437-438、arm_swing 留档、m2_intercept.cpp:285-288 |
| 频率 | nmpc6 100Hz 反退(RefineAt 同频抖动);50Hz+τ̂ 折中 | sim: PROGRESS.md:439-441 |
| 误差补偿 | 符号反发散 −0.68;全量首解注入推出可行域;clamp 不对称=系统性短左病征 | sim: PROGRESS.md:449-458 |
| 慢放/时窗 | dt 拉长改变最优解本身(倒退助跑+缠腕疯计划);dt2=视野窗≠墙钟 | rob: planner.h:40-46 |
| 判据 | 观察窗"没落地"≠"落地前击中"(GUI 计数污染);统计口径须先定义 | sim commit 9d10886 |
