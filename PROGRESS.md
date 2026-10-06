# 项目进展记录(2026-10-06 更新)

## 一句话状态
**M2 闭环拦截验收通过(8/10 = 80%,门槛 ≥80%)**;M0/M1/M1b 此前已全过。
下一步:反弹模型标定、臂 NLP(取代五次多项式基线)、MLP 集成 A/B。

## 仓库
- 路径: `/home/ubuntu/sckd/sim`(git 最新 `b305ca4`)
- 仿真代码: `sckd_sim/`(C++17 核心 + Python 离线工具)
- 设计文档: `docs/SIM_SYSTEM_DESIGN.md`(v1.1,唯一决策来源)
- 参考工程: `/home/ubuntu/research/sckd-robot`(实机), `/home/ubuntu/sckd/trobot`(上位机)
- 环境: conda `sim`(```/home/ubuntu/miniconda3/envs/sim/bin/python```)

## 里程碑总览

| 里程碑 | 验收 | 关键数字 |
|---|---|---|
| M0 模型管线/电机/HAL | 7/7 PASS | 确定性逐位一致;88.7k steps/s;J5 kd 界 ~2.2 |
| M1 伺服/里程计/重力FF | 8/8 PASS | 臂静差 0.01°;航向融合 0.056 rad;停稳 0.05m/s |
| M1b 底盘 MPC | PASS | 1500 解零失败;p99 3.3ms;横向 0.124m |
| M2 球栈(验证①) | PASS | 干净反弹前 RMSE 0.099<0.10;早期 0.135<0.15;EKF vs CA +92.5% |
| **M2 闭环拦截** | **PASS** | **8/10=80%;击球点偏差 0.105m;触拍距离 0.05~0.09m** |

## M2 闭环修复全记录(0/10 → 8/10,共 9 个根因)

| # | 根因 | 修复 |
|---|---|---|
| 1 | 真值侧气动单位错(xfrc 写成加速度,阻力大 17.5×) | 系数 0.5ρCdA(力单位) |
| 2 | 观测延迟语义错(当前位置+过去时间戳) | sim_backend 延迟回放缓冲 |
| 3 | 预测器 v=0 初始化慢收敛 | 60ms 差分预热(σ_v≈0.7m/s) |
| 4 | 过程噪声过大(qv=2/qa=100) | qv=0.05/qa=2.0 |
| 5 | 场景不可命中(8m/s 平击越界早于管线 0.9s) | 喂球 4.5m/s/40°/spawn(4.8,-0.5,3.0) |
| 6 | 规划评分奖励晚击(log1p(margin)) | 裕度封顶+早击奖励+反弹前门控(NextBounceTime) |
| 7 | 重规划拒绝精化(高度偏好偏爱陈旧高 z 计划) | RefineAt:t_hit 锚定仅精化 p_hit |
| 8 | ArmIk 用活体 d(整车瞬移到 IK 姿态→假接触)+ 按指向角 yaw 解(偏 0.7m) | scratch mjData + yaw=0 |
| 9 | J1 增益在 9ms 延迟下抖振(kd=10);面内跟踪滞后 0.12m | 按轴惯量整定 + 惯性前馈 Î·q̈ + 挥拍 0.45s + 穿击提前量 0.045s |

验证方法:逐步验证(①预测器真前视 RMSE→②锁止击球点 vs 真值→③拍面系分解(法向/面内)→④命中);
拍面系分解(法向=盒薄轴)是定位 9 号根因的关键工具。

## 关键诊断工具(可复用)
- `m2_predictor`:0.3s 真前视 RMSE,5 段分解(干净反弹前/跨反弹/反弹后/早期闭环窗口)
- `m2_intercept` 诊断打印:[plan]/[lock](滤波vs真值)/[eval](IK复解vs实测、拍位、法向/面内分解)

## 待办(按优先级)
1. **反弹模型标定**:跨反弹 RMSE 0.73/反弹后 0.78 → 从 MuJoCo 接触参数拟合 e/μ,替换 BounceParams 占位
2. **臂 NLP**(CasADi nlpsol):终端击球流形(位置+拍面法向 n*+拍速),取代五次多项式基线;接口不变(SwingGenerator 同位替换)
3. **MLP 集成**:trobot EKF→MLP 管线接入 A/B(当前 CA 基线已被 EKF 大幅超越)
4. M3:移动击球、rally FSM、底盘 MPC 替换 P 控、航向窗
5. 羽毛球变体(x4_badminton,仅 J5 电机与拍惯量不同)

## 待用户提供(按优先级,不变)
1. 网球拍末端实际质量/惯量(拍 CAD 或称重)
2. 拍-球回弹实测(几组:挥速×拍面位置→出球速度)
3. 上位机接口原文档(DDS 22 值定义)
4. QR 视觉参数(θ_max/距离/FOV)
5. 实际来球分布(速度/旋转/落点)

## 文件速查

| 文件 | 内容 |
|---|---|
| `docs/SIM_SYSTEM_DESIGN.md` | 设计文档(全部决策/契约/风险) |
| `sckd_sim/eval/run_m0.sh` / `run_m1.sh` | M0/M1 验收(一键) |
| `sckd_sim/build/m2_predictor` | M2 验证①(预测器前视 RMSE) |
| `sckd_sim/build/m2_intercept` | M2 闭环验收(10 次喂球) |
| `sckd_sim/cpp/apps/demo_hit.cpp` | 击球演示(GUI 可看) |
| `sckd_sim/models/config/` | 全部配置(motors/variant/chassis/ball) |
