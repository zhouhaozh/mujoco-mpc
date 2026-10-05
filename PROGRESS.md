# 项目进展记录(2026-10-05 session 结束)

## 一句话状态
M0/M1/M1b 全部通过验收;M2 模块级全部编译通过、单测全绿,**闭环拦截未通过(0/10)**,根因已定位(预测器收敛速度+挥拍时序),修复路径明确。

## 仓库
- 路径: `/home/ubuntu/sckd/sim`(git 已管理,最新 `136b7f8`)
- 仿真代码: `sckd_sim/`(C++17 核心 + Python 离线工具)
- 设计文档: `docs/SIM_SYSTEM_DESIGN.md`(v1.1,唯一决策来源)
- 参考工程: `/home/ubuntu/research/sckd-robot`(实机), `/home/ubuntu/sckd/trobot`(上位机)
- 环境: conda `sim` 环境(```/home/ubuntu/miniconda3/envs/sim/bin/python```)

## 已完成(全部提交、测试全绿)

| 里程碑 | 验收 | 关键数字 |
|---|---|---|
| M0 模型管线/电机/HAL | 7/7 PASS | 确定性逐位一致;88.7k steps/s;发现 J5 kd 界 ~2.2 |
| M1 伺服/里程计/重力FF | 8/8 PASS | 臂静差 0.01°(有FF);陀螺融合航向 0.056 vs 1.88 rad;停稳 0.05m/s |
| M1b 底盘 MPC | PASS | 1500 解零失败;p99 3.3ms;横向 0.124m;速度 0.140m/s |
| 击球演示 | 可运行 | IK 0.8mm;球触拍回击(拍面不受控→高飞 44m) |
| M2 球栈 | 单测 5/5 | 气动/发球器/EKF+反弹分段 vs CA(反弹段 74% 提升) |
| M2 规划 | 编译通过 | 命中点/ArmIk(0.8mm)/SwingGenerator |

## M2 闭环失败根因与修复路径

### 根因
1. **预测器收敛慢**: v=0 初始化,P_v=100 → 需 300+ 帧才收敛;200ms/60帧时速度仍差 → 预测球道偏离真值 → 击球点错 → 拍够不到
2. **挥拍时序**: SwingGenerator.Init 用规划时刻的 t_hit(相对时间);重规划更新 plan 但 swing 不重启 → 时序脱节
3. **集成验证不足**: 各模块独立测试通过,但预测器在闭环中的实际精度未先行验证

### 修复路径(按序)
```
a. 预测器初始化:首观测后用 3 帧差分估计初速(代替 v=0),P_v 相应减小
b. 挥拍重构:Update 改用绝对仿真时刻(非相对);重规划只更新 q_hit 不重启 swing
c. 逐步验证:
   ① 独立验证:固定发球 → 300Hz 观测 → EKF 预测 0.3s 前视 RMSE < 0.1m
   ② 半闭环:预测准了 → 规划出的击球点 vs 真值球道偏差 < 0.2m
   ③ 全闭环:加 IK + 挥拍 → 拍-球最小距离 < 0.15m
   ④ 验收:10 次定点喂球 ≥80% 命中(方向改变)
```

## 下次 session 起点

```bash
cd /home/ubuntu/sckd/sim/sckd_sim
# 1. 恢复环境
source project_env.sh  # 如有
# 2. 确认构建/测试仍绿
cmake --build build -j && ./build/test_ball && ./build/test_swerve && ./build/test_motor_model
# 3. 开始修复
vim cpp/estimation/ball_predictor.cpp  # ①改进初始化(3帧差分)
vim cpp/control/swing_generator.cpp    # ②改用绝对时刻
vim cpp/apps/m2_intercept.cpp          # ③加逐步验证
```

## 关键架构决策(详见设计文档)

| 决策 | 要点 |
|---|---|
| 引擎 | MuJoCo 单引擎双后端(CPU+MJX) |
| 算法核心 | C++17,Python 仅离线工具(codegen/训练) |
| 执行器 | 全 17 轴混合律(非速度环);三层等价(MPC 见等效一阶模型) |
| 增益迁移 | 仿真增益=保守初值;实机行为对齐;数据流"实机标定→仿真" |
| 击球分解 | 意图(位置+出球速度) → 反弹逆解(v*,n*) → 底盘+臂 |
| 航向窗 | MPC 软约束 |ψ−ψ_qr|≤θ_max(±30°占位,待视觉参数) |
| 预测器 | 物理EKF+反弹分段(主线);MLP(trobot)基线;混合方案待管线全通后 |
| MPC 频率 | 底盘 QP 50Hz + 臂 NLP 50Hz;伺服 1kHz;与旧版 250Hz 口径的差异有实测数据 |

## 待用户提供(按优先级)

1. 网球拍末端实际质量/惯量(拍 CAD 或称重)
2. 拍-球回弹实测(几组:挥速×拍面位置→出球速度)
3. 上位机接口原文档(DDS 22 值定义)
4. QR 视觉参数(θ_max/距离/FOV)
5. 实际来球分布(速度/旋转/落点)

## 文件速查

| 文件 | 内容 |
|---|---|
| `docs/SIM_SYSTEM_DESIGN.md` | 设计文档(全部决策/契约/风险) |
| `sckd_sim/eval/run_m0.sh` | M0 验收(一键) |
| `sckd_sim/eval/run_m1.sh` | M1 验收(一键) |
| `sckd_sim/cpp/apps/demo_hit.cpp` | 击球演示(GUI可看) |
| `sckd_sim/python/tools/demo_viewer.py` | GUI passive viewer |
| `sckd_sim/models/config/` | 全部配置(motors/variant/chassis/ball) |
| `sckd_sim/models/generated/` | 生成的 MJCF + manifest |
