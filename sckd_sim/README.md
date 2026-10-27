# sckd_sim — 网球/羽毛球对打机器人仿真系统(C++17 核心 + MuJoCo)

首次部署请先按 [环境安装与部署](../docs/环境安装与部署.md) 执行一键安装和验收。

设计文档: [`docs/SIM_SYSTEM_DESIGN.md`](../docs/SIM_SYSTEM_DESIGN.md)(契约与决策的唯一来源)。
当前里程碑:**M0 内核**——模型管线、电机模型、HAL in-process、确定性、四项验收。

## 环境

- C++17:g++ ≥13 / CMake ≥3.16;MuJoCo 3.14.0 与 CasADi 3.8.0 的头文件和库默认位于工程根目录 `deps/mujoco`、`deps/casadi`,可用 `-DMUJOCO_DIR=`、`-DCASADI_DIR=` 覆盖
- Python:工程根目录 `.venv`(已在 Python 3.12 验证,mujoco==3.14.0、casadi==3.8.0、numpy),用于模型构建、离线工具和 viewer
- apt 依赖:libeigen3-dev、nlohmann-json3-dev

## 快速开始

```bash
# 从工程根目录安装依赖、编译并运行 CTest 和三球无头仿真
bash tools/install_ubuntu.sh

# 进入仿真目录运行 M0 验收
cd sckd_sim
bash eval/run_m0.sh          # 全量验收,报告 -> eval/reports/m0_report.txt
```

`run_m0.sh`默认使用工程根目录`.venv/bin/python`;使用其他解释器时可设置
`SCKD_PY=/path/to/python bash eval/run_m0.sh`。

分步:

```bash
# 1) 模型管线:URDF+STL -> MJCF + provenance manifest
../.venv/bin/python models/build_model.py

# 2) 构建 + 单元测试
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j
# 非默认依赖目录可在配置时追加:
#   -DMUJOCO_DIR=/path/to/mujoco -DCASADI_DIR=/path/to/casadi
./build/test_motor_model

# 3) 四项验收
./build/m0_determinism
./build/m0_step_rate
./build/m0_static_contact
./build/m0_arm_step_response        # CSV -> eval/reports/arm_step_response.csv
```

图形演示可从工程根目录运行`tools/run_mujoco_demo.sh [策略] [球数]`;
默认策略为`pfly`,默认运行3球。

## X4_950 独立模型（仅几何）

在工程根目录运行：

```bash
.venv/bin/python sckd_sim/models/build_model.py --variant x4_950
```

产物为 `models/generated/x4_950_scene.mjcf`、同内容 `.xml` 和
`x4_950_manifest.json`。脚本只做加载、关节映射、质量、轮底及拍心几何检查，
不推进动力学；不会改写网球模型，默认变体仍为 `x4_tennis`。

源文件和 18 个网格来自 `models/source/x4_950/`，附带的 collision URDF
仅用于追溯轮子形状。初始底座高度为 0；Link6 保持固定，`racket_center`
位于其原点，`racket_face` 薄轴为 geom 局部 Z（接近 Link6 局部 X）。
碰撞只包含八个轮子、一个拍面和世界地面；车体和臂网格仅视觉。

模型有 17 个活动关节但 **没有执行器**（`nq=24, nv=23, nu=0`），不能直接
交给 M0/M1/M2 控制程序。J2/J3 的源限位全零，导入后不限位，不代表真实行程。
缺失的电机、限位、伺服、摩擦等参数在 `models/config/x4_950.json` 中标为
`null`；不使用网球电机参数。MuJoCo 默认接触参数仅供预览，不是硬件标定值。
几何接口已适配：`ArmIk` 从模型 `home`（无则 `qpos0`）读取底座高度，
从 `racket_center` 读取拍心，从 `racket_face` 盒的正薄轴读取法向；
网球模型使用 Y 轴，X4_950 使用 Z 轴并包含 geom 旋转。
符号 FK 保留固定 Link6 的变换，只为 Joint1–5 分配角度；整机 MPC/OCP
按活动关节名读取限位。X4_950 的 FK、法向和 Jacobian 可用于几何检查，
但 J2/J3 限位未确认时，调用有界 IK 会抛异常，整机 MPC 构造和 OCP 入口的
限位读取也会拒绝继续；`WithinLimits` 返回 false，不补用旧行程。
NMPC 几何入口为 `third_party/nmpc6/robot_fk.h`：保留原 `sim_fk.h` 网球公式，
X4_950 使用独立 `x4_950_fk.h`，生成器、运行时末端复算和诊断按机器人选择。
仍为五轴、14 状态/7 控制，底盘 yaw=0；固定 Link6 不增加自由度。
`gen_nlp --robot x4_950 --geometry-only` 只导出当前目录下的
`x4_950_geometry.c`（位置、法向、速度和 Jacobian），在创建求解器之前返回。
不加 `--geometry-only` 的 X4_950 生成请求和 X4_950 NMPC 初始化均明确拒绝，
不会覆盖/加载网球 NLP 库。没有新增可运行的 X4_950 求解库。

接触几何入口为 `cpp/sim/racket_geometry.h`，从模型读取拍面中心、薄轴、
面内轴和半尺寸，供边界检查及诊断共用；网球保留原冲量公式和参数。
X4_950 拍球冲量在读取球状态前拒绝执行，不使用网球默认参数。
缺失的运动能力、羽毛球物理/接触参数及避碰约束在 X4_950 JSON 中继续以
`null` 列出；这些是待确认清单，填写 JSON 不会自动解除本阶段执行阻断。
本阶段仅验证几何，未运行控制器或动力学；没有添加羽毛球或执行器，
也未适配羽毛球飞行、冲量响应、地面逻辑及电机控制。

## 结构(对应设计 §10)

```
cpp/core      types.h(单位/坐标系/17 轴契约)、config(手册参数加载)
cpp/robot     motor_model(RS04/RS06 混合律+T-N 包络+9ms 延迟+锁存/使能)
cpp/hal       robot_interface(算法可见接口)、sim_backend(直连 MuJoCo C API)
cpp/apps      M0 验收可执行文件
cpp/tests     doctest 单元测试
models/       build_model.py(管线)、config/(motors.json+x4_tennis.json)、
              source/x4_tennis/(URDF+STL)、generated/(MJCF+manifest)
eval/         run_m0.sh、reports/
third_party/  doctest.h
```

## 关键契约速查(详情见设计 §4.2 / `cpp/core/types.h`)

- 单位 m/s/rad/N·m;车体系 +x 前 +y 左 +z 上
- 契约轴序 0..16:底盘 FL,FR,RL,RR ×(steer,wheel_L,wheel_R);臂 J1..J5 = 12..16
- 电机混合律 `τ = Kp(q_d−q)+Kd(dq_d−dq)+τ_ff`,限幅 = min(机构限矩, 峰值/连续档, T-N 包络(方向相关))
- 已知边界:T-N 为手册读图估值(`calibrated=false`);连杆惯量未辨识;地面摩擦名义值;
  **J5 滚转轴等效惯量 ~0.0011 kg·m²,armature=0 下 1 kHz 离散 kd 稳定界 ≈2.2 N·m·s/rad(M0 发现,转子惯量待 F3 辨识)**

## 下一步(M1)

舵轮分配 IK + 轮速 PI + 臂重力前馈伺服;里程计(轮式 + IMU ESKF)与噪声注入;轨迹跟踪场景。
