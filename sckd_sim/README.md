# sckd_sim — 网球/羽毛球对打机器人仿真系统(C++17 核心 + MuJoCo)

设计文档: [`docs/SIM_SYSTEM_DESIGN.md`](../docs/SIM_SYSTEM_DESIGN.md)(契约与决策的唯一来源)。
当前里程碑:**M0 内核**——模型管线、电机模型、HAL in-process、确定性、四项验收。

## 环境

- C++17:g++ ≥13 / CMake ≥3.16;MuJoCo 3.14.0 与 CasADi 3.8.0 的头文件和库默认位于工程根目录 `deps/mujoco`、`deps/casadi`,可用 `-DMUJOCO_DIR=`、`-DCASADI_DIR=` 覆盖
- Python:工程根目录 `.venv`(已在 Python 3.12 验证,mujoco==3.14.0、casadi==3.8.0、numpy),用于模型构建、离线工具和 viewer
- apt 依赖:libeigen3-dev、nlohmann-json3-dev

## 快速开始

```bash
# 从工程根目录安装依赖、编译并运行 CTest
python3 tools/setup_predictor_ubuntu.py --build-dir sckd_sim/build

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
