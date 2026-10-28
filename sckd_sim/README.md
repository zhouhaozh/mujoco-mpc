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

## 羽毛球场景地图

场地定义集中在 `models/badminton_court.py`。`models/patch_demo.py` 原先的
`court_xml()` 网球场已替换：旧入口 `models/generated/x4_tennis_demo.{mjcf,xml}`
保留文件名兼容现有程序；X4_950 场景及其回放也使用相同羽毛球场。
尺寸依据 [BWF Laws of Badminton 第1节及 Diagram A](https://extranet.bwf.sport/docs/document-system/81/1466/1470/Section%204.1%20-%20Laws%20of%20Badminton%20-%2026%20April%202025%20V5.0%20%282%29%20.pdf)。

场地长 13.40 m，双打宽 6.10 m，单打宽 5.18 m；白线宽 0.04 m，尺寸包含标线。
本方底线外缘 `x=0`，对方底线外缘 `x=13.40`，球网 `x=6.70`，宽度中心 `y=0`。
短发球线靠网外缘距网 1.98 m；双打后发球线靠底线外缘距底线外缘 0.76 m。
场地包括两组边线、前后发球线及两半场各自的中线。
网中央高 1.524 m、两端高 1.55 m、网深 0.76 m，上沿白带宽 0.075 m。
网使用两片刚性几何近似下垂；四周各 2 m 的色块仅作视觉围区。

在工程根目录重新生成：

```bash
.venv/bin/python sckd_sim/models/build_model.py --variant x4_950
.venv/bin/python sckd_sim/models/build_x4_replay.py
.venv/bin/python sckd_sim/models/patch_demo.py
.venv/bin/python -m mujoco.viewer --mjcf=sckd_sim/models/generated/x4_950_scene.xml
```

X4_950 的场地和球网仅作视觉参考，保持原接触契约；旧演示保留球网/立柱碰撞。
本次更换地图不转换旧演示的网球飞行、回弹及发球参数，因此旧程序仍不是完整羽毛球动力学仿真。

## X4_950 独立模型（含执行器，控制器仍阻断）

在工程根目录运行：

```bash
.venv/bin/python sckd_sim/models/build_model.py --variant x4_950
```

产物为 `models/generated/x4_950_scene.mjcf`、同内容 `.xml` 和
`x4_950_manifest.json`。脚本只做加载、关节限位、执行器映射/力矩范围、质量、轮底及拍心检查，
不推进动力学；不会改写网球模型，默认变体仍为 `x4_tennis`。

源文件和 18 个网格来自 `models/source/x4_950/`，附带的 collision URDF
仅用于追溯轮子形状。初始底座高度为 0；Link6 保持固定，`racket_center`
位于其原点，`racket_face` 薄轴为 geom 局部 Z（接近 Link6 局部 X）。
碰撞只包含八个轮子、一个拍面和世界地面；车体和臂网格仅视觉。

模型有 17 个活动关节和 **17 个执行器**（`nq=24, nv=23, nu=17`），
执行器生成不代表控制器已启用。J2/J3 的归档源 URDF 限位仍全零。
2026-10-28 用户确认两轴行程均为 `[-3.14, 0.5] rad`，已记录在
`models/config/x4_950.json` 的 `confirmed_joint_position_limits_rad`；
构建器读取该字段，覆盖生成 MJCF 中两轴的 `range` 并设置 `limited=true`；
源 URDF 保留原值用于追溯，生成模型中的位置限位已生效。

独立电机资料为 `models/config/motors_x4_950.json`，配合 `x4_950.json`
中的 17 轴映射使用：J1/J2=RS04，J3/J4、底盘12轴=RS06，J5=RS05。
RS04/RS06 电机对象逐值复用 `motors.json` 的已有72V模型资料；RS05 按用户
授权采用手册48V模型，手册归档于 `docs/motors/RS05使用说明书251210.pdf`。
这不表示实机各电机供电已确认，也不把RS05缩放到72V。

RS05 额定/峰值力矩为1.6/5.5 N·m，减速比7.75:1，运控量程为位置
±12.57 rad、速度±50 rad/s、Kp 0–500、Kd 0–5。48V T-N点是读图近似，
0rpm平台及480rpm零力矩闭合为显式假设，`calibrated=false`。独立配置
选连续档，制动限幅1.6 N·m、零延迟、1ms更新和关节裁剪上限均单独标为
仿真假设；实际制动能力、额外机构限矩/传动和供电版本仍留空。
构建器读取该独立配置，按契约轴序生成 `gear=1` 的力矩执行器，对称
`ctrlrange` 取机构裁剪、所选力矩档和制动裁剪的最小值。当前连续档下
J1/J2 为 ±40 N·m，J3/J4 与底盘12轴为 ±11 N·m，J5 为 ±1.6 N·m。
MJCF 只落实静态力矩范围，不包含 T-N 包络、伺服增益或命令延迟；
独立运行接线见下节，控制器执行阻断保持不变。
其他未确认参数继续以 `null` 保留。MuJoCo 默认接触参数仅供预览，不是硬件标定值。
几何接口已适配：`ArmIk` 从模型 `home`（无则 `qpos0`）读取底座高度，
从 `racket_center` 读取拍心，从 `racket_face` 盒的正薄轴读取法向；
网球模型使用 Y 轴，X4_950 使用 Z 轴并包含 geom 旋转。
符号 FK 保留固定 Link6 的变换，只为 Joint1–5 分配角度；整机 MPC/OCP
按活动关节名读取限位。X4_950 的 FK、法向和 Jacobian 可用于几何检查，
生成模型已提供有效 J2/J3 限位，供这些接口按名读取；本阶段不调用控制器验证。
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
本阶段仅验证模型和配置，未运行控制器或推进动力学；没有添加羽毛球，
也未适配羽毛球飞行、冲量响应及地面逻辑；电机执行链仅完成接线，未做闭环验证。

## X4_950 独立运行接线（仍禁止控制器执行）

`models/config/runtime_x4_950.json` 用现有配置入口关联新模型、独立电机、
机器人和底盘配置；相对路径以 `sckd_sim/` 为工作目录。该预设不是可运行的
羽毛球控制器：保留入口、NMPC 初始化/生成和击球冲量阻断；指定的
`x4_950_nlp.so` 尚未生成，不能退回网球求解库。

- 物理、电机、底盘分配/轮速环统一1ms；命令延迟0，复用原延迟队列的直通分支。
- `truth_feedback=true`、`encoder_bits=0`；关节直接读真值，底盘偏移、噪声和IMU延迟置零。
  球观测接口在存在球时直接返回当前真值/时间戳；当前模型无球，未验证羽毛球预测。
- `x4_950.json.arm_servo` 保存J1–J5独立增益和准备姿态 `[0,-0.5,-0.5,0,0]`。
  增益按该姿态的模型对角惯量、假设ωn=4rad/s和ζ=1计算；仅未整定初值。
  新机仅保留模型重力前馈，不使用网球惯量前馈或辨识增益；NMPC播放补偿延迟取0。
- `chassis_x4_950.json` 采用新轮径0.0825m、双轮有效偏移0.024m和模型质量。
  舵增益同样按新惯量给出；轮速先用Kp=0.2、Ki=0，平移加速度0.4m/s²。
  这些是独立仿真假设，不能当作硬件性能或稳定性结论。
- T–N插值、方向判定及超速保护复用原MotorModel；抗饱和查询同一动态限矩。
  连续档底盘低速上限11N·m，高速推动按T–N降额，制动单独裁剪；候选积分不能越限。
  网球仍使用原配置、原36N·m抗饱和阈值和积分时序。
- 加载时核对X4模型/配置配套、周期/反馈假设及17轴绑定、gear=1和静态限矩。
  必要检查仅包含配置读取、静态模型及编译；伺服整定、动态稳定性和击球性能尚未验证。

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
