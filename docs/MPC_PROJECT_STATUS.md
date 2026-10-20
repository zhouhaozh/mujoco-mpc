> 2026-10-20 GitLab协作整理：保留原73个提交、原团队与预测分支；当前Ubuntu源码形成模块化分支提交，初次上传快照单独归档并关联到默认main。操作范围为版本管理，未重跑控制验收；见[分支协作](../BRANCH_COLLABORATION.md)。

> 2026-10-19 团队主线：分支 `codex/pipeline-modularization` 已实施首轮同步模块化，10/10 CTest、9次端到端运行通过；最终旧新控制对照17/18一致，wbmpc有60ms墙钟守门差异，未验收异步/实时性。完整记录见[实施记录](PIPELINE_IMPLEMENTATION_20261019.md)。下文保留历史阶段说明。

# MPC 项目总览：进度、接口、说明与未来计划

更新：2026-09-30（北京时间）。当前主线为 **X4-T1 / MuJoCo 整机仿真与后续独立 C++ MPC 接入**。本文依据当前源码、已有验收数据及本次 Ubuntu 文件检查；历史测量与本次交付检查分别标注。

## 1. 当前结论

整机模型、17 轴电机控制、独立仿真服务、TCP/C SDK、网页控制台及测量工具已可运行。Ubuntu 主目录为 `/home/ubuntu/mpc_sim`。**C++ MPC 求解器尚未接入当前 MuJoCo 主线**；已有 C++ 程序是通信与控制接口性能基准。

原生 C++ 三 RPC 循环接近 500/1000 Hz，但长测仍有超时和跳过槽，尚未达到每个周期均满足截止时间的要求。厂家 robotd、原遥控/IMU 链路和实机动力学标定仍待完成。

旧 PyBullet NMPC、Isaac Lab MPC/FastDDS/PPO 是历史工程与迁移参考。旧平台已有回放不能证明新 MuJoCo MPC 已完成；PPO 也不是 SAC。原五天计划保留为历史路线，其中“面板未实施”等文字已滞后，现状以本文和最新报告为准。

## 2. 进度与验收证据

| 阶段 | 成果 | 验证及边界 |
|---|---|---|
| 旧工程研究与回放，9 月 22 日起 | PyBullet NMPC、Isaac MPC/DDS/PPO 接口梳理；保存 MPC/PPO 视频及球状态日志 | 旧平台结果，不能当作新平台控制或正式命中率验收 |
| 模型基线，9 月 23 日 | URDF→MJCF、浮动底盘、17 轴映射、质量惯量/方向/接触检查 | [Day1](docs/day1_report.md)；数值合法不等于实机参数准确 |
| 电机和底盘，9 月 24–28 日 | 五轴混合控制，8 轮 4 舵速度 PI、运动学、联动和日志 | [Day2](docs/day2_report.md)、[Day3](docs/day3_report.md)；电机/摩擦参数仍含假设 |
| X4-T1 独立链，9 月 29 日 | 模型、17 电机状态机、TCP/C SDK、7 路 Linux vcan、网页六方向和五轴参数面板 | [三日报告](docs/x4t1_report.md)；真实 robotd 未验证；换球拍前的数据不可当作现模型验收 |
| 新 Ubuntu 基线，9 月 30 日 | 隔离环境、阶跃、停车、静止、五轴响应，1/0.5 ms 对照 | 18 场景×2，36/36 测量有效；[第一阶段报告](docs/stage1_report.md) |
| 原生 C++ 优化，9 月 30 日 | 无显式休眠、SDK 发送优化、500/1000 Hz 长测、GUI/GC/FIFO 和吞吐对照 | [最新频率报告](docs/native_timing_report.md)记载 84 项回归通过；尚无 MPC 求解负载 |
| 本次文件交付 | 本总览、SHA256 清单、Ubuntu 主线及配套资料检查 | 见第 7 节，不把文件同步算作新增算法验收 |

以下数值来自已保存实验，本次整理未重复整套性能长测：

| 实验 | 结果 | 结论 |
|---|---|---|
| 普通调度、关闭循环 GC，500 Hz×60 s | 499.617 Hz；13 次超时、23 个跳过槽 | 严格每周期 2 ms 未通过 |
| 同配置，1000 Hz×60 s | 999.300 Hz；59958/60000 轮；11 次超时、42 个跳过槽；p99 0.0860 ms，最大 8.9155 ms | 仍有长尾；实际 MPC 还需计求解时间 |
| 同配置，自由吞吐×30 s | 13491.28 个完整三 RPC 循环/s，新状态 998.73/s，重复 step 92.60% | 约 1.3 万轮是通信吞吐，不是物理/MPC 频率 |
| 0.5 ms 物理、2000 Hz 客户端×10 s | 1999.80 Hz，2 次超时、2 个跳过槽 | 仅短时探索 |
| 0.2 m/s 遥控松键 | 外层以 0.3 m/s² 减速，约 0.667 s 归零；观察到反向回摆 | 需分别评估参考整形、PI 积分、接触和惯性 |
| 无重力补偿 PD 的 HOME | J2 +4.49°、J3 −5.45°、J4 +4.83° 静差 | 接口有效，精确末端控制尚未达标 |

证据：[60 秒摘要](artifacts/new_ubuntu/native/final_no_gc_60s/summary.json)、[最大吞吐](artifacts/new_ubuntu/native/final_max_30s/summary.json)、[响应基线](artifacts/new_ubuntu/stage1/summary.json)。`transport_ok`、`orchestration_ok` 或平均 Hz 达标均不能替代零超时、零跳过和新状态率验收。

## 3. 工程位置与文件职责

| 位置 | 用途 |
|---|---|
| `ubuntu@192.168.171.142`，hostname `ubuntu-11` | 当前 Ubuntu；DHCP 地址，换网后用 `ip -4 -br addr` 确认 |
| `/home/ubuntu/mpc_sim` | 当前主线：源码、模型、Linux 环境、SDK、说明和实验数据 |
| `/home/ubuntu/mpc_workspace` | 本次保存的工作区配套资料；`demo/mujoco_motor_lab` 链接到主线 |
| Mac `mpc/demo/mujoco_motor_lab` | 主线副本；`.venv-mac` 仅适用于 Mac |
| 旧 Ubuntu `/home/sckd/mujoco_motor_lab` | 原阶段工作目录，不是当前新机入口 |
| 旧 Ubuntu `/home/sckd/mpc_-pybullet-5`、`my_isaaclab_badminton` 等 | 历史 NMPC/Isaac 工程，本次未认定完整迁入新机 |

主线目录（下文未注明时均相对 `/home/ubuntu/mpc_sim`）：

```text
MPC_PROJECT_STATUS.md       本总览
setup_ubuntu.sh             创建/恢复项目 Python 环境、构建 C SDK
requirements.lock.txt      Python 锁定依赖
project_env.sh             缓存、临时目录和 Python 隔离设置
start_x4t1_demo.sh          tmux 三维服务+网页入口
launch_x4t1.sh              server / panel / teleop / validate
source/X4_5/               原始 URDF 和 18 份 STL 网格
models/                    x4_t1.xml、旧 x4_5 模型及映射
config/                    几何、电机和设备说明快照
x4t1_sim/                  引擎、TCP/CAN、面板、验收和基准
sdk/                       C 头文件/实现、C++ 基准及构建脚本
examples/、tests/          调用示例、回归测试
docs/、artifacts/          详细报告、原始数据、截图和交付记录
tools/verify_project_files.py  SHA256 清单及模型引用检查
.venv/、.cache/、.tmp/      Linux 环境、可再生缓存和临时文件
```

XML 通过相对路径引用 `source/X4_5/meshes`，不能只复制 XML。`build_model.py` 生成原 X4_5，`x4t1_sim/build_model.py` 生成 T1，当前加载 `models/x4_t1.xml`。不要在未确认几何变化时重建覆盖基线。

实际设备映射在 `engine.py` 的 `DEV`，`config/x4t1_devices.json` 只是说明快照。当前轮半径 0.0825 m，恢复原球拍拍心 J5 局部 y=0.915 m；保留源连杆惯量、附加转子惯量为零。B/d4 几何定义和改尺寸后的惯量仍待核对。

## 4. 接口契约

### 4.1 分层、坐标和单位

```text
网页/遥控/C++ 控制器 → Python client 或 C SDK → TCP
    → 独立仿真服务 → 电机 PI/PD → MuJoCo → 状态/诊断
未来真实 robotd → Linux vcan → 暂定 CAN 电机桥 → 物理引擎
```

六方向运动学和舵外环在客户端 `teleop.py`。服务只有一个物理推进者和一个写控制所有者；网页与第二个 SDK 写客户端不能同时持有写权。

| 项目 | 契约 |
|---|---|
| 单位 | m、s、rad、rad/s、N·m；网页角度为 degree，发送前转 rad |
| 车体系 | +x 前、+y 左、+z 上；俯视逆时针为 +wz |
| 17 轴顺序 | FR、FL、RL、RR，每角内轮/外轮/舵；随后 J1–J5 |
| SDK 域 | MJCF 关节坐标；12 轴速度+5 轴混合参数 |
| CAN 域 | 电机坐标；边界转换一次 `q_joint=sign*q_motor+offset`，速度/力矩乘 sign |
| 电机 | J1/J2 为 RS04，其余 RS06；J5 机构限矩 5.5 N·m |
| STATE | x/y/z 是 base_link 原点世界位置；vx/vy 是质心速度的车体系分量；wz 是世界 z 角速度 |

停车分析使用 DIAGNOSTICS 中参考点/坐标系一致的数据，不将 STATE 世界位置和车体系质心速度直接混用。

### 4.2 C/C++ SDK

[sdk/robot_sim.h](sdk/robot_sim.h)、[sdk/robot_sim.c](sdk/robot_sim.c)；Linux 库为 `sdk/build/librobot_sim.so`。这是本项目模拟器 API v1，不是厂家 ABI。一个句柄供一个线程；成功返回 0、失败 −1，open 失败返回 NULL。

| API | 输入/语义 |
|---|---|
| `robot_open(port)` / `robot_close` | 连接本机、自动 CLAIM 写权；关闭不隐式清除目标 |
| `robot_get_info` / `robot_get_state` | 配置说明、车体及 17 轴状态 |
| `robot_enable(index,on)` | index=0..16，−1 为全部；on=0/1 |
| `robot_set_velocity(target[12])` | 交错轴顺序的轮/舵目标 rad/s |
| `robot_set_arm_hybrid(target[5])` | 每轴 q、dq、kp、kd、tau_ff；当前 tau_ff 必须 0 |
| `robot_brake` | 底盘零速目标，臂按测量姿态保位；清空臂延迟队列、恢复默认增益，不自动使能 |
| `robot_set_watchdog(seconds)` | 0 为关闭、断流目标锁存；演示设 0.5 s |
| `robot_sim_reset/pause/step` | 仿真管理；step 仅手动服务，1..1000 步 |
| `robot_send_frame` / `robot_last_error` | 封装暂定 CAN 帧 / 错误说明 |

臂控制为限幅后的 `kp*(q_ref-q)+kd*(dq_ref-dq)`，命令延迟 9 ms，无额外重力前馈。失能为零输出但物理重力/接触继续；目标可锁存，模式只能在失能时改。RS04 ±36、RS06 ±66 rad/s 是协议量程，不是机构安全运动范围。

网页 `arm_hold` 使用最近提交的增益保位，与 BRAKE 恢复默认增益的行为不同。

`examples/sdk_smoke.c` 需要 `server --manual`。实时控制不调用 step，应按墙钟调度状态读取、求解、下发。

### 4.3 TCP/Python

默认 `127.0.0.1:18765`。头 16 字节 `!4sHHII`：magic=`X4T1`、version=1、opcode、sequence、payload_size；数值为小端 float64，最大载荷 32768 字节，不传 C 原始结构体。成功响应 opcode|0x8000、保留序号；错误 0xffff、UTF-8 文本。超时/断流后重连。

| opcode | 名称 | 请求 |
|---|---|---|
| 1/2/3 | INFO/STATE/CLAIM | 空 |
| 4/5/6 | ENABLE/VELOCITY/HYBRID | index,on / 12 doubles / 25 doubles |
| 7/8/9 | BRAKE/WATCHDOG/RELEASE | 空 / seconds / 空 |
| 20/21/22 | RESET/PAUSE/STEP | 空 / 0或1 / 步数 |
| 30/31/32 | FRAME/FEEDBACK/MODE | `!BIB`+数据 / 空 / index,mode |
| 33 | DIAGNOSTICS | 空，返回只读 JSON |

STATE 共 167 doubles：前 14 项 `sim_s,wall_us,steps,paused,fault,watchdog,x,y,z,yaw,vx,vy,wz,tilt`，随后 17×9 项 `q,dq,tau_actual,tau_requested,dq_target,q_target,enabled,mode,torque_cap`。`wall_us` 是单调时钟，不是 Unix 时间。

DIAGNOSTICS 原子读取目标、参考、积分、请求/实际力矩、延迟队列、臂参数和参考点明确的速度，不推进物理、不刷新心跳、不申请写权。当前 C 头文件没有专用 DIAGNOSTICS 函数，Python 用法：

```python
from x4t1_sim.client import Client
with Client(port=18765, claim=False) as client:
    state = client.state()
    diagnostics = client.diagnostics()
```

### 4.4 网页与 vcan

网页 `127.0.0.1:18767`，`GET /` 页面、`GET /state` 状态、`POST /action` 动作。动作包括 `initialize,target,cycle,brake,disable,pause,resume,reset,arm_apply,arm_hold,limits`；target 需续约，300 ms deadman；演示服务 watchdog 500 ms。Host/Origin 需匹配本机来源及端口。字段见 [panel.py](x4t1_sim/panel.py)、[调用示例](examples/panel_action.py)。

`sdk/setup_vcan.sh` 建立 vcan0..6，`server --vcan` 将写控制交给 CAN。**只读观察用 Python `Client(claim=False)`；当前 C `robot_open()` 自动 CLAIM，在 vcan 模式会被拒绝，不能充当只读 open。**

CAN 暂定 MIT 4×uint16 大端、模式 int32 大端、寄存器 float 小端、位置按 8π 回卷；温度固定 25°C。厂家故障位、vpad 33 字节、HI91/PTY 尚不完整。实际 vcan 测试使用合成主站，未证明真实 robotd 兼容。详见 [接口说明](docs/x4t1_interface.md)。

### 4.5 周期和停止语义

| 层次/操作 | 当前行为 |
|---|---|
| 物理/电机内环 | 默认 1 ms，名义 1000 Hz；0.5 ms 可对照 |
| 遥控/面板后台 | 目标 4 ms，即 250 Hz，不是未来 MPC 最终频率 |
| C++ 基准 | STATE→VELOCITY→HYBRID，独立客户端及测试服务 |
| 显示/记录 | GUI/常规 CSV 目标 50 Hz，浏览器查询/续约另有周期 |
| 松键 | 目标改零，经外层斜坡和内环减速 |
| BRAKE | 底盘零速，臂按当前姿态保位，仍可能有静差 |
| PAUSE / DISABLE / RESET | 冻结物理 / 零电机输出但物理继续 / 重置仿真和控制状态 |

## 5. Ubuntu 使用说明

已记录环境：Ubuntu 24.04.3、Python 3.12.3、MuJoCo 3.13.0、NumPy 2.5.3；Core Ultra 9 285K（24 CPU）、RTX 5090。依赖安装在项目 `.venv`。基础 Python、venv、编译器、tmux 和适用图形库由 Ubuntu 提供，不能用 Mac venv/dylib 替代。

正常启动：

```bash
cd /home/ubuntu/mpc_sim
source project_env.sh
bash start_x4t1_demo.sh
```

Ubuntu 打开 `http://127.0.0.1:18767`，初始暂停/失能，先初始化再操作。tmux 为 `x4t1-sim`；脚本保留已有会话，会话存在不自动代表服务健康。关闭三维窗口后需重启相应服务。

新安装/环境损坏时使用 `bash setup_ubuntu.sh`，C++ 基准构建用 `bash sdk/build_timing_benchmark.sh`。Mac 查看 Ubuntu 网页：

```bash
ssh -F /dev/null -N -L 28767:127.0.0.1:18767 ubuntu@192.168.171.142
```

Mac 打开 `http://127.0.0.1:28767`，三维窗口仍在 Ubuntu。转发、tmux、vcan 不承诺跨重启恢复。现有图形说明使用 DISPLAY=:1，换用户/桌面后须核对。

常用复验（均在主线目录）：

```bash
source project_env.sh
.venv/bin/python tools/verify_project_files.py \
  --manifest artifacts/deployment/project_manifest.json --runtime \
  --report artifacts/deployment/file_verification.json
.venv/bin/python -m pip check
.venv/bin/python -m unittest discover -s tests -v

# 手动推进的物理响应，不能证明墙钟实时性
.venv/bin/python -m x4t1_sim.validate_stage1 --dt .001 .0005

# 独立端口的原生 C++ 长测，不占用主网页写控制权
bash run_cpp_timing.sh --server-cpu 4 --client-cpu 1 \
  --rates 500 1000 --duration 60
```

自由吞吐用 `--rates 0 --duration 30`；正常使用无需 FIFO 调度。详见 [新机说明](docs/new_ubuntu_setup.md)、[频率报告](docs/native_timing_report.md)。

| 数据目录 | 内容 |
|---|---|
| `artifacts/x4t1_runs/` | 常规参数、state.csv、事件及退出结果 |
| `artifacts/x4t1_stage1/` | 新机响应基线原始 CSV |
| `artifacts/x4t1_timing/` | Python/ctypes 三 RPC 数据 |
| `artifacts/x4t1_native/` | Ubuntu 原生 C++ 数据，包括大型 samples.csv |
| `artifacts/new_ubuntu/` | 已回传 Mac 的摘要/日志/部分 CSV，不是全部原始数据 |
| `artifacts/day*`、`artifacts/x4t1_ubuntu/` | 旧阶段证据，须结合模型、时间和参数阅读 |
| `artifacts/deployment/` | 环境、依赖、文件清单及本次核验 |

目前本地 Git 尚未形成完整已跟踪基线，不能仅依靠 git clone/提交号复现。本次以明确文件清单固定交付内容；后续应建立源码提交及模型/参数版本管理。

## 6. 未来计划与验收

以下为实施顺序，不表示已完成或固定日期承诺，不创建定时任务。

| 优先级 | 工作/交付 | 验收与依赖 |
|---|---|---|
| P0 | 冻结世界系拍心/末端 pose 6 维、线速度 3 维、逐点到达时间 t 契约 | 明确姿态表示、拍心框架、时钟起点、过期点、单位、五轴+底盘可达性和输入示例 |
| P0 | 独立 C++ MPC/PVT 外环，复用 SDK | 状态重排/零位/坐标/轮舵分配正确；求解失败/超时可追溯；控制器与物理服务独立 |
| P0 | 运动+实际求解负载下最低 500 Hz、目标 1000 Hz | 分别长测，报告完整周期、miss/skip、新状态率、p99/最大及故障；严格通过需零 miss/skip |
| P1 | 停车与臂跟踪改进 | 纳入 9 ms 延迟、参考整形、限矩；PI 积分策略与重力补偿分别做同场景对照；tau_ff=0 契约如需修改要明确版本 |
| P1 | 时序长尾定位 | 区分客户端/服务调度、GUI、GC、日志、内核等待；再决定线程隔离、合并 RPC 或共享内存 |
| P1 | 厂家 robotd/L2 对齐 | 取得同版本 robotd、codec、can_bus/server、robot_8w4s.ini/robot_sim.ini 和参考帧；验收通道、mechPos、模式及 6 ms 超时 |
| P1 | 原遥控/IMU/日志链 | 取得 vpad_player/33 字节和 HI91/安装坐标；实现 PTY、原 Stop/失联语义及 37 列日志 |
| P2 | 实机动力学标定 | 同步输入输出及负载/地面/供电/固件信息齐全；标定阶跃、臂重力/惯量和轮地滑移 |
| P2 | 羽毛球与策略 | 迁移飞行/碰撞，检查双重反弹/穿透；再做 Gymnasium、SAC、checkpoint 和正式命中/落点评估 |

旧计划“先 250 Hz”属于早期建议，最新目标为最低 500 Hz、目标 1000 Hz。复用旧 NMPC 前必须处理核心源码缺失及求解成功码问题。

## 7. 本次 Ubuntu 文件完整性交付

2026-09-30 已在 `ubuntu-11` 实际完成以下检查：

| 检查 | 本次结果 |
|---|---|
| 主线交付清单 | 671 项全部存在且 SHA256 一致，缺失 0、变更 0 |
| 远端扫描 | 共 1078 项，包含另行保留的 407 项 Ubuntu 文件；约 3.03 GB 内容完成哈希检查 |
| 模型资源 | 152 个 XML/URDF 文件引用检查通过；原 URDF 和 18 份 STL 均在项目内 |
| 实际模型加载 | Ubuntu MuJoCo 3.13.0 加载 x4_t1.xml 成功：nq=24、nv=23、nu=17 |
| 配套工作区 | 159 份文件 SHA256 一致；主线入口链接正确 |
| Python 依赖 | requirements.lock.txt 版本逐项一致，pip check 通过 |
| Linux SDK/程序 | .so、benchmark_timing、sdk_smoke 存在，动态依赖均可解析，ctypes 加载 SDK 成功 |
| 本次回归 | Ubuntu 新运行 84 项测试，全部通过；[测试日志](artifacts/deployment/handoff_regression.log) |
| 原始实验数据保留 | 响应 36 份、时序 10 份、原生 C++ 40 份 CSV，均在 Ubuntu |

保证范围为**当前 MuJoCo 主线的源码、源 URDF/STL、模型、配置、SDK 源码、测试、文档、本地现有实验材料，以及当前工作区配套资料**。Ubuntu 原有大 CSV、Linux `.venv`、SDK `.so` 及历史基准副本继续保留。

Mac `.venv-mac`、平台构建目录、字节码、可再生缓存及主线 `.tmp` 不作为跨平台文件复制；Linux 环境及 SDK 在目标机检查。工作区顶层 `tmp/` 中的文档源资料另行保存，不同于主线运行缓存。

正式清单：[project_manifest.json](artifacts/deployment/project_manifest.json)。每项含相对路径、大小及 SHA256；Ubuntu 校验缺失或变更返回非零，允许保留额外实验文件。工具检查大小写冲突、越界链接和 XML 资源引用，`--runtime` 实际加载主模型。清单和核验输出不参与自身哈希。

本次结果见 [文件核验](artifacts/deployment/file_verification.json) 和 [Ubuntu 环境/SDK/测试核验](artifacts/deployment/ubuntu_handoff_checks.json)。配套资料在 `/home/ubuntu/mpc_workspace`，使用独立 `workspace_support_manifest.json`、`workspace_support_verification.json`；`demo/mujoco_motor_lab` 链接到 `/home/ubuntu/mpc_sim`。

同步不删除远端独有文件；有替换时，旧文件保存在 `/home/ubuntu/mpc_delivery_backups/20260930`。交付检查不重新触发主网页运动，也不重复长时间性能实验。

**尚未取得、不能宣称齐全的后续材料：** 厂家 robotd/完整 codec/vpad/HI91 定义、实机标定日志、未定位的 NMPC 核心实现，以及旧 Isaac/PyBullet 的完整环境、权重和库。当前 MuJoCo 主线不依赖这些材料；研究笔记/源码节选不是完整旧工程副本。

## 8. 旧 MPC/DDS 迁移参考

以下不属于 X4-T1 TCP v1。配套工作区 `notes/pybullet-study.md`、`isaaclab-study.md`、`integration-study.md`、`variants-study.md` 保留原始依据。

- DDS domain 0、topic `MPCSHOTCOMMAND` → `libfastdds_bridge.so` → Python receiver/MpcController → `libmpc_c_api.so` → `libnmpc.so` → Isaac。
- MPC 状态 14 维 `[x,y,q1..q5,vx,vy,dq1..dq5]`；默认 3×7 输出为底盘 XY 与五臂的速度增量，需累加。旧臂符号 `[-1,-1,-1,+1,-1]` 不直接套到新模型；`times=[1]` 会复制至每段，不是总时长 1 s。
- DDS 22 个数值：row_id/time_s、target xyz/dt_remain、4 个 hit 参数、hit_stage、robot_target xyz、robot_curr xyz、五关节位置。FIFO 4096，满时丢最旧；CSV 固定 period_ms 发布，不按 time_s 自动调度。
- 旧优化失败未可靠透传为非零返回码；Python 无异常不代表求解成功。已定位目录缺 nmpc.cpp，不能认定能完整重建求解器。
- 旧库含 OpenCV .406/.413 混合依赖及 CasADi/Boost/FastDDS/FastCDR，未作为当前主线依赖安装。
- 旧 Isaac 直接写 root velocity 移动车体，不能替代轮地动力学验收；已播放策略为 PPO，SAC 仍待实施。

后续每次更新进度，应同时更新验收证据及文件清单。
