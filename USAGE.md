# 操作手册(速查)

> 首次部署请先按 [环境安装与部署](docs/环境安装与部署.md) 执行一键安装和验收。

仿真仓库日常操作指令。研究记录见 `PROGRESS.md`,本文件只放"怎么跑"。
所有命令默认在 `sckd_sim/` 目录下执行;Python 使用工程根目录
`.venv/bin/python`(在本目录下为`../.venv/bin/python`)。CMake 默认从工程根目录
`deps/mujoco`与`deps/casadi`加载头文件和库,也可用`-DMUJOCO_DIR=`、
`-DCASADI_DIR=`覆盖。

## 1. 构建与测试

```bash
cd sckd_sim                    # 从工程根目录执行
cmake --build build -j8          # 增量构建
ctest --test-dir build           # 全量单测(10 项)
ctest --test-dir build -R legacy_nmpc   # 只跑旧机 NMPC 回归
```

## 2. 三策略切换与 A/B(m2_intercept)

```
./build/m2_intercept <场景> <发数> <挥拍层> <起挥延迟> <种子> <策略> [发球族] [xb/v2]
```

| 参数 | 取值 | 说明 |
|---|---|---|
| 场景 | `models/generated/x4_tennis_demo.mjcf` | 默认场景 |
| 挥拍层 | `qp` | QP 挥拍(A/B 口径都用它;`base` 为五次多项式老基线) |
| 起挥延迟 | `0.25` | 定值,勿改(扫描平台中心) |
| 种子 | 任意整数 | **同种子 = 各策略面对逐位相同的发球** |
| 策略 | `pfly` / `wbmpc` / `nmpc6` | B′ 基线 / 自研全身MPC / 旧仓库实机 NMPC;留空=定点 P 控 |
| 发球族 | `fixed`(默认)/ `rand` / `gs` | 定点(测精度)/ 随机(测泛化)/ 抽球(测最难) |
| 第 9 参 | `xb` | 跨反弹规划,**gs 族必带**(否则可达 0/10) |

```bash
# 三策略同口径对比(同种子 2026)
./build/m2_intercept models/generated/x4_tennis_demo.mjcf 10 qp 0.25 2026 pfly
./build/m2_intercept models/generated/x4_tennis_demo.mjcf 10 qp 0.25 2026 wbmpc
./build/m2_intercept models/generated/x4_tennis_demo.mjcf 10 qp 0.25 2026 nmpc6
# 随机族 / 抽球族
./build/m2_intercept models/generated/x4_tennis_demo.mjcf 10 qp 0.25 7 nmpc6 rand
./build/m2_intercept models/generated/x4_tennis_demo.mjcf 10 qp 0.25 7 nmpc6 gs xb
```

**输出判读**:末尾汇总行 `命中 N/M` = 方向翻转+拍球距离<0.30m+落地前;
`回退 K trial` = 该策略中途降级到 B′ 的发数(看策略纯度);`出球速度`
(意图 8 m/s)、`触球底盘速度`(fly-through 判据 >0.5)。退出码 0/1 =
过/不过门槛(fixed≥80%,rand/gs 可达域内≥60%)。

**当前基准**(详见 PROGRESS「旧仓库 NMPC 独立模块化」节。2026-10-09 起
待机位姿=实机 home(J2 抬臂 35°),三策略起步口径统一):

| 族 | nmpc6(无补偿) | B′(pfly) | wbmpc |
|---|---|---|---|
| fixed(待机起步,10 发) | 70% | 100% | 60%(待机远起点适配待办) |
| rand 三种子(旧口径*) | 30/30/40 | 70/70/70 | 70/70/70 |
| gs+xb 三种子(旧口径*) | 60/40/0 | 60/60/70 | 60/60/70 |

*旧口径=臂零目标保持起步(待机修正前的历史数据,趋势参考;重跑即得新口径)。

## 3. GUI 可视化(看仿真效果)

从工程根目录可直接启动 viewer 和默认3球`pfly`演示;也可传入策略和球数:

```bash
tools/run_mujoco_demo.sh
tools/run_mujoco_demo.sh nmpc6 3
```

需要分开控制两个进程时,在`sckd_sim/`目录执行:

```bash
# 终端①:启动 viewer(DISPLAY 按实际 X server;无显示环境跳过本步)
DISPLAY=:1 ../.venv/bin/python python/tools/demo_viewer.py
# 终端②:任一策略加 M2_VIEW=1 即可视化(实时节拍 + 60Hz 状态同步)
M2_VIEW=1 ./build/m2_intercept models/generated/x4_tennis_demo.mjcf 3 qp 0.25 2026 nmpc6
M2_VIEW=1 ./build/m2_intercept models/generated/x4_tennis_demo.mjcf 3 qp 0.25 2026 wbmpc
M2_VIEW=1 ./build/m2_intercept models/generated/x4_tennis_demo.mjcf 3 qp 0.25 2026 pfly
# 随机族 / 抽球族同样可可视化 —— 三族 GUI 接线同一主循环,已验证。
# 注意 gs 族必须带 xb(跨反弹规划),漏了 = 可达 0/10,机器人全程待机不挥拍
# (易误读为"GUI 没接线";2026-10-10 实测两族状态文件与统计均正常)
M2_VIEW=1 ./build/m2_intercept models/generated/x4_tennis_demo.mjcf 3 qp 0.25 2026 nmpc6 rand
M2_VIEW=1 ./build/m2_intercept models/generated/x4_tennis_demo.mjcf 3 qp 0.25 2026 nmpc6 gs xb
```

**录制视频**:viewer 加 `REC` 环境变量,离屏渲染逐帧经系统 ffmpeg 直编 H.264
(640×480@60,固定全场机位;待机段不产帧;关 viewer 或 Ctrl+C 时落盘):

```bash
REC=1 DISPLAY=:1 ../.venv/bin/python python/tools/demo_viewer.py  # → /tmp/sckd_demo.mp4
REC=/path/to/demo.mp4 DISPLAY=:1 ../.venv/bin/python python/tools/demo_viewer.py
```

协议:m2_intercept/m2_demo 60Hz 写 `/tmp/sckd_demo_state.bin`
([seq:u64][nq:i32][nv:i32][qpos][qvel]),viewer 只读渲染,两个 app 共用。

**规则与症状排查**(2026-10-09 修复):

| 症状 | 原因 | 现行为/处理 |
|---|---|---|
| viewer 一启动就"在挥拍" | 上次运行的残留状态文件被回放 | viewer 只应用 1s 内的新鲜帧,写者退出即冻结 |
| 画面像两套指令打架 | 两个仿真实例同时写同一状态文件 | pid 锁互斥(`.lock`):第二个实例自动禁用视图并打印提示;**同时只开一个仿真实例** |
| 画面 N 倍速快放 | 批处理比实时快 | M2_VIEW 自带实时节拍,无需处理 |
| wbmpc 模式偏卡/慢动作 | wbmpc 单解可达 ~100ms,计算密度高 | 播放慢于实时属正常;演示优先 pfly/nmpc6 |
| 击球瞬间看不到就跳下一发 | 批处理触球后 30ms 即终止 | M2_VIEW 下触球后观察 0.8s + trial 间驻留 0.7s |
| 实例崩溃后锁残留 | 进程被 kill | 锁含 pid,新实例检测到进程已死自动抢占 |
| 无显示/黑屏 | X server 未起 | `echo $DISPLAY` 确认;无头环境跑批量(不加 M2_VIEW)即可 |

## 4. 调试旋钮(环境变量)

| 变量 | 作用 |
|---|---|
| `M2_VIEW=1` | GUI 同步+实时节拍(m2_intercept) |
| `NMPC6_DEBUG=1` | nmpc6 驱动逐解打印(ok/loose/acc-fail/jump + 误差) |
| `N6_TRACE=1` | nmpc6 末段跟踪 trace(实测 vs 计划@每解拍) |
| `N6_SPD=x` | nmpc6 拍速目标倍率(默认 0.8=标定值;扫参用) |
| `WB_CHASSIS=1` | wbmpc 底盘跟踪打印 |

```bash
NMPC6_DEBUG=1 ./build/m2_intercept ... nmpc6        # 看拒收原因分布
N6_TRACE=1 ./build/m2_intercept ... nmpc6 | grep n6+ # 看末段终端退化
```

## 5. nmpc6 生成码重生成(改公式/常数时才需要)

改 `third_party/nmpc6/gen_nlp.cc`(OCP 公式)或 `sim_fk.h`(几何)后,
**两文件必须重生成**(README 有完整说明):

```bash
cd sckd_sim/
g++ -O2 -std=c++17 -I ../deps/casadi/include -I third_party/nmpc6 \
    -L ../deps/casadi/lib -Wl,-rpath,../deps/casadi/lib \
    third_party/nmpc6/gen_nlp.cc -o /tmp/gen_nlp -lcasadi && /tmp/gen_nlp
gcc -fPIC -shared -Ofast nlp.c -o third_party/nmpc6/nlp.so -lm
mv nlp.c third_party/nmpc6/nlp.c
ctest --test-dir build -R legacy_nmpc    # 回归必须绿
```

## 6. 其他常用入口

```bash
./build/m2_demo 3            # 单发演示(GUI viewer 配套,写同一状态文件)
./build/demo_hit             # 反弹标定演示
./build/m2_predictor         # 预测器验收
ctest --test-dir build -R wbmpc   # 全身 MPC 单测
```

## 7. 旧仓库(nmpc6 来源)

原 `badminton-mpc-6.0` 仓库不随本项目分发，仅作为历史参考。移植版在
`sckd_sim/cpp/control/legacy_nmpc.{h,cpp}` + `third_party/nmpc6/`,
差异与考古记录见 `sckd_sim/third_party/nmpc6/README.md`。

## 8. MIT 单机协议仿真(独立,2026-10-18)

真机臂执行链的单关节独立仿真(纯标准库,不依赖 MuJoCo/sckd_core),
四变体同参并行对比跟踪曲线:A 纯MIT(真机现状,rob J2 kp50/kd30)/
B+速度环积分(供应商固件假设)/C+理想τ前馈(T位固件假设)/D 兼有。

```bash
cmake --build build --target mit_motor_sim
./build/mit_motor_sim all eval/reports    # step/chirp/swing 三场景 → CSV
python3 python/tools/plot_mit_motor.py    # 三张四行图(位置/速度/误差/力矩)+指标表
# 参数覆盖(B 实验扰动档直接复用):./build/mit_motor_sim all eval/reports J=0.24 G=9.6
# 前馈辨识误差注入(独立输出目录,勿覆盖标称):
mkdir -p eval/reports/fferr_x && ./build/mit_motor_sim all eval/reports/fferr_x ff_J=0.16 ff_G=9.6
python3 python/tools/mit_ff_err.py eval/reports/mit_ff_err.png eval/reports \
  "J−20%=eval/reports/fferr_J16" "G+20%=eval/reports/fferr_G96"  # 跨配置对比(C/D+速度误差行)
```

常量来源:量化帧=旧仓 robot_control.h RS04(±12.57/±36/kp[0,5000]/
kd[0,100],τ 槽 ±120 为 T 位假设);T-N 包络=motors.json;增益=rob
robot_sim.ini [Arm] J2 track。ki/ti_max/τ 槽为假设档,待供应商固件
确认(交接文档 §1.3「臂执行帧」行)。

## 9. 5 关节臂前馈参数辨识——仿真先行验证(2026-10-20)

`mit_arm_id_sim`(MuJoCo 5 关节耦合真值,直链 mujoco 不链 sckd_core)+
`mit_arm_id.py`(系统 python 辨识管线):把真机标定规程(poses 重力/
constvel 摩擦/chirp 惯量/swing 闭环验收)在仿真里跑通并验证无偏。
B 积分变体已删——供应商将提供标准 MIT 固件(含 τ 前馈槽、无积分项)。
**真机操作规程见 `docs/真机臂前馈辨识操作手册.md`**(含现场校核项/验收三道关)。

```bash
cmake --build build --target mit_arm_id_sim
D=eval/reports/mitarm && mkdir -p $D
./build/mit_arm_id_sim poses    $D poses_n=64   # ~10s:64 姿位双向趋近
./build/mit_arm_id_sim constvel $D              # 恒速双向斜坡(0.3/0.6/1.0 rad/s)
./build/mit_arm_id_sim chirp    $D              # 40s 多正弦(幅度按 T-N 调度)
./build/mit_arm_id_sim swing    $D              # 先跑一遍出参考列(无 C_id)
python3 python/tools/mit_arm_id.py $D --stage all --pose-counts 8,16,32,64 --misalign
./build/mit_arm_id_sim swing    $D ff_file=$D/ff_swing.csv   # C_id 闭环重跑
python3 python/tools/mit_arm_id.py $D --stage report
# 单阶段重跑(参数持久化在 $D/mitarm_id_params.npz):
#   --stage fk|grav|fric|inert|ff|report|plots;科氏项关节选择:MITARM_CORIOLIS=1
# 常用覆盖:poses_n= chirp_T= vnoise=0.002 seed=1(带噪方差复验)、
#   b1..b5= tc1..t5= arm1..5=(摩擦/转子反射惯量真值)、hold_kp1..= mv_kp1..=
python3 python/tools/mit_arm_id.py $D --stage plots   # 3 张图
```

注意:①管道会吞 python 偶发段错误(本机 numpy 环境性 flake)的退出码,
  `--stage ff` 后确认 `ff_swing.csv` 行数非 0 再重跑 swing;②标称数据目录
  `eval/reports/mitarm` 勿被扰动档覆盖(同单关节 fferr 惯例)。
CSV 契约(真机日志同构,`mit_arm_id.py` 可直接吃真机数据):命令回显
p/v/kp/kd/tff_cmd + 遥测 q/dq(250Hz ZOH)+ 真值诊断 tau/grav/Mii(真机无,
缺失时自动跳过验证段);manifest 首行组 `# k=v`。
