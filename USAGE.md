# 操作手册(速查)

仿真仓库日常操作指令。研究记录见 `PROGRESS.md`,本文件只放"怎么跑"。
所有命令默认在 `sckd_sim/` 目录下执行;python 为 conda env `sim`
(`/home/ubuntu/miniconda3/envs/sim/bin/python`)。

## 1. 构建与测试

```bash
cd /home/ubuntu/sckd/sim/sckd_sim
cmake --build build -j8          # 增量构建
ctest --test-dir build           # 全量单测(6 项)
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

**当前基准**(详见 PROGRESS「旧仓库 NMPC 独立模块化」节):

| 族 | nmpc6(无补偿) | B′(pfly) | wbmpc |
|---|---|---|---|
| fixed 20 发 | 65% | 100%×3 | ~100% |
| rand 三种子 | 30/30/40 | 70/70/70 | 70/70/70 |
| gs+xb 三种子 | 60/40/0 | 60/60/70 | 60/60/70 |

## 3. GUI 可视化(看仿真效果)

```bash
# 终端①:启动 viewer(DISPLAY 按实际 X server;无显示环境跳过本步)
cd /home/ubuntu/sckd/sim/sckd_sim && DISPLAY=:1 python python/tools/demo_viewer.py
# 终端②:任一策略加 M2_VIEW=1 即可视化(实时节拍 + 60Hz 状态同步)
M2_VIEW=1 ./build/m2_intercept models/generated/x4_tennis_demo.mjcf 3 qp 0.25 2026 nmpc6
M2_VIEW=1 ./build/m2_intercept models/generated/x4_tennis_demo.mjcf 3 qp 0.25 2026 wbmpc
M2_VIEW=1 ./build/m2_intercept models/generated/x4_tennis_demo.mjcf 3 qp 0.25 2026 pfly
```

协议:m2_intercept/m2_demo 60Hz 写 `/tmp/sckd_demo_state.bin`
([seq:u64][nq:i32][nv:i32][qpos][qvel]),viewer 只读渲染,两个 app 共用。

**规则与症状排查**(2026-10-09 修复):

| 症状 | 原因 | 现行为/处理 |
|---|---|---|
| viewer 一启动就"在挥拍" | 上次运行的残留状态文件被回放 | viewer 只应用 1s 内的新鲜帧,写者退出即冻结 |
| 画面像两套指令打架 | 两个仿真实例同时写同一状态文件 | pid 锁互斥(`.lock`):第二个实例自动禁用视图并打印提示;**同时只开一个仿真实例** |
| 画面 N 倍速快放 | 批处理比实时快 | M2_VIEW 自带实时节拍,无需处理 |
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
g++ -O2 -std=c++17 -I /home/ubuntu/deps/casadi/include -I third_party/nmpc6 \
    -L /home/ubuntu/deps/casadi/lib -Wl,-rpath,/home/ubuntu/deps/casadi/lib \
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

`/home/ubuntu/sckd/badminton-mpc-6.0`(只读参考)。移植版在
`sckd_sim/cpp/control/legacy_nmpc.{h,cpp}` + `third_party/nmpc6/`,
差异与考古记录见 `sckd_sim/third_party/nmpc6/README.md`。
