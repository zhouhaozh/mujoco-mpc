# nmpc6 —— 旧仓库实机 NMPC 生成码(badminton-mpc-6.0)+ sim 精确 FK

## 是什么

- `nlp.c` / `nlp.so`:CasADi `generate_dependencies("nlp.c")` 产出的 OCP 数值
  函数库(f/g/梯度/雅可比/黑塞全静态编入,ldd 仅 libm/libc,IPOPT 在加载
  侧)。决策 21(3 步 × [dvx,dvy,dω1..5],块序)、参数 27、约束 87:
  [刹车位置 3×7 | 位置 3×7 | 底盘速 3×2 | 关节速 3×5 | 加速度 3×7 |
  自碰 z 3×1],加速度上限 [8,8,30,30,180,400,200](m/s², rad/s²)。
- **FK 为 sim(x4_tennis)精确版**(2026-10-09,`sim_fk.h`,与
  `cpp/control/legacy_nmpc.cpp` 数值侧同源):闭式 5 旋转链,常数取自
  MJCF;速度=符号 jacobian 精确 J·ω+plat·(vx,vy,0)。对 MuJoCo 300 随机
  位形 max|err|≈4e-4(常数打印截断级),生成码 z 行验证 5.9e-7。
- 与驱动层强绑定:布局定义在生成码里。**改公式/常数必须重生成两文件**。

## 旧仓库 FK 差异留档(为何必须换)

旧 HEAD 的解析 FK 仍为羽毛球机参数:d1=0.685/a2=0.705 与本机一致(肩高
0.687/大臂 0.705),但拍长 **d5=0.85 vs 本机腕滚半径 0.584**(网球拍短),
腕部 d4=−0.1705 vs 本机分布式偏移(0.188 竖段+横向 0.029/−0.0265);
旧"网球化"提交(c0d98ae)只改了 ACC/PD,FK 未动。对拍残差 0.1-0.3m 级
(数值验证 2.33,超拍面半径)。真机部署若臂几何同 x4_tennis,同样适用
本 FK;若为羽毛球原机,换回旧解析式并重生成。

## 当前 nlp.c/nlp.so 的来源(2026-10-09)

由 `gen_nlp.cc` 在本机 CasADi 3.7 运行库生成(与加载侧 ABI 同版)。首版
曾直接拷贝旧仓库 Sep 8 残留 nlp.so,数值探针反解证实其为 26 参数/45 约束
陈旧版(ACC=[5,5,30,30,200,200,80]),不匹配 git 任何提交;工程师确认
**源码即真源**、nlp.so=gcc 编译 nlp.c 所得,遂按 HEAD 重生成(存档
`nlp_sep8.{c,so}` 留证)。

## 重新生成(改公式时)

```bash
cd sckd_sim/
g++ -O2 -std=c++17 -I /home/ubuntu/deps/casadi/include -I third_party/nmpc6 \
    -L /home/ubuntu/deps/casadi/lib -Wl,-rpath,/home/ubuntu/deps/casadi/lib \
    third_party/nmpc6/gen_nlp.cc -o /tmp/gen_nlp -lcasadi && /tmp/gen_nlp
# 在 sckd_sim/ 下产出 ./nlp.c(生成文件名不得含目录成分,CodeGenerator
# 以其为 C 入口名,斜杠非法)
gcc -fPIC -shared -Ofast nlp.c -o third_party/nmpc6/nlp.so -lm
mv nlp.c third_party/nmpc6/nlp.c
ctest --test-dir build -R legacy_nmpc   # 回归(实机日志参数用例,sim FK 下
                                       # 严格档 e≈(0,0.014,0.011))
```

调参入口:ACC 表/代价权重/约束块在 `gen_nlp.cc`;几何常数在 `sim_fk.h`
(生成与驱动共用,改一处两边同步);驱动侧 kAccInv/约束界装配在
`legacy_nmpc.cpp` 须同步。

## 许可

CasADi runtime 部分 MIT-0,用户代码部分归用户(灵足/旧仓库作者)。

## 2026-10-15：可选执行器状态预测库

`gen_nlp.cc`不传参数时仍生成原27参数/87约束模型；传入项目辨识JSON时生成独立37参数/87约束 `nlp_servo.c`及`actuator_rollout.c`。新增10个参数是已生效的五轴位置/速度命令；局部二阶关节状态与9ms传输参与末端预测。

在团队根目录运行：

```bash
.venv/bin/python tools/build_servo_nmpc.py --team .
# 可用 --output <独立目录> 做重建检查，不覆盖当前运行库
```

辨识参数保存在 `models/config/servo_identified_20261015.json`，来自当前MuJoCo模型；不是实机标定。原 `nlp.so`保留。该候选不含完整底盘非线性；本批几何拦截未优于辨识增益候选，详见主工程 `docs/tennis_pfly_nmpc6_control_optimization.md`。
