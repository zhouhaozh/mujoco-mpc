# nmpc6 —— 旧仓库实机 NMPC 生成码(badminton-mpc-6.0)

## 是什么

- `nlp.c` / `nlp.so`:CasADi `generate_dependencies("nlp.c")` 产出的 OCP 数值
  函数库(f/g/梯度/雅可比/黑塞全静态编入,ldd 仅 libm/libc,IPOPT 在加载
  侧)。决策 21(3 步 × [dvx,dvy,dω1..5],块序)、参数 27、约束 87:
  [刹车位置 3×7 | 位置 3×7 | 底盘速 3×2 | 关节速 3×5 | 加速度 3×7 |
  自碰 z 3×1],加速度上限 [8,8,30,30,180,400,200](m/s², rad/s²)。
- 与 `cpp/control/legacy_nmpc.{h,cpp}`(驱动层)强绑定:布局定义在生成码
  里,驱动层只做装配与守门。**改公式必须重生成本目录两文件**。

## 当前 nlp.c/nlp.so 的来源(2026-10-09)

由本目录 `gen_nlp.cc`(逐式复刻旧仓库 HEAD `set_my_nmpc_solver()` 的
符号构型)在本机 CasADi 3.7 运行库上重新生成——与加载侧 ABI 严格同版。
背景:旧仓库根目录残留的 nlp.so(Sep 8)为 26 参数/45 约束的陈旧版本,
与 git 任何提交的 nmpc.cc 都不匹配(数值探针反解证实,ACC=[5,5,30,30,
200,200,80]);2026-10-09 与工程师确认**源码即真源**,nlp.so 按其指令
`gcc -fPIC -shared -Ofast nlp.c -o nlp.so -lcasadi -lipopt -lm -ldl`
编译(生成码自含 runtime,-lcasadi/-lipopt 实际无未解析符号,仅 -lm
必需)。陈旧版存档为 `nlp_sep8.{c,so}` 留证。

## 重新生成(改公式时)

```bash
cd sckd_sim/
g++ -O2 -std=c++17 -I /home/ubuntu/deps/casadi/include \
    -L /home/ubuntu/deps/casadi/lib -Wl,-rpath,/home/ubuntu/deps/casadi/lib \
    third_party/nmpc6/gen_nlp.cc -o /tmp/gen_nlp -lcasadi && /tmp/gen_nlp
# 在 sckd_sim/ 下产出 ./nlp.c(生成文件名不得含目录成分,CodeGenerator
# 以其为 C 入口名)
gcc -fPIC -shared -Ofast nlp.c -o third_party/nmpc6/nlp.so -lm
mv nlp.c third_party/nmpc6/nlp.c
ctest --test-dir build -R legacy_nmpc   # 实机日志参数回归
```

调参入口都在 `gen_nlp.cc` 里改(ACC 表/代价权重/约束块),改完重生成;
驱动层 `legacy_nmpc.cpp` 的 kAccInv/约束界装配须同步。

## 许可

CasADi runtime 部分 MIT-0,用户代码部分归用户(灵足/旧仓库作者)。
