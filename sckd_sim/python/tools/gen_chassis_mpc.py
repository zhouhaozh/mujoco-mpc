#!/usr/bin/env python3
"""底盘 MPC QP 离线构建 + C 代码生成(CasADi qpsol + OSQP 后端)。

用法: /home/ubuntu/miniconda3/envs/sim/bin/python python/tools/gen_chassis_mpc.py

产物: cpp/control/gen/chassis_mpc.c / .h   (自包含,运行时零 casadi/python 依赖)
设计: docs/SIM_SYSTEM_DESIGN.md §6.2 方案A(底盘块)——
  状态 x=[px,py,ψ,vx,vy,wz],输入 u=[ax,ay,α];
  动力学:重解时刻航向 (c0,s0) 线性化的双积分器(参数化);
  约束:摩擦圆(8 边内接多边形,保守侧)‖(ax,ay)‖、‖(vx,vy)‖ ≤ 上限;
        |α|、|wz|;航向窗 |ψ_k−ψ_ref,k| ≤ θ_max(QR 可见性,§6.2)。
参数(运行时传入):x0(6), 参考轨迹 (N+1)×6, (c0,s0), lim[5]=[a_max,v_max,α_max,wz_max,θ_max]。
"""
from pathlib import Path

import casadi as ca
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "cpp" / "control" / "gen"
OUT.mkdir(parents=True, exist_ok=True)

NX, NU = 6, 3          # x=[px,py,ψ,vx,vy,wz] u=[ax,ay,α]
N = 100                # 视界步数(dt=20ms → 2s)
NPOLY = 8              # 摩擦圆内接多边形边数
DT = 0.02

# ---------- 决策变量与参数 ----------
x = ca.SX.sym("x", NX * (N + 1))      # 堆叠状态
u = ca.SX.sym("u", NU * N)            # 堆叠输入
z = ca.vertcat(x, u)
x0 = ca.SX.sym("x0", NX)
ref = ca.SX.sym("ref", NX * (N + 1))  # 参考轨迹(堆叠)
cs = ca.SX.sym("cs", 2 * N)           # 每步 (cosψk, sinψk):按参考航向逐步线性化
lim = ca.SX.sym("lim", 5)             # [a_max, v_max, alpha_max, wz_max, theta_max]
p = ca.vertcat(x0, ref, cs, lim)

def X(k):
    return x[k * NX:(k + 1) * NX]

def U(k):
    return u[k * NU:(k + 1) * NU]

def R(k):
    return ref[k * NX:(k + 1) * NX]

# ---------- 动力学(等式约束)+ 初始条件 ----------
g = []
for k in range(N):
    c0, s0 = cs[2 * k], cs[2 * k + 1]  # 逐步线性化(参考航向)
    xk, uk = X(k), U(k)
    nxt = ca.vertcat(
        xk[0] + DT * (xk[3] * c0 - xk[4] * s0),
        xk[1] + DT * (xk[3] * s0 + xk[4] * c0),
        xk[2] + DT * xk[5],
        xk[3] + DT * uk[0],
        xk[4] + DT * uk[1],
        xk[5] + DT * uk[2],
    )
    g.append(X(k + 1) - nxt)
g.append(X(0) - x0)

# ---------- 代价 ----------
J = 0
W_POS, W_PSI, W_V, W_W = 10.0, 2.0, 1.0, 0.5
W_U, W_A, W_TER = 0.005, 0.005, 30.0
for k in range(N + 1):
    e = X(k) - R(k)
    ter = W_TER if k == N else 1.0
    J += ter * (W_POS * (e[0] ** 2 + e[1] ** 2) + W_PSI * e[2] ** 2
                + W_V * (e[3] ** 2 + e[4] ** 2) + W_W * e[5] ** 2)
    if k < N:
        J += W_U * (U(k)[0] ** 2 + U(k)[1] ** 2) + W_A * U(k)[2] ** 2

# ---------- 不等式约束(多边形摩擦圆 + 角限 + 航向窗) ----------
thetas = [(2 * np.pi * i) / NPOLY for i in range(NPOLY)]
poly_scale = float(np.cos(np.pi / NPOLY))     # 内接(保守侧)
for k in range(N):                            # 输入:‖(ax,ay)‖ ≤ a_max;|α|
    for t in thetas:
        g.append((np.cos(t) * U(k)[0] + np.sin(t) * U(k)[1]) - lim[0] * poly_scale)
    g.append(U(k)[2] - lim[2])
for k in range(N + 1):                        # 状态:‖(vx,vy)‖ ≤ v_max;|wz|;航向窗
    for t in thetas:
        g.append((np.cos(t) * X(k)[3] + np.sin(t) * X(k)[4]) - lim[1] * poly_scale)
    g.append(X(k)[5] - lim[3])
    g.append(X(k)[2] - R(k)[2] - lim[4])

g = ca.vertcat(*g)

# ---------- QP 求解器(qpsol + OSQP)与代码生成 ----------
qp = {"x": z, "p": p, "f": J, "g": g}
solver = ca.qpsol("chassis_mpc", "osqp", qp,
                  {"error_on_fail": False, "osqp": {"verbose": False}})
solver.generate("chassis_mpc")
for f in ("chassis_mpc.c", "chassis_mpc.h"):
    src = Path(f)
    if src.exists():
        (OUT / f).write_text(src.read_text())
        src.unlink()

nz = NX * (N + 1) + NU * N
# 同步工作区尺寸到封装头(防止重生成后 kW 过期 -> 堆越界)
import re
src = (OUT / "chassis_mpc.c").read_text()
w = int(re.search(r"\*sz_w = (\d+)\*sizeof\(casadi_real\)", src).group(1))
hdr = ROOT / "cpp" / "control" / "chassis_mpc.h"
h = hdr.read_text()
h = re.sub(r"static constexpr int kW = \d+;", f"static constexpr int kW = {w};", h)
hdr.write_text(h)
print(f"[OK] nz={nz} ng={g.shape[0]} npar={p.shape[0]} kW={w} -> {OUT}/chassis_mpc.c(.h)")
print(f"     约束构成: 动力学 {NX * (N + 1)} + 输入 {N * (NPOLY + 1)} + 状态 {(N + 1) * (NPOLY + 2)}")
