#!/usr/bin/env python3
"""臂轨迹优化codegen工具(§6.4,取代五次多项式基线)——两阶段方案。

阶段1(C++ 运行时,已有模块):ArmIk(DLS)解终值构型 q_hit(拍心=p*)+
  AlignFaceJ5(拍面法向≈n*);dq_hit = J⁺ v*(最小范数,拍速=反弹逆解)。
阶段2(本工具 codegen 的 QP):Chebyshev 样条系数 C(5×M),
  min ∫q̈²+∫q⃛²  s.t. 边界(实测 q0,dq0)、终值(q_hit,dq_hit@内点 s_hit)、
  末端静止、速度限位(72V)、系数盒 —— 全线性约束+二次目标(底盘 MPC 同构)。

放弃整体 NLP 的调试记录(2026-10-06,留档):
- sumsqr(FK残差)进目标:大残差二阶项不定(eig_min −3.5e6),OSQP 拒收
- 残差改等式+松弛:拉格朗日 Hessian 含约束曲率×乘子,仍不定
- L-BFGS:单体基条件数爆炸(类 Hilbert)发散;换 Chebyshev 后 QP 内层
  OSQP 迭代不达容忍(error_on_fail 吞掉)步长失控 ||q||~1e5
- CasADi 该版本 sqpmethod/osqp 选项面受限,不可调 polish/eps/max_iter
→ 两阶段把非线性全部收敛到已验证的 IK 层,轨迹层为纯 QP。

产物:cpp/control/gen/arm_nlp.{c,json};验证:两族代表工况离线解。
用法(conda sim): python python/tools/gen_arm_nlp.py [--nocodegen]
"""
import argparse
import json
import sys
import time
from pathlib import Path

import casadi as ca
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
OUT_C = ROOT / "cpp" / "control" / "gen" / "arm_nlp.c"
OUT_JSON = ROOT / "cpp" / "control" / "gen" / "arm_nlp.json"

META = {
    "M": 9,                 # 每关节 Chebyshev 系数数(s∈[0,1])
    "S": 24,                # 平滑积分/速度限位采样数
    "vlim": [18.0, 18.0, 25.0, 25.0, 30.0],   # rad/s(J1/J2=RS04-36 半;J3-5=RS06-75 保守)
    "cbox": [3.0, 3.0, 3.0, 3.0, 6.0],        # 系数盒(近似关节盒)
    "w2": 1.0, "w3": 0.05,                     # ∫q̈² / ∫q⃛² 权(s 域)
    "reg": 1e-4,                               # H 正则
    "vscale": 0.65,                            # 速度限位 s 域换算(标称 T;vlim·T)
    "s_hit": 0.72,                             # 终值内点(剩余为随挥收势)
}


def basis(s, der=0):
    """Chebyshev 基向量 (T_0..T_{M-1})(s) 及其 s 域 1/2/3 阶导(递推;×2^der 链式)。

    C++ 侧 cpp/control/arm_nlp.cpp 用同一递推,保持一致。
    """
    M = META["M"]
    u = 2 * s - 1
    Tk = np.zeros(M); d1 = np.zeros(M); d2 = np.zeros(M); d3 = np.zeros(M)
    if M > 0: Tk[0] = 1.0
    if M > 1: Tk[1] = u; d1[1] = 1.0
    for k in range(2, M):
        Tk[k] = 2 * u * Tk[k - 1] - Tk[k - 2]
        d1[k] = 2 * Tk[k - 1] + 2 * u * d1[k - 1] - d1[k - 2]
        d2[k] = 4 * d1[k - 1] + 2 * u * d2[k - 1] - d2[k - 2]
        d3[k] = 6 * d2[k - 1] + 2 * u * d3[k - 1] - d3[k - 2]
    return (Tk, d1, d2, d3)[der] * (2 ** der)


def build_qp():
    """x=C 展平(列主:关节 j 占 [j*M,(j+1)*M));p=[q0(5),dq0(5),q_hit(5),dq_hit(5),T](21)。

    时间契约:QP 在归一化 s∈[0,1] 上表达,所有导数约束用 s 域值
    (C++ 侧把物理 dq 乘 T 后传入;平滑权重亦按 s 域标定)→ 约束/目标对 p 线性。
    """
    M, S = META["M"], META["S"]
    n = 5 * M
    H = META["reg"] * np.eye(n)
    for k in range(1, S + 1):
        s = (k - 0.5) / S
        d2, d3 = basis(s, 2), basis(s, 3)
        for j in range(5):
            sl = slice(j * M, (j + 1) * M)
            H[sl, sl] += (META["w2"] * np.outer(d2, d2)
                          + META["w3"] * np.outer(d3, d3)) / S

    def eq_row(bs):
        r = np.zeros((5, n))
        for j in range(5):
            r[j, j * M:(j + 1) * M] = bs
        return r

    I5 = np.eye(5)
    def pblk(col0):
        B = np.zeros((5, 21))
        B[:, col0:col0 + 5] = I5
        return B

    s_hit = META["s_hit"]
    A_eq = np.vstack([
        eq_row(basis(0.0, 0)), eq_row(basis(0.0, 1)),
        eq_row(basis(s_hit, 0)), eq_row(basis(s_hit, 1)),
        eq_row(basis(1.0, 1)),
    ])
    B_eq = np.vstack([pblk(0), pblk(5), pblk(10), pblk(15), np.zeros((5, 21))])

    rows_in, b_in = [], []
    for k in range(1, S + 1):
        bs = basis(k / S, 1)
        for j in range(5):
            r = np.zeros((1, n)); r[0, j * M:(j + 1) * M] = bs
            rows_in += [r, -r]; b_in += [META["vlim"][j] * META["vscale"],
                                         META["vlim"][j] * META["vscale"]]
    for j in range(5):
        for a in range(M):
            r = np.zeros((1, n)); r[0, j * M + a] = 1.0
            rows_in += [r, -r]; b_in += [META["cbox"][j]] * 2
    A_in = np.vstack(rows_in)
    b_in = np.array(b_in)

    x = ca.SX.sym("x", n)
    p = ca.SX.sym("p", 21)
    f = 0.5 * ca.dot(x, ca.DM(H) @ x)
    g = ca.vertcat(A_eq @ x - B_eq @ p, A_in @ x)
    return {"x": x, "f": f, "g": g, "p": p}, A_eq.shape[0], A_in.shape[0], \
        (A_eq, B_eq, A_in, b_in, H)


def solve_case(solver, P, n_eq, n_in, name):
    lbg = np.concatenate([np.zeros(n_eq), -np.inf * np.ones(n_in)])
    ubg = np.concatenate([np.zeros(n_eq), np.inf * np.ones(n_in)])
    t0 = time.perf_counter()
    r = solver(p=P, lbg=lbg, ubg=ubg)
    dt = (time.perf_counter() - t0) * 1e3
    st = solver.stats()
    x = np.array(r["x"]).ravel()
    print(f"[{name}] success={st.get('success')} iter={st.get('iter_count')} "
          f"obj={float(r['f']):.2f} 耗时={dt:.2f}ms")
    return x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nocodegen", action="store_true")
    args = ap.parse_args()

    qp, n_eq, n_in, (A_eq, B_eq, A_in, b_in, H) = build_qp()
    print(f"QP: 变量={qp['x'].numel()} 等式={n_eq} 不等式={n_in} "
          f"H 条件数≈{np.linalg.cond(H):.1e}")
    solver = ca.qpsol("arm_qp", "osqp", qp, {"verbose": False, "error_on_fail": False})

    Tn = 0.65
    s_hit = META["s_hit"]
    cases = {
        "低位族": dict(q_hit=[0.2, -0.5, 0.3, -1.4, 0.5], dq_hit=[3.0, 1.5, 1.0, 0.0, 0.0]),
        "高位族": dict(q_hit=[0.1, -0.2, 0.1, -1.2, 0.8], dq_hit=[3.5, 2.0, 1.2, 0.0, 0.0]),
    }
    for name, c in cases.items():
        # C++ 契约:dq 乘 T 转 s 域
        P = np.array(list([0, -1.0, 0.5, 0, 0.8]) + [0.0] * 5
                     + list(c["q_hit"]) + list(np.array(c["dq_hit"]) * Tn) + [Tn])
        x = solve_case(solver, P, n_eq, n_in, name)
        M = META["M"]
        Cm = x.reshape(5, META["M"])  # 关节连续块:x[j*M+a]=C[j][a]
        qh = Cm @ basis(s_hit, 0)
        dh = Cm @ basis(s_hit, 1) / Tn
        vmax = amax = 0.0
        for k in range(1, 200):
            s = k / 200
            vmax = max(vmax, float(np.max(np.abs(Cm @ basis(s, 1))) / Tn))
            amax = max(amax, float(np.max(np.abs(Cm @ basis(s, 2))) / Tn ** 2))
        print(f"   q(sh)={np.round(qh, 3)} 误差={np.max(np.abs(qh - np.array(c['q_hit']))):.2e}")
        print(f"   dq(sh)={np.round(dh, 2)} 峰|dq|={vmax:.1f} 峰|ddq|={amax:.0f}")

    if not args.nocodegen:
        # 数据头导出(2026-10-06 改版):H/A 全常数、b 线性依赖 p → 不需要 codegen
        # 运行时求解器(casadi qpsol codegen 包装层与 vendored OSQP 互操作失败:
        # 同数据 python 内置 OSQP 收敛、生成 C+vendored OSQP 判不可行,根因未深挖)。
        # 导出 H/A/B_eq 常量,C++ 直调 vendored OSQP(osqp_setup/update_data_vec)。
        OUT_C.parent.mkdir(parents=True, exist_ok=True)
        _, _, _, (A_eq, B_eq, A_in, b_in, H) = 0, 0, 0, build_qp()[3]
        A = np.vstack([A_eq, A_in])                       # (355,45)
        m_ng, n_x = A.shape
        row_ub = np.concatenate([np.zeros(n_eq), b_in])   # 行上界(等式行占位 0)
        hdr = [f"// 自动生成:python/tools/gen_arm_nlp.py —— 臂轨迹 QP 常量数据",
               f"// 问题: min ½xᵀHx  s.t.  l ≤ Ax ≤ u(前 {n_eq} 行等式,其余 ≤)",
               f"#pragma once", "#include <array>", "",
               f"namespace sckd::arm_qp_data {{",
               f"inline constexpr int kNx = {n_x};",
               f"inline constexpr int kNg = {m_ng};",
               f"inline constexpr int kNeq = {n_eq};", ""]
        def arr(name, mat, fmt="%.10e"):
            if mat.ndim == 1:
                vals = ", ".join(fmt % v for v in mat)
                return f"inline constexpr double {name}[{mat.size}] = {{{vals}}};"
            rows = []
            for r in mat:
                rows.append("{%s}" % ", ".join(fmt % v for v in r))
            return (f"inline constexpr double {name}[{mat.shape[0]}][{mat.shape[1]}] = {{"
                    + ", ".join(rows) + "};")
        hdr.append(arr("H", H))
        hdr.append(arr("A", A))
        hdr.append(arr("B_eq", B_eq))                     # 等式右端 = B_eq @ p
        hdr.append(arr("u_row", row_ub))                  # 不等式行上界(等式行忽略)
        hdr.append("}  // namespace sckd::arm_qp_data")
        (OUT_C.parent / "arm_qp_data.h").write_text("\n".join(hdr) + "\n")
        meta = dict(META)
        meta["n_eq"], meta["n_ineq"] = n_eq, n_in
        meta["param_order"] = "q0(5),dq0(5),q_hit(5),dq_hit(5),T(dq 类已乘T入s域)"
        meta["var_layout"] = "x[j*M+a] = C[j][a](关节 j 连续块)"
        OUT_JSON.write_text(json.dumps(meta, ensure_ascii=False, indent=2))
        print(f"[OK] 数据头 → {OUT_C.parent / 'arm_qp_data.h'} "
              f"(H {H.shape}, A {A.shape}, B_eq {B_eq.shape}) + {OUT_JSON}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
