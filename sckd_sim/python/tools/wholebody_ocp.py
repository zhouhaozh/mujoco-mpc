#!/usr/bin/env python3
"""方案 B v0 整机 OCP 离线原型(2026-10-07,设计文档 §6.2 分期首步)。

问题:底盘(平面运动学,x,y,vx,vy,加速度界)+ 5 轴臂(q,dq,加速度界)联合,
硬终端流形:拍心=p*、拍速=v*(v_base + J·dq —— 耦合项即方案 B 的全部意义)、
拍面法向软对准 n*。IPOPT(mumps),CasADi opti。
对照:B′ 的 v_base* 启发式(接近方向投影+Hermite 上限)两次求解——
  (a) 终端底盘速度自由 → 联合最优 v_base_opt
  (b) 终端底盘速度钉死 = B′ 启发式 → 成本差 = 分解的"红利/损失"度量
场景取自 m2_intercept 实测评测日志(固定族高位 + 随机族远跑)。
用法(conda sim):python python/tools/wholebody_ocp.py [model.mjcf]
"""
import sys
import time

import casadi as ca
import mujoco
import numpy as np

from arm_fk_sym import build, base_height

# ---- 与 C++ 同源的参数(hit_planner/base_ref_gen/m2_intercept 契约) ----
REBOUND_A, REBOUND_B = 1.5, -0.5   # a=1+k,b=−k(k=0.5;§3.5 拍面支,与 C++ 同源;
                                    # 2026-10-08 由旧 e=0.6 更新——旧 b 符号与反射物理相反)
V_OUT_SPEED, V_OUT_ELEV = 8.0, 15.0     # intent
FLY_SPEED_MAX, HERMITE_W = 1.2, 2.2     # v_base* 上限
BACK = 0.8                    # BaseTargetFromHit 后撤距
Q_READY = np.array([0.0, -1.0, 0.5, 0.0, 0.8])
DQ_LIM = np.array([18.0, 18.0, 25.0, 25.0, 30.0])
DDQ_LIM = np.array([80.0, 80.0, 120.0, 120.0, 200.0])
V_CH_MAX, A_CH_MAX = 1.5, 8.0
N = 20

SCENARIOS = {
    # 实测日志代表工况:[plan] 行 + [lock] flt v(击球点来流速度近似)
    'S1 固定族高位': dict(p_hit=(0.81, -0.97, 1.21), v_in=(-3.4, -0.6, -6.2), T=0.98),
    'S2 随机族远跑': dict(p_hit=(1.67, -1.00, 1.28), v_in=(-3.2, -0.8, -5.6), T=0.94),
}

# v2 场景:抽球族(2026-10-08),起点 = 闭环 Init 时刻的滞后底盘状态
# (base0/vbase0 取自 m2_intercept gs 日志量级;lag 沿 d̂ ~0.15-0.25 是实测)
V2_SCENARIOS = {
    'G1 gs 中央': dict(p_hit=(0.95, -0.55, 0.70), v_in=(-5.5, -0.3, -1.7), T=0.85,
                      base0=(-0.05, -0.10), vbase0=(0.8, -0.4)),
    'G2 gs 侧向': dict(p_hit=(0.70, 0.90, 0.62), v_in=(-5.8, 0.2, -1.5), T=0.95,
                      base0=(0.05, 0.30), vbase0=(0.2, -1.0)),
    'G3 gs 远跑紧时': dict(p_hit=(1.30, -0.85, 0.75), v_in=(-5.2, -0.5, -1.8), T=0.80,
                        base0=(0.0, -0.20), vbase0=(1.1, -0.5)),
}
V2_LAG = 0.18      # triage 补丁口径的闭环残差(沿 d̂,P 控误差动力学终值量级)


def rebound_inverse(v_in, v_out):
    """hit_planner.cpp ReboundInverse 同式:n*=v_out/|v_out|,v*=n*·vr_n。"""
    n = np.array(v_out) / max(np.linalg.norm(v_out), 0.01)
    vo = np.linalg.norm(v_out)
    v_in_n = np.dot(v_in, n)
    vr_n = (vo - REBOUND_B * v_in_n) / REBOUND_A
    return n * vr_n, n


def bprime_vbase_star(v_star, base_target, T):
    """B′ 启发式(hit_planner.cpp Plan 内同式):d̂ 上投影 + 双上限。"""
    d = np.array(base_target, dtype=float)
    D0 = np.hypot(d[0], d[1])
    if D0 <= 0.30:
        return np.zeros(2), D0, 0.0
    dhat = d / D0
    va = v_star[0] * dhat[0] + v_star[1] * dhat[1]
    if va <= 0.05:
        return np.zeros(2), D0, va
    cap = min(FLY_SPEED_MAX, HERMITE_W * D0 / max(T, 0.3))
    vf = min(va, cap)
    return vf * dhat, D0, va


def solve_ocp(fk, h_base, scen, v_base_fix=None, pin_base=None):
    """解整机 OCP;v_base_fix 非 None 时终端底盘速度钉死(对照 run);
    pin_base 非 None 时终端底盘【位置】钉死(v2:闭环现状/补丁对照);
    起点取 scen['base0']/['vbase0'](缺省原点零速,v0 口径)。"""
    p_hit = np.array(scen['p_hit'])
    v_in = np.array(scen['v_in'])
    v_out = V_OUT_SPEED * np.array([
        np.cos(np.radians(V_OUT_ELEV)), 0.0, np.sin(np.radians(V_OUT_ELEV))])
    v_star, n_star = rebound_inverse(v_in, v_out)
    T = scen['T']
    h = T / N

    f_pos, f_R = fk['f_pos'], fk['f_R']
    qS = fk['q']
    J_sym = ca.jacobian(f_pos(qS), qS)             # 3×5(yaw=0,平移不进 J)
    n_sym = f_R(qS) @ ca.SX([0, 1, 0])             # 拍面法向(m2 同约定)
    Fk_pos = ca.Function('Fp', [qS], [f_pos(qS)])
    Fk_J = ca.Function('FJ', [qS], [J_sym])
    Fk_n = ca.Function('Fn', [qS], [n_sym])

    opti = ca.Opti()
    X = [opti.variable(14) for _ in range(N + 1)]
    U = [opti.variable(7) for _ in range(N)]
    s_pos = opti.variable(3)      # 终端松弛(诊断;重罚)
    s_vel = opti.variable(3)

    x0_full = np.concatenate([list(scen.get('base0', (0.0, 0.0)))
                              + list(scen.get('vbase0', (0.0, 0.0))),
                              Q_READY, np.zeros(5)])
    opti.subject_to(X[0] == x0_full)
    for k in range(N):
        xk, uk = X[k], U[k]
        x, y, vx, vy = xk[0], xk[1], xk[2], xk[3]
        q, dq = xk[4:9], xk[9:14]
        ax, ay, ddq = uk[0], uk[1], uk[2:7]
        xn = X[k + 1]
        opti.subject_to(xn[0] == x + h * vx + 0.5 * h * h * ax)
        opti.subject_to(xn[1] == y + h * vy + 0.5 * h * h * ay)
        opti.subject_to(xn[2] == vx + h * ax)
        opti.subject_to(xn[3] == vy + h * ay)
        for j in range(5):
            opti.subject_to(xn[4 + j] == q[j] + h * dq[j] + 0.5 * h * h * ddq[j])
            opti.subject_to(xn[9 + j] == dq[j] + h * ddq[j])
            opti.subject_to(opti.bounded(-DDQ_LIM[j], ddq[j], DDQ_LIM[j]))
        opti.subject_to(opti.bounded(-A_CH_MAX, ax, A_CH_MAX))
        opti.subject_to(opti.bounded(-A_CH_MAX, ay, A_CH_MAX))
    for k in range(N + 1):                      # 底盘速度界全时窗(含终端拍速度!
        opti.subject_to(opti.bounded(-V_CH_MAX, X[k][2], V_CH_MAX))
        opti.subject_to(opti.bounded(-V_CH_MAX, X[k][3], V_CH_MAX))
    for k in range(N + 1):                          # 关节位置/速度界(全时窗)
        for j in range(5):
            opti.subject_to(opti.bounded(-DQ_LIM[j], X[k][9 + j], DQ_LIM[j]))
            opti.subject_to(opti.bounded(scen['q_lo'][j], X[k][4 + j], scen['q_hi'][j]))

    # 终端流形
    qN = X[N][4:9]
    pos_N = Fk_pos(qN) + ca.vertcat(X[N][0], X[N][1], h_base)
    vel_N = ca.vertcat(X[N][2], X[N][3], 0) + Fk_J(qN) @ X[N][9:14]
    opti.subject_to(pos_N == p_hit + s_pos)
    opti.subject_to(vel_N == v_star + s_vel)
    opti.subject_to(opti.bounded(-0.05, s_pos, 0.05))
    opti.subject_to(opti.bounded(-0.30, s_vel, 0.30))
    if v_base_fix is not None:
        opti.subject_to(X[N][2] == v_base_fix[0])
        opti.subject_to(X[N][3] == v_base_fix[1])
    if pin_base is not None:
        opti.subject_to(X[N][0] == pin_base[0])
        opti.subject_to(X[N][1] == pin_base[1])

    cost = 0
    for k in range(N):
        uk = U[k]
        cost += h * ((uk[0]**2 + uk[1]**2) / A_CH_MAX**2
                     + sum((uk[2 + j] / DDQ_LIM[j])**2 for j in range(5)))
    cost += 1e3 * ca.sumsqr(s_vel) + 1e2 * ca.sumsqr(s_pos)
    cost += 20.0 * (1 - ca.dot(Fk_n(qN), n_star) / np.linalg.norm(n_star))
    opti.minimize(cost)
    for k in range(N + 1):
        opti.set_initial(X[k][4:9], Q_READY)

    p_opts = {'print_time': False, 'expand': True}
    s_opts = {'print_level': 0, 'max_iter': 800, 'linear_solver': 'mumps', 'sb': 'yes'}
    opti.solver('ipopt', p_opts, s_opts)
    t0 = time.time()
    try:
        sol = opti.solve()
        ok = True
    except RuntimeError as e:
        sol = opti.debug
        ok = False
        print(f'      [ipopt] {str(e).splitlines()[-1][:120]}')
    ms = (time.time() - t0) * 1e3

    def val(e):
        return np.array(sol.value(e)).reshape(-1)

    xN = val(X[N]) if ok else None
    u = np.array([val(U[k]) for k in range(N)]) if ok else None
    r = dict(ok=ok, ms=ms, cost=float(sol.value(cost)) if ok else np.nan,
             v_base=(xN[2], xN[3]) if ok else None,
             base=(xN[0], xN[1]) if ok else None,
             q_hit=xN[4:9] if ok else None,
             dq=xN[9:14] if ok else None, slack_p=val(s_pos) if ok else None,
             slack_v=val(s_vel) if ok else None,
             n_dot=float(sol.value(ca.dot(Fk_n(qN), n_star))) if ok else np.nan, u=u)
    return r, v_star, n_star


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else 'models/generated/x4_tennis_demo.mjcf'
    fk = build(path)
    h_base = float(base_height(path)[2])
    m = mujoco.MjModel.from_xml_path(path)
    q_lo, q_hi = np.zeros(5), np.zeros(5)      # 关节限位(MJCF,链序)
    b = m.site_bodyid[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, 'racket_center')]
    i = 4
    while mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) != 'base_link':
        j = m.body_jntadr[b]
        # 无限位关节(jnt_limited=0)jnt_range 读出 [0,0] 是垃圾值 → 取 ±3.5
        # (ArmSwing QP 系数盒 [3,3,3,3,6] 同为宽松口径;线缆行程未确认,§3.2)
        if m.jnt_limited[j]:
            q_lo[i], q_hi[i] = m.jnt_range[j]
        else:
            q_lo[i], q_hi[i] = -3.5, 3.5
        b = m.body_parentid[b]
        i -= 1

    print(f'== 方案 B v0 整机 OCP(N={N},IPOPT/mumps;终端:拍心/p*、拍速/v* '
          f'等式+松弛,法向软对准)==')
    for name, scen0 in SCENARIOS.items():
        scen = dict(scen0, q_lo=q_lo, q_hi=q_hi)
        yaw = np.arctan2(scen['p_hit'][1], scen['p_hit'][0])
        bt = np.array(scen['p_hit'][:2]) - BACK * np.array([np.cos(yaw), np.sin(yaw)])
        r_free, v_star, n_star = solve_ocp(fk, h_base, scen, None)
        vb_he, D0, va = bprime_vbase_star(v_star, bt, scen['T'])
        r_fix, _, _ = solve_ocp(fk, h_base, scen, vb_he)
        print(f'\n-- {name}:p*={scen["p_hit"]} T={scen["T"]}s '
              f'base_target=({bt[0]:.2f},{bt[1]:.2f}) D0={D0:.2f}m')
        print(f'   v*={np.round(v_star, 2)} |v*|={np.linalg.norm(v_star):.2f} m/s')
        print(f'   B′ 启发式 v_base*={np.round(vb_he, 2)} '
              f'({np.hypot(*vb_he):.2f} m/s,投影 va={va:.2f})')
        for tag, r in (('自由(联合最优)', r_free), ('钉死=B′启发式', r_fix)):
            if not r['ok']:
                print(f'   [{tag}] 求解失败({r["ms"]:.0f}ms)')
                continue
            vb = r['v_base']
            print(f'   [{tag}] cost={r["cost"]:.3f} {r["ms"]:.0f}ms '
                  f'v_base=({vb[0]:.2f},{vb[1]:.2f}) {np.hypot(*vb):.2f} m/s '
                  f'(与 B′ 差 {np.hypot(vb[0]-vb_he[0], vb[1]-vb_he[1]):.2f}) '
                  f'n·n*={r["n_dot"]:.3f} 松弛 |sp|={np.linalg.norm(r["slack_p"]):.3f} '
                  f'|sv|={np.linalg.norm(r["slack_v"]):.3f} '
                  f'peak|dq|={np.abs(r["dq"]).max():.1f}')
        if r_free['ok'] and r_fix['ok']:
            dc = r_fix['cost'] - r_free['cost']
            print(f'   分解代价:B′ 相对联合最优 cost 差 {dc:+.3f} '
                  f'({100 * dc / max(r_free["cost"], 1e-9):+.1f}%)')


def sweep():
    """v1 第一步:联合最优 v_base* 的参数扫描(方向 θ×跑道 D0×时窗 T)。

    目标:检验 v0 两点发现是否成规律——幅值 ≈ k·D0/T、方向 ≈ d̂ 与 v̂* 的
    混合;若干净 → 蒸馏成闭合规则进 hit_planner 闭环 A/B,否则结论为
    "启发式不可蒸馏,须在线 OCP"。
    """
    fk = build('models/generated/x4_tennis_demo.mjcf')
    h_base = float(base_height('models/generated/x4_tennis_demo.mjcf')[2])
    m = mujoco.MjModel.from_xml_path('models/generated/x4_tennis_demo.mjcf')
    q_lo, q_hi = np.zeros(5), np.zeros(5)
    b = m.site_bodyid[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, 'racket_center')]
    i = 4
    while mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) != 'base_link':
        j = m.body_jntadr[b]
        q_lo[i], q_hi[i] = m.jnt_range[j] if m.jnt_limited[j] else (-3.5, 3.5)
        b = m.body_parentid[b]
        i -= 1
    v_out = V_OUT_SPEED * np.array([
        np.cos(np.radians(V_OUT_ELEV)), 0.0, np.sin(np.radians(V_OUT_ELEV))])
    print('  θ(°)   D0    T   |v_opt| ang_opt  |v_heur| ang_heur  D0/T  v_opt/(D0/T)  '
          'cost_heur/cost_opt')
    for T in (0.95, 1.25):
        for th in (0.0, -20.0, -40.0, -60.0):
            for D0 in (0.4, 0.7, 1.0, 1.4, 1.8):
                r_hit = D0 + BACK          # |p_hit_xy|(BaseTargetFromHit 反解)
                ph = (r_hit * np.cos(np.radians(th)),
                      r_hit * np.sin(np.radians(th)), 1.1)
                scen = dict(p_hit=ph, v_in=(-3.4, -0.5, -6.0), T=T,
                            q_lo=q_lo, q_hi=q_hi)
                r_free, v_star, _ = solve_ocp(fk, h_base, scen, None)
                if not r_free['ok']:
                    print(f'  {th:5.0f} {D0:4.1f} {T:4.2f}  FAIL')
                    continue
                vb_he, _, _ = bprime_vbase_star(v_star, ph[:2] - BACK * np.array(
                    [np.cos(np.radians(th)), np.sin(np.radians(th))]), T)
                r_fix, _, _ = solve_ocp(fk, h_base, scen, vb_he)
                vo = np.array(r_free['v_base'])
                ratio = np.hypot(*vo) / (D0 / T)
                cr = (r_fix['cost'] / r_free['cost']) if r_fix['ok'] else np.nan
                print(f'  {th:5.0f} {D0:4.1f} {T:4.2f}  {np.hypot(*vo):6.2f} '
                      f'{np.degrees(np.arctan2(vo[1], vo[0])):6.1f}  '
                      f'{np.hypot(*vb_he):6.2f}   {np.hypot(*vb_he) * 0 + np.degrees(np.arctan2(vb_he[1], vb_he[0])):6.1f}  '
                      f'{D0 / T:4.2f}  {ratio:6.2f}  {cr:8.2f}')


def v2():
    """v2 原型:终端底盘位置自由的红利(triage 后,2026-10-08)。

    三跑对照(同一工况,起点 = 闭环 Init 时刻的滞后底盘状态 base0/vbase0):
      A joint :终端底盘位置/速度全自由(联合最优)
      B bprime:位置钉 base_target(back=0.8)+速度钉 B′ 启发式 = 闭环现状
      C pred  :位置钉 base_pred(base_target−lag·d̂)= triage 补丁的锚
    关键读数:B/C 的终端松弛 |sp| = 拍心偏离 p* 的距离 —— 闭环"面内 miss"
    的离线代理;A 的 base 位置 = 联合最优想把底盘放哪(距 base_target 多远)。
    """
    fk = build('models/generated/x4_tennis_demo.mjcf')
    h_base = float(base_height('models/generated/x4_tennis_demo.mjcf')[2])
    m = mujoco.MjModel.from_xml_path('models/generated/x4_tennis_demo.mjcf')
    q_lo, q_hi = np.zeros(5), np.zeros(5)
    b = m.site_bodyid[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, 'racket_center')]
    i = 4
    while mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) != 'base_link':
        j = m.body_jntadr[b]
        q_lo[i], q_hi[i] = m.jnt_range[j] if m.jnt_limited[j] else (-3.5, 3.5)
        b = m.body_parentid[b]
        i -= 1

    print('== 方案 B v2 原型:终端底盘位置自由 vs 钉死(抽球族工况,起点=滞后底盘) ==\n')
    for name, scen0 in V2_SCENARIOS.items():
        scen = dict(scen0, q_lo=q_lo, q_hi=q_hi)
        p_hit = np.array(scen['p_hit'])
        base0 = np.array(scen['base0'], dtype=float)
        # BaseTargetFromHit 同式:yaw = 当前底盘→击球点方向
        yaw = np.arctan2(p_hit[1] - base0[1], p_hit[0] - base0[0])
        dhat = np.array([np.cos(yaw), np.sin(yaw)])
        bt = p_hit[:2] - BACK * dhat
        pred = bt - V2_LAG * dhat
        rA, v_star, n_star = solve_ocp(fk, h_base, scen, None, None)
        vb_he, D0, va = bprime_vbase_star(v_star, bt, scen['T'])
        rB, _, _ = solve_ocp(fk, h_base, scen, vb_he, bt)
        rC, _, _ = solve_ocp(fk, h_base, scen, vb_he, pred)
        print(f'-- {name}:p*={scen["p_hit"]} T={scen["T"]}s 起点 base={scen["base0"]}'
              f' v={scen["vbase0"]}')
        print(f'   base_target=({bt[0]:.2f},{bt[1]:.2f}) D0={D0:.2f}m '
              f'base_pred=({pred[0]:.2f},{pred[1]:.2f}) '
              f'|v*|={np.linalg.norm(v_star):.2f} B′v_base*={np.hypot(*vb_he):.2f} m/s')
        for tag, r in (('A 联合自由', rA), ('B 钉B′(闭环现状)', rB),
                       ('C 钉pred(triage锚)', rC)):
            if not r['ok']:
                print(f'   [{tag}] 求解失败({r["ms"]:.0f}ms)')
                continue
            bN = np.array(r['base'])
            print(f'   [{tag}] cost={r["cost"]:.4f} {r["ms"]:.0f}ms '
                  f'终端base=({bN[0]:.2f},{bN[1]:.2f})'
                  f'(离target {np.hypot(*(bN - bt)):.2f}m) '
                  f'|sp|={np.linalg.norm(r["slack_p"]):.3f} '
                  f'|sv|={np.linalg.norm(r["slack_v"]):.3f} '
                  f'peak|dq|={np.abs(r["dq"]).max():.1f} n·n*={r["n_dot"]:.3f}')
        if rA['ok'] and rB['ok'] and rC['ok']:
            dc = rB['cost'] - rA['cost']
            print(f'   红利:B 相对 A cost {100 * dc / max(rA["cost"], 1e-9):+.1f}%'
                  f' | 拍心松弛 |sp|: B={np.linalg.norm(rB["slack_p"]):.3f}'
                  f' C={np.linalg.norm(rC["slack_p"]):.3f}'
                  f' A={np.linalg.norm(rA["slack_p"]):.3f} m')
        print()


def v2sweep():
    """v2 第二步:联合最优的终端后撤距 back*(z) 扫描——可蒸馏性检验。

    v2 三跑发现:低位抽球(back=0.8 钉死)拍心松弛 7.2cm + 法向偏 39°,联合
    最优把底盘前移到离 p* 仅 ~0.34m——0.8 后撤把低位击球推到臂边缘。若
    back* 随击球高 z 有干净规律 → 蒸馏进 BaseTargetFromHit(v1 方法论:先蒸馏
    再上在线 OCP);否则须真 OCP。
    """
    fk = build('models/generated/x4_tennis_demo.mjcf')
    h_base = float(base_height('models/generated/x4_tennis_demo.mjcf')[2])
    m = mujoco.MjModel.from_xml_path('models/generated/x4_tennis_demo.mjcf')
    q_lo, q_hi = np.zeros(5), np.zeros(5)
    b = m.site_bodyid[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, 'racket_center')]
    i = 4
    while mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) != 'base_link':
        j = m.body_jntadr[b]
        q_lo[i], q_hi[i] = m.jnt_range[j] if m.jnt_limited[j] else (-3.5, 3.5)
        b = m.body_parentid[b]
        i -= 1

    print('== v2 sweep:back*(击球高 z;θ=0 正前,r=1.05,T=0.9,起点(0,0)v=(0.6,0)) ==')
    print('   z*   back*  |sp|@0.8  |sp|@0.45  n·n*@free  peak|dq|  cost比(0.8/free)')
    for z in (0.55, 0.65, 0.75, 0.90, 1.10, 1.30, 1.60):
        p_hit = np.array([1.05, 0.0, z])
        scen = dict(p_hit=p_hit, v_in=(-5.5, -0.2, -1.6), T=0.9,
                    base0=(0.0, 0.0), vbase0=(0.6, 0.0), q_lo=q_lo, q_hi=q_hi)
        rA, v_star, _ = solve_ocp(fk, h_base, scen, None, None)
        if not rA['ok']:
            print(f'  {z:4.2f}  FAIL')
            continue
        bA = np.array(rA['base'])
        back_star = (p_hit[:2] - bA)[0]          # θ=0:d̂=x̂,投影即 x 差
        sp08 = sp045 = np.nan
        for back, tag in ((0.8, 'sp08'), (0.45, 'sp045')):
            rB, _, _ = solve_ocp(fk, h_base, scen, None, (1.05 - back, 0.0))
            if rB['ok']:
                if tag == 'sp08':
                    sp08 = np.linalg.norm(rB['slack_p'])
                else:
                    sp045 = np.linalg.norm(rB['slack_p'])
        print(f'  {z:4.2f}  {back_star:5.2f}   {sp08:6.3f}    {sp045:6.3f}'
              f'      {rA["n_dot"]:.3f}      {np.abs(rA["dq"]).max():4.1f}')


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'sweep':
        sweep()
    elif len(sys.argv) > 1 and sys.argv[1] == 'v2':
        v2()
    elif len(sys.argv) > 1 and sys.argv[1] == 'v2sweep':
        v2sweep()
    else:
        main()
