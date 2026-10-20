#!/usr/bin/env python3
"""拍-球反弹标定台架(§3.5 拍面支,2026-10-07):受控击球采样。

真实跑动接触太杂(擦边/边缘法向/自旋耦合混杂,m2_intercept [calib] 28 样本
1D 回归 R²<0)→ 按设计文档本意做受控实验:
  拍:臂摆到固定位形(FK 搜索:法向≈意图 n*、拍高≈0.95),关节速度每步
     强制为 dq=J⁺·v_r(运动学恒速拍,无限惯量速度源——标定接触映射本身)
  球:置于法向前 0.25m,速度 v_in=−n·|vin_n|,自旋 ω(绕 y)
  采样:vr_n×vin_n×命中偏移×自旋 → 接触分离后稳态 v_out
拟合:v_out·n = a·vr_n + b·v_in_n(拍心/无旋子集)→ 替换 ReboundInverse
的 e=0.6;偏移/自旋敏感度作观测项留档。
用法(conda sim):python python/tools/rebound_calib.py [model.mjcf]
"""
import sys

import casadi as ca
import mujoco
import numpy as np

from arm_fk_sym import build, base_height

N_STAR = np.array([np.cos(np.radians(15)), 0.0, np.sin(np.radians(15))])


def find_pose(fk):
    """随机搜索 q:拍面法向≈N_STAR、拍心 z≈0.95、位形留在真实击球族
    (ready 附近)——自由域会搜出近奇异拧位形,拍沿法向有效质量极小,
    球把拍打飞、出射非物理(2026-10-07 定位)。"""
    qS = fk['q']
    Fn = ca.Function('n', [qS], [fk['f_R'](qS) @ ca.SX([0, 1, 0])])
    Fp = fk['f_pos']
    rng = np.random.default_rng(11)
    q_ready = np.array([0.0, -1.0, 0.5, 0.0, 0.8])
    lo = q_ready + np.array([-1.2, -0.8, -0.9, -1.5, -1.5])
    hi = q_ready + np.array([1.2, 0.8, 0.0, 1.5, 1.5])
    best, bc = None, 1e9
    for _ in range(60000):
        q = rng.uniform(lo, hi)
        n = np.array(Fn(q)).reshape(3)
        z = float(Fp(q)[2])
        c = 4 * (1 - n @ N_STAR) ** 2 + max(0.0, abs(z - 0.95) - 0.10) ** 2
        if c < bc:
            bc, best = c, q
    n = np.array(Fn(best)).reshape(3)
    print(f'[pose] q={np.round(best, 3)} n·n*={n @ N_STAR:.4f} n={np.round(n, 3)} '
          f'拍高(base系)={float(Fp(best)[2]):.3f}')
    assert n @ N_STAR > 0.995, '位形搜索失败:法向未对准'
    return best


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else 'models/generated/x4_tennis_demo.mjcf'
    fk = build(path)
    m = mujoco.MjModel.from_xml_path(path)
    d = mujoco.MjData(m)
    sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, 'racket_center')
    bgid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, 'ball_geom')
    qadr, dofadr = [], []                       # 臂关节地址(链序)
    b = m.site_bodyid[sid]
    while mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) != 'base_link':
        j = m.body_jntadr[b]
        qadr.append(m.jnt_qposadr[j])
        dofadr.append(m.jnt_dofadr[j])
        b = m.body_parentid[b]
    qadr.reverse(); dofadr.reverse()
    base_j = m.body_jntadr[b]                    # base_link 自由关节
    bdof = m.jnt_dofadr[base_j]
    ball_q = m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, 'ball_free')]
    ball_v = m.jnt_dofadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, 'ball_free')]
    base_world = base_height(path)

    q_pose = find_pose(fk)
    qS = fk['q']
    J = np.array(ca.Function('J', [qS], [ca.jacobian(fk['f_pos'](qS), qS)])(q_pose)
                 ).reshape(3, 5)
    Jpin = np.linalg.pinv(J)
    n_out = np.array(
        ca.Function('n', [qS], [fk['f_R'](qS) @ ca.SX([0, 1, 0])])(q_pose)).reshape(3)
    site_local = np.array(fk['f_pos'](q_pose)).reshape(3)
    t1 = np.cross(n_out, [0, 1, 0.001])         # 面内切向(近水平)
    t1 /= np.linalg.norm(t1)
    sp_world = base_world + site_local

    def run(vr_n, vin_n, off, spin):
        """单次受控击球:返回稳态 v_out(世界系)或 None。"""
        mujoco.mj_resetData(m, d)
        mujoco.mj_resetDataKeyframe(m, d, 0)
        for i, a_ in enumerate(qadr):
            d.qpos[a_] = q_pose[i]
        aim = sp_world + t1 * off
        bp0 = aim + n_out * 0.25
        for k in range(3):
            d.qpos[ball_q + k] = bp0[k]
        d.qpos[ball_q + 3:ball_q + 7] = [1, 0, 0, 0]
        for k in range(3):
            d.qvel[ball_v + k] = -n_out[k] * abs(vin_n)
        d.qvel[ball_v + 3:ball_v + 6] = [0.0, spin, 0.0]   # 绕 y 自旋
        dq = Jpin @ (n_out * vr_n)                          # 拍速沿法向
        touched, sep = False, 0
        for _ in range(1200):                               # ≤1.2s
            # 恒速强制仅在触球前(建立稳定挥拍);触球后停强制——臂按自身
            # 惯性随挥。否则恒速拍会追着慢出球反复连击,出射非单调(定位于
            # vr=8 出射反而低于 vr=4,2026-10-07)
            if not touched:
                for i, a_ in enumerate(dofadr):
                    d.qvel[a_] = dq[i]
            for k in range(6):
                d.qvel[bdof + k] = 0.0                      # 底盘冻结
            has = any((d.contact[c].geom1 == bgid or d.contact[c].geom2 == bgid)
                      for c in range(d.ncon))
            if has:
                touched, sep = True, 0
            elif touched:
                sep += 1
                if sep >= 30:                               # 分离 30ms → 稳态
                    return d.qvel[ball_v:ball_v + 3].copy()
            mujoco.mj_step(m, d)
        return None

    print('\n== 受控击球扫描(拍心/无旋子集 → 拟合 a,b)==')
    rows = []
    for vr_n in (2.0, 4.0, 6.0, 8.0):
        for vin_n in (-4.0, -6.0, -8.0):
            vo = run(vr_n, vin_n, 0.0, 0.0)
            if vo is None:
                print(f'  vr={vr_n} vin={vin_n}: 未接触/超时')
                continue
            rows.append((vr_n, vin_n, float(vo @ n_out)))
    A = np.array([[r[0], r[1]] for r in rows])
    y = np.array([r[2] for r in rows])
    x, *_ = np.linalg.lstsq(A, y, rcond=None)
    r2 = 1 - ((y - A @ x) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    a, bb = x
    print(f'  拟合 v_out·n = {a:.3f}·vr_n + {bb:.3f}·v_in_n  (n={len(y)},R²={r2:.4f},'
          f'a+b={a + bb:.3f},理想 2)→ 等效 e = a−1 = {a - 1:.3f}(planner 现行 0.60)')
    for r in rows:
        print(f'    vr={r[0]:.0f} vin={r[1]:.0f} → v_out·n={r[2]:.2f}')

    print('\n== 敏感度(固定 vr_n=6, vin_n=−6)==')
    for off in (0.0, 0.04, 0.08, -0.04):
        vo = run(6.0, -6.0, off, 0.0)
        if vo is not None:
            print(f'  偏移 {off * 100:+.0f}cm → v_out·n={vo @ n_out:.2f} '
                  f'|v_out|={np.linalg.norm(vo):.2f}')
    for spin in (-84.0, 0.0, 84.0):                        # ±800rpm
        vo = run(6.0, -6.0, 0.0, spin)
        if vo is not None:
            print(f'  自旋 {spin:.0f} rad/s → v_out·n={vo @ n_out:.2f} '
                  f'|v_out|={np.linalg.norm(vo):.2f}')


if __name__ == '__main__':
    main()
