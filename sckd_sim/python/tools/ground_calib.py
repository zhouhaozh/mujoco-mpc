#!/usr/bin/env python3
"""地面反弹标定(§3.5 地面支,2026-10-08):落球扫描拟合 e_z / μ(旋转耦合)。

数据源 = MuJoCo 原始 ball↔floor 软接触(在排除对/解析冲量化之前的模型上跑;
拟合结果 = "已知碰撞模型"的参数初值,实机实测到位后同形式直接替换)。
扫描:vz(2-7) × 水平速度(0/2/4 m/s) × 旋转(绕 y 上/下旋、绕 z 侧旋)。
每发:记录接触前最后一步与分离后第一步的 (v, ω)。
模型(刚体 Coulomb+stick,接触点 r_c=(0,0,−R)):
  法向  vz' = −e_z·vz
  切向  u = v_t + ω×r_c(即 (vx−R·ωy, vy+R·ωx))
        粘滞冲量 P = −(2/7)·u(使 u'=0);界 |P| ≤ μ·(1+e_z)·|vz|
        滑动  |P| = μ·(1+e_z)·|vz|,方向 −û
拟合:e_z(无旋子集,分速度档查依赖)+ 1D 搜索 μ 最小化 v_t' 残差;
粘/滑分区与残差留档(残差 = 对 MuJoCo 原始接触的标定残差,§11 M2 <5% 口径)。
用法(conda sim):python python/tools/ground_calib.py [model.mjcf]
"""
import sys

import mujoco
import numpy as np

R = 0.033
RPM = 2 * np.pi / 60.0


def model_facts(m):
    """留档:worldbody 直接可碰撞 geom(排除对决策)、球/地面接触参数、球惯量。"""
    print("[model] worldbody 直接 geom(contype!=0):")
    for g in range(m.ngeom):
        if m.geom_bodyid[g] == 0 and m.geom_contype[g] != 0:
            name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or "?"
            print(f"  geom {g} '{name}' contype={m.geom_contype[g]} "
                  f"conaffinity={m.geom_conaffinity[g]} parent=world")
    for nm in ("ball_geom", "floor"):
        g = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, nm)
        if g >= 0:
            print(f"[model] {nm}: solref={m.geom_solref[g]} solimp={m.geom_solimp[g]} "
                  f"friction={m.geom_friction[g]} parent_body="
                  f"{mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[g])}")
    b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "ball")
    print(f"[model] ball: mass={m.body_mass[b]:.4f} inertia(主轴)={m.body_inertia[b]} "
          f"(实心球 2/5·mR² = {0.4 * m.body_mass[b] * R * R:.6f})")


def one_drop(m, d, ball_q, ball_v, pos, v0, w0):
    """单发落球:返回 (v_in, w_in, v_out, w_out) 或 None(1.2s 内未分离)。
    接触判定走 ncon(球 geom 参与的接触对),入射 = 首触前一步,
    出射 = 分离且 vz>0 的首步(软接触渐分离,取无接触力的首帧为渐近出射)。"""
    mujoco.mj_resetData(m, d)
    mujoco.mj_resetDataKeyframe(m, d, 0)
    for i in range(3):
        d.qpos[ball_q + i] = pos[i]
    d.qpos[ball_q + 3] = 1.0
    d.qpos[ball_q + 4] = d.qpos[ball_q + 5] = d.qpos[ball_q + 6] = 0.0
    for i in range(3):
        d.qvel[ball_v + i] = v0[i]
        d.qvel[ball_v + 3 + i] = w0[i]
    bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "ball")
    bgid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "ball_geom")
    area = np.pi * R * R
    kd = 0.5 * 1.225 * area * 0.55          # 拖曳力系数(与 app 真值侧一致,N)
    v_in = w_in = v_out = w_out = None
    seen = False
    for _ in range(1200):
        bv = d.qvel[ball_v:ball_v + 3]
        vm = np.linalg.norm(bv)
        d.xfrc_applied[bid, :3] = -kd * vm * bv   # python 绑定为 (nbody,6)
        d.xfrc_applied[bid, 3:] = 0
        touching = any((c.geom1 == bgid or c.geom2 == bgid)
                       for c in d.contact[:d.ncon])
        if touching and not seen:
            seen = True
            v_in = bv.copy()
            w_in = d.qvel[ball_v + 3:ball_v + 6].copy()
        if seen and not touching and d.qvel[ball_v + 2] > 0:
            v_out = d.qvel[ball_v:ball_v + 3].copy()
            w_out = d.qvel[ball_v + 3:ball_v + 6].copy()
            return v_in, w_in, v_out, w_out
        mujoco.mj_step(m, d)
    return None


def predict_coulomb(v, w, e_z, mu):
    """刚体 Coulomb+stick 出射预测(v_in, w_in → v_out, w_out, 分区)。
    Δω = I⁻¹·(r_c × P),I = 2/5·mR² → 系数 5/(2R);r_c × P = (R·Py, −R·Px, 0)。"""
    vin_z = abs(v[2])
    v = v.copy()
    w = w.copy()
    v[2] = -e_z * v[2]
    u = np.array([v[0] - R * w[1], v[1] + R * w[0]])
    un = np.linalg.norm(u)
    jn = (1 + e_z) * vin_z                      # 法向冲量(单位质量)
    p_stick = -(2.0 / 7.0) * u
    if un > 1e-9 and np.linalg.norm(p_stick) <= mu * jn:
        p = p_stick                              # 粘滞:u'≈0
        mode = "stick"
    elif un > 1e-9:
        p = -mu * jn * u / un                    # 滑动
        mode = "slide"
    else:
        p = np.zeros(2)
        mode = "rest"
    v[0] += p[0]
    v[1] += p[1]
    w[0] += 5.0 / (2 * R) * (R * p[1])
    w[1] += 5.0 / (2 * R) * (-R * p[0])
    return v, w, mode


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "models/generated/x4_tennis_demo.mjcf"
    m = mujoco.MjModel.from_xml_path(path)
    d = mujoco.MjData(m)
    model_facts(m)
    ball_q = m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "ball_free")]
    ball_v = m.jnt_dofadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "ball_free")]

    vzs = [2, 3, 4, 5, 6, 7]
    vts = [0.0, 2.0, 4.0]
    spins = [(0.0, 0.0)]                       # (ωy, ωz) rad/s;首项 = 无旋(e_z 子集)
    for s in (1000, -1000, 2000):
        spins.append((s * RPM, 0.0))
    for s in (1000, -1000):
        spins.append((0.0, s * RPM))
    pos = np.array([2.0, 1.0, 1.2])

    runs = []
    for vz in vzs:
        for vt in vts:
            for (wy, wz) in spins:
                v0 = np.array([vt, 0.0, -vz])
                w0 = np.array([0.0, wy, wz])
                r = one_drop(m, d, ball_q, ball_v, pos, v0, w0)
                if r is None:
                    print(f"  [warn] vz={vz} vt={vt} wy={wy/RPM:+.0f}rpm "
                          f"wz={wz/RPM:+.0f}rpm 未分离,弃")
                    continue
                runs.append(dict(vz0=vz, vt0=vt, wy0=wy, wz0=wz, r=r))
    print(f"[scan] 有效样本 {len(runs)} / {len(vzs)*len(vts)*len(spins)}")

    # ---- e_z:无旋子集(分速度档查依赖)----
    nos = [r for r in runs if abs(r["wy0"]) < 1 and abs(r["wz0"]) < 1]
    print("\n[e_z] 无旋子集:")
    for vz in vzs:
        sub = [r for r in nos if r["vz0"] == vz]
        if not sub:
            continue
        ez = np.mean([r["r"][2][2] / abs(r["r"][0][2]) for r in sub])
        print(f"  |vz_in|≈{np.mean([abs(r['r'][0][2]) for r in sub]):.2f} → "
              f"e_z={ez:.4f} (n={len(sub)})")
    ez_raw = np.mean([r["r"][2][2] / abs(r["r"][0][2]) for r in nos])
    print(f"  合并 e_z(原始求解器实测)= {ez_raw:.4f}")

    # ---- μ:1D 搜索,仅无旋子集(2026-10-08 用户定向:平抽球先行,
    #      旋转耦合后置——含旋样本只作影响量留档,不进拟合)----
    def tangential_resid(mu, ez, subset):
        s2, modes = 0.0, {}
        for r in subset:
            vp, _, mode = predict_coulomb(r["r"][0], r["r"][1], ez, mu)
            s2 += (vp[0] - r["r"][2][0]) ** 2 + (vp[1] - r["r"][2][1]) ** 2
            modes[mode] = modes.get(mode, 0) + 1
        return s2 / len(subset), modes

    grid = np.arange(0.02, 1.01, 0.01)
    resid = [tangential_resid(mu, ez_raw, nos)[0] for mu in grid]
    mu_fit = float(grid[int(np.argmin(resid))])
    _, modes = tangential_resid(mu_fit, ez_raw, nos)
    print(f"\n[μ] Coulomb+stick 拟合(无旋子集,e_z={ez_raw:.3f}):μ = {mu_fit:.2f} "
          f"(切向 RMSE={np.sqrt(min(resid)):.3f} m/s, 分区 {modes})")

    # ---- 残差明细(代表性样本:切向 + ωy)----
    print("\n[residual] 代表性样本(平抽=无旋行;含旧行留档旋转影响量):")
    flat = [r for r in nos if r["vt0"] > 0.1]
    for r in flat + [x for x in runs if abs(x["wy0"]) > 1 or abs(x["wz0"]) > 1][:6]:
        v_in, w_in, v_out, w_out = r["r"]
        vp, wp, mode = predict_coulomb(v_in, w_in, ez_raw, mu_fit)
        print(f"  vz={abs(v_in[2]):.2f} vt=({v_in[0]:+.2f},{v_in[1]:+.2f}) "
              f"ωy={w_in[1]/RPM:+5.0f} ωz={w_in[2]/RPM:+5.0f}rpm | "
              f"vt'实测=({v_out[0]:+.2f},{v_out[1]:+.2f}) 预测=({vp[0]:+.2f},{vp[1]:+.2f}) "
              f"[{mode}] ωy'实测={w_out[1]/RPM:+5.0f} 预测={wp[1]/RPM:+5.0f}rpm")

    # ---- 旋转影响量留档(平抽先行,旋转模块后置的证据)----
    ts = [r for r in runs if abs(r["wy0"]) > 1500 * RPM and r["vt0"] < 0.3]
    if ts:
        dvy = np.mean([abs(r["r"][2][0] - r["r"][0][0]) for r in ts])
        print(f"\n[spin-impact] 纯旋入射(vt≈0, |ωy|≈2000rpm)反弹切向速度增量 "
              f"≈{dvy:.2f} m/s —— 旋转耦合量级留档(后置模块)")

    nz = [abs(r["r"][2][2] - ez_raw * abs(r["r"][0][2])) for r in nos]
    vin_z_mean = np.mean([abs(r["r"][0][2]) for r in nos])
    print(f"\n[normal] 无旋法向残差 mean={np.mean(nz):.3f} "
          f"p95={np.percentile(nz, 95):.3f} m/s(相对 {100*np.mean(nz)/vin_z_mean:.1f}%)")
    print(f"\n[verdict] 原始求解器 e_z≈{ez_raw:.3f} ≈ 死球(solref 0.02/1 临界阻尼,"
          f"与网球 0.73 相差 8 倍)→ 不拟合计入真值;解析冲量按已知模型替换:"
          f"e_z=0.73(ITF 文献初值,待实机替换), μ={mu_fit:.2f}(本台架无旋拟合初值)")


if __name__ == "__main__":
    main()
