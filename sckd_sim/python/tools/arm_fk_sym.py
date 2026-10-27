#!/usr/bin/env python3
"""臂符号 FK(方案 B v0,2026-10-07):MJCF 运动链 → CasADi 符号函数。

从 MJCF 读 racket_center 所在运动链(含 X4_950 的固定 Link6),
按 MuJoCo 变换语义(Translate(body_pos)·Rot(body_quat)
·Translate(jnt_pos)·Rot(jnt_axis, q)·Translate(−jnt_pos))累积合成:
  f(q5)        拍心位置(底盘 base_link 原点系,yaw=0)
  R(q5)        racket_center 姿态矩阵
  normal(q5)   racket_face 盒的正薄轴(网球局部 Y,X4_950 局部 Z)
自测:随机 q 对拍 MuJoCo 运动学(不步进),位姿、法向和 Jacobian 误差 <1e-9。
调用方:wholebody_ocp.py(方案 B v0 整机 OCP)导入 build()/selftest()。
用法(conda sim):python python/tools/arm_fk_sym.py [model.mjcf]
"""
import sys

import casadi as ca
import mujoco
import numpy as np


def _rot_axis(axis, q):
    """罗德里格斯旋转(casadi 符号;axis 单位向量)。"""
    x, y, z = axis
    c, s = ca.cos(q), ca.sin(q)
    C = 1 - c
    return ca.vertcat(
        ca.horzcat(c + x * x * C, x * y * C - z * s, x * z * C + y * s),
        ca.horzcat(y * x * C + z * s, c + y * y * C, y * z * C - x * s),
        ca.horzcat(z * x * C - y * s, z * y * C + x * s, c + z * z * C))


def _rot_quat(w):
    """四元数→旋转矩阵(w 单位;MJCF [w,x,y,z])。"""
    a, b, c, d = w[0], w[1], w[2], w[3]
    return ca.vertcat(
        ca.horzcat(a * a + b * b - c * c - d * d, 2 * (b * c - a * d), 2 * (b * d + a * c)),
        ca.horzcat(2 * (b * c + a * d), a * a - b * b + c * c - d * d, 2 * (c * d - a * b)),
        ca.horzcat(2 * (b * d - a * c), 2 * (c * d + a * b), a * a - b * b - c * c + d * d))


def build(model_path):
    """返回 q、f_pos、f_R、f_normal 和五个活动关节名。"""
    m = mujoco.MjModel.from_xml_path(model_path)
    sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, 'racket_center')
    gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, 'racket_face')
    assert sid >= 0 and gid >= 0, '缺少 racket_center / racket_face'
    assert m.geom_type[gid] == mujoco.mjtGeom.mjGEOM_BOX
    assert m.geom_bodyid[gid] == m.site_bodyid[sid], '拍心和拍面须属于同一 body'
    # 运动链:site body → 向上收集到 base_link(含),world 不含
    chain = []
    b = m.site_bodyid[sid]
    while b != 0 and mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) != 'base_link':
        chain.append(b)
        b = m.body_parentid[b]
    assert b != 0, 'racket_center 链上未找到 base_link'
    chain.append(b)           # base_link(其关节为底盘自由关节,不入臂链)
    chain.reverse()           # [base_link, Link1..Link5]

    q = ca.SX.sym('q', 5)
    p = ca.SX.zeros(3)        # base_link 原点系
    R = ca.SX.eye(3)
    qi = 0
    jnt_names = []
    for b in chain[1:]:       # base_link 之后为臂
        p = p + R @ m.body_pos[b]
        R = R @ _rot_quat(m.body_quat[b])
        if m.body_jntnum[b] == 0:  # 固定 Link6 保留位姿,不占 q 槽位
            continue
        # 关节铰链(锚点旋转;锚点均为 0,保守起见仍按一般式)
        assert m.body_jntnum[b] == 1 and m.jnt_type[m.body_jntadr[b]] == mujoco.mjtJoint.mjJNT_HINGE
        j = m.body_jntadr[b]
        name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j)
        assert name == f'Joint{qi + 1}', '臂关节顺序须为 Joint1..Joint5'
        jnt_names.append(name)
        anchor, axis = m.jnt_pos[j], m.jnt_axis[j]
        Rj = _rot_axis(axis, q[qi])
        p = p + R @ (ca.SX(anchor) - Rj @ ca.SX(anchor))
        R = R @ Rj
        qi += 1
    assert qi == 5
    face_axis = int(np.argmin(m.geom_size[gid]))
    normal = (R @ _rot_quat(m.geom_quat[gid]))[:, face_axis]
    p = p + R @ ca.SX(m.site_pos[sid])
    R = R @ _rot_quat(m.site_quat[sid])     # site 变换(拍心)
    return {
        'q': q,
        'f_pos': ca.Function('f_pos', [q], [p]),
        'f_R': ca.Function('f_R', [q], [R]),
        'f_normal': ca.Function('f_normal', [q], [normal]),
        'jnt_names': jnt_names,
    }


def base_height(model_path):
    """默认站立 keyframe 下 base_link 世界位置(整机 OCP 的底盘高度先验)。"""
    m = mujoco.MjModel.from_xml_path(model_path)
    d = mujoco.MjData(m)
    home = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, 'home')
    if home >= 0:
        mujoco.mj_resetDataKeyframe(m, d, home)
    mujoco.mj_kinematics(m, d)
    b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, 'base_link')
    assert b >= 0, '缺少 base_link'
    return d.xpos[b].copy()


def selftest(model_path):
    m = mujoco.MjModel.from_xml_path(model_path)
    d = mujoco.MjData(m)
    fk = build(model_path)
    sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, 'racket_center')
    joints = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name)
              for name in fk['jnt_names']]
    qadr, vadr = m.jnt_qposadr[joints], m.jnt_dofadr[joints]
    gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, 'racket_face')
    face_axis = int(np.argmin(m.geom_size[gid]))
    f_J = ca.Function('f_J', [fk['q']], [ca.jacobian(fk['f_pos'](fk['q']), fk['q'])])
    jac = np.zeros((3, m.nv))
    home = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, 'home')
    if home >= 0:
        mujoco.mj_resetDataKeyframe(m, d, home)
    base_world = base_height(model_path)
    rng = np.random.default_rng(7)
    pmax = rmax = nmax = jmax = 0.0
    for _ in range(50):
        qv = rng.uniform(-1.5, 1.5, 5)  # 仅数值几何采样,不是硬件行程
        for i, a in enumerate(qadr):
            d.qpos[a] = qv[i]
        mujoco.mj_kinematics(m, d)
        mujoco.mj_comPos(m, d)
        mujoco.mj_jacSite(m, d, jac, None, sid)
        p_sym = np.array(fk['f_pos'](qv)).reshape(3) + base_world
        R_sym = np.array(fk['f_R'](qv)).reshape(3, 3)
        R_mj = d.site_xmat[sid].reshape(3, 3)
        pmax = max(pmax, float(np.abs(p_sym - d.site_xpos[sid]).max()))
        rmax = max(rmax, float(np.abs(R_sym - R_mj).max()))
        n_sym = np.array(fk['f_normal'](qv)).reshape(3)
        nmax = max(nmax, float(np.abs(n_sym - d.geom_xmat[gid].reshape(3, 3)[:, face_axis]).max()))
        jmax = max(jmax, float(np.abs(np.array(f_J(qv)) - jac[:, vadr]).max()))
    print(f'[fk] 几何自测 50 随机 q:pos {pmax:.2e}, R {rmax:.2e}, normal {nmax:.2e}, J {jmax:.2e}')
    assert max(pmax, rmax, nmax, jmax) < 1e-9, '符号几何与 MuJoCo 不一致'
    print(f"[fk] base_link 世界位置(默认站立):{np.round(base_world, 4)}")
    return base_world


if __name__ == '__main__':
    path = sys.argv[1] if len(sys.argv) > 1 else 'models/generated/x4_tennis_demo.mjcf'
    selftest(path)
