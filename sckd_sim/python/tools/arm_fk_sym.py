#!/usr/bin/env python3
"""臂符号 FK(方案 B v0,2026-10-07):MJCF 运动链 → CasADi 符号函数。

从 models/generated/x4_tennis_demo.mjcf 读 racket_center 所在运动链
(base_link→Link1..Link5),按 MuJoCo 变换语义(Translate(body_pos)·Rot(body_quat)
·Translate(jnt_pos)·Rot(jnt_axis, q)·Translate(−jnt_pos))累积合成:
  f(q5)        拍心位置(底盘 base_link 原点系,yaw=0)
  R(q5)        拍姿态矩阵;拍面法向 = R·[0,1,0](与 m2_intercept 法向分解同约定)
自测:随机 q 对拍 mj_forward(同模型),位差 <1e-9、姿态差 <1e-9。
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
    """返回 dict:q(casadi 符号), f_pos(q)->3, f_R(q)->3x3, jnt 名单。"""
    m = mujoco.MjModel.from_xml_path(model_path)
    sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, 'racket_center')
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
    for b in chain[1:]:       # base_link 之后为臂
        R = R @ _rot_quat(m.body_quat[b])
        p = p + R @ m.body_pos[b]
        # 关节铰链(锚点旋转;锚点均为 0,保守起见仍按一般式)
        assert m.body_jntnum[b] == 1 and m.jnt_type[m.body_jntadr[b]] == mujoco.mjtJoint.mjJNT_HINGE
        j = m.body_jntadr[b]
        anchor, axis = m.jnt_pos[j], m.jnt_axis[j]
        Rj = _rot_axis(axis, q[qi])
        p = p + R @ (ca.SX(anchor) - Rj @ ca.SX(anchor))
        R = R @ Rj
        qi += 1
    assert qi == 5
    R = R @ _rot_quat(m.site_quat[sid])     # site 变换(拍心)
    p = p + R @ ca.SX(m.site_pos[sid])
    return {
        'q': q,
        'f_pos': ca.Function('f_pos', [q], [p]),
        'f_R': ca.Function('f_R', [q], [R]),
        'jnt_names': [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, m.body_jntadr[b])
                      for b in chain[1:]],
    }


def base_height(model_path):
    """默认站立 keyframe 下 base_link 世界位置(整机 OCP 的底盘高度先验)。"""
    m = mujoco.MjModel.from_xml_path(model_path)
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, 0)
    mujoco.mj_forward(m, d)
    sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, 'racket_center')
    b = m.site_bodyid[sid]
    while mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) != 'base_link':
        b = m.body_parentid[b]
    return d.xpos[b].copy()


def selftest(model_path):
    m = mujoco.MjModel.from_xml_path(model_path)
    d = mujoco.MjData(m)
    fk = build(model_path)
    sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, 'racket_center')
    qadr = []
    b = m.site_bodyid[sid]
    while mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) != 'base_link':
        j = m.body_jntadr[b]
        assert m.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE
        qadr.append(m.jnt_qposadr[j])
        b = m.body_parentid[b]
    qadr.reverse()
    base_world = base_height(model_path)
    rng = np.random.default_rng(7)
    pmax = rmax = 0.0
    for _ in range(50):
        qv = rng.uniform(-1.5, 1.5, 5)
        for i, a in enumerate(qadr):
            d.qpos[a] = qv[i]
        mujoco.mj_forward(m, d)
        p_sym = np.array(fk['f_pos'](qv)).reshape(3) + base_world
        R_sym = np.array(fk['f_R'](qv)).reshape(3, 3)
        R_mj = d.site_xmat[sid].reshape(3, 3)
        pmax = max(pmax, float(np.abs(p_sym - d.site_xpos[sid]).max()))
        rmax = max(rmax, float(np.abs(R_sym - R_mj).max()))
    print(f'[fk] 位姿自测 50 随机 q:pos max err {pmax:.2e}, R max err {rmax:.2e}')
    assert pmax < 1e-9 and rmax < 1e-9, '符号 FK 与 MuJoCo 不一致'
    print(f"[fk] base_link 世界位置(默认站立):{np.round(base_world, 4)}")
    return base_world


if __name__ == '__main__':
    path = sys.argv[1] if len(sys.argv) > 1 else 'models/generated/x4_tennis_demo.mjcf'
    selftest(path)
