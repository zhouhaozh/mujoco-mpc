#!/usr/bin/env python3
"""击球演示 GUI:passive viewer 从共享状态文件同步 C++ 仿真进程(60Hz)。
用法: DISPLAY=:1 python demo_viewer.py   (先启动本程序,再运行 ./build/demo_hit)
"""
import os, struct, sys, time
import mujoco
import mujoco.viewer

MODEL = "/home/ubuntu/sckd/sim/sckd_sim/models/generated/x4_tennis_demo.xml"
STATE = "/tmp/sckd_demo_state.bin"

def main():
    m = mujoco.MjModel.from_xml_path(MODEL)
    d = mujoco.MjData(m)
    # 初始姿态 = 实机 sckd-robot home 待机位(J2 抬臂 35°,与 m2_intercept/
    # m2_demo kStandby 同源):模型默认 qpos 是 J2 水平,无写者时 viewer 冻结
    # 在默认姿态上会被误读为"待机姿态不对"(2026-10-09 用户实测)
    STANDBY = {"Joint1": -0.0244, "Joint2": 0.6109, "Joint3": -0.3927,
               "Joint4": -1.5359, "Joint5": 0.0}
    for name, q in STANDBY.items():
        jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid >= 0:
            d.qpos[m.jnt_qposadr[jid]] = q
    mujoco.mj_forward(m, d)
    last = -1
    with mujoco.viewer.launch_passive(m, d) as v:
        while v.is_running():
            try:
                raw = open(STATE, "rb").read()
                if len(raw) >= 16:
                    seq, nq, nv = struct.unpack("<Qii", raw[:16])
                    need = 16 + 8 * (nq + nv)
                    # 新鲜度防护(2026-10-09):仅应用 1s 内的帧 —— 写者已退出时
                    # 冻结当前姿态,不回放残留状态文件(旧症状:viewer 一启动
                    # 就"已在挥拍"= 上次运行的残留帧)
                    if time.time() - os.path.getmtime(STATE) > 1.0:
                        time.sleep(0.016)
                        continue
                    if len(raw) >= need and seq != last:
                        d.qpos[:] = struct.unpack(f"<{nq}d", raw[16:16 + 8 * nq])
                        d.qvel[:] = struct.unpack(f"<{nv}d", raw[16 + 8 * nq:need])
                        mujoco.mj_forward(m, d)
                        v.sync()
                        last = seq
            except FileNotFoundError:
                pass
            time.sleep(0.016)
    return 0

if __name__ == "__main__":
    sys.exit(main())
