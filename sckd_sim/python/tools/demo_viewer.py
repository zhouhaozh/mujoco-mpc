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
    # 待机显示位 = 实机 sckd-robot home(robot_8w4s.ini home_deg=-1.4,35,
    # -22.5,-88,0°,§23.1 三锚标定)映射到 sim 关节系:J2/J3/J4 符号翻转
    # (实机模型系正=抬臂,sim 系负=抬臂;原 m2_demo 直抄未映射 → J2 下垂+
    # 拍甩右侧,2026-10-10 实机照片三视交叉定位:抬臂/小臂竖直/拍举高,
    # 拍心(+0.52,−0.58,+1.27))。状态机:启动=待机;有写者=实时;写者退出=
    # 回待机
    STANDBY = {"Joint1": 0.0244, "Joint2": -0.6109, "Joint3": 0.3927,
               "Joint4": 1.5359, "Joint5": 0.0}

    def show_standby():
        mujoco.mj_resetData(m, d)
        for name, q in STANDBY.items():
            jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name)
            if jid >= 0:
                d.qpos[m.jnt_qposadr[jid]] = q
        mujoco.mj_forward(m, d)
        v.sync()

    last = -1
    live = False
    with mujoco.viewer.launch_passive(m, d) as v:
        show_standby()
        while v.is_running():
            fresh = False
            raw = b""
            try:
                raw = open(STATE, "rb").read()
                fresh = time.time() - os.path.getmtime(STATE) <= 1.0
            except FileNotFoundError:
                pass
            if fresh and len(raw) >= 16:
                seq, nq, nv = struct.unpack("<Qii", raw[:16])
                need = 16 + 8 * (nq + nv)
                if len(raw) >= need and seq != last:
                    d.qpos[:] = struct.unpack(f"<{nq}d", raw[16:16 + 8 * nq])
                    d.qvel[:] = struct.unpack(f"<{nv}d", raw[16 + 8 * nq:need])
                    mujoco.mj_forward(m, d)
                    v.sync()
                    last = seq
                live = True
            elif live:   # 写者退出(文件删/过期)→ 回待机显示
                show_standby()
                live = False
                last = -1
            time.sleep(0.016)
    return 0

if __name__ == "__main__":
    sys.exit(main())
