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
    # 待机显示位(2026-10-10 定稿):sim 零位构型=大臂水平/小臂竖直/拍水平
    # 后折,与实机 sckd-robot 待机(电机系 0,−35,−20,−90,0 + joint_sign/
    # offset)几何同构。J2=−35° 抬大臂;J3=−35°(净转角 0,小臂保持模型
    # 自带近竖直);J4=0(拍水平回折)。FK 验证:大臂仰 43.6°/小臂 78.8°/
    # 拍仰角 0.0°,拍心(−0.06,0,1.26)。前两版教训:①直抄实机模型系值
    # (J2 正=下垂+拍甩右)②纯符号翻转(J3/J4 不对)——根因=电机零位≠
    # 模型零位(ini joint_offset)+两模型连杆构型不同,不可代数映射,按
    # 用户三几何规约直接解
    STANDBY = {"Joint1": 0.0, "Joint2": -0.6109, "Joint3": -0.6109,
               "Joint4": 0.0, "Joint5": 0.0}

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
