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
    # 待机显示位 = 实机 sckd-robot home(J2 抬臂 35°,与 m2_intercept/m2_demo
    # kStandby 同源)+ 整车复位(转向轮全 0、球移出视野)。
    # 状态机(2026-10-09):启动=待机;有新鲜写者=实时跟随;写者退出/无写者
    # =自动回待机 —— 旧逻辑冻结在末帧(随挥姿态+跑动中打偏的转向轮),
    # "很多关节没在待机位"即此(用户实测)
    STANDBY = {"Joint1": -0.0244, "Joint2": 0.6109, "Joint3": -0.3927,
               "Joint4": -1.5359, "Joint5": 0.0}

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
