#!/usr/bin/env python3
"""击球演示 GUI:passive viewer 从共享状态文件同步 C++ 仿真进程(60Hz)。
用法: DISPLAY=:1 python demo_viewer.py   (先启动本程序,再运行 ./build/demo_hit)
"""
import struct, sys, time
import mujoco
import mujoco.viewer

MODEL = "/home/ubuntu/sckd/sim/sckd_sim/models/generated/x4_tennis_demo.xml"
STATE = "/tmp/sckd_demo_state.bin"

def main():
    m = mujoco.MjModel.from_xml_path(MODEL)
    d = mujoco.MjData(m)
    last = -1
    with mujoco.viewer.launch_passive(m, d) as v:
        while v.is_running():
            try:
                raw = open(STATE, "rb").read()
                if len(raw) >= 16:
                    seq, nq, nv = struct.unpack("<Qii", raw[:16])
                    need = 16 + 8 * (nq + nv)
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
