#!/usr/bin/env python3
"""Ubuntu离线检验SDK模型角→MuJoCo显示映射；仅调用Forward，不启动设备。

生成纯FK共享库、133组对照与verification.json；不会修改XML、SDK或映射。
"""
from __future__ import annotations

import argparse
import configparser
import ctypes
import hashlib
import itertools
import json
from pathlib import Path
import subprocess
import sys

import mujoco
import numpy as np


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rotation(axis: str, angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    if axis == "y":
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sdk", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--mapping", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if not sys.platform.startswith("linux"):
        parser.error("按项目约定，仅可在Ubuntu执行")
    sdk, model_path = args.sdk.resolve(), args.model.resolve()
    mapping = json.loads(args.mapping.read_text())
    if digest(model_path) != mapping["source_model_sha256"]:
        parser.error("模型摘要与显示映射证据不一致")
    library = sdk / "lib/libarm.so.0.1.0"
    if digest(library) != mapping["source_sdk_libarm_sha256"]:
        parser.error("SDK libarm摘要与显示映射证据不一致")
    config = configparser.ConfigParser(inline_comment_prefixes=("#", ";"), strict=False)
    config.read(sdk / "config/robot_8w4s.ini")
    dims = np.array([float(x) for x in config["Arm"]["dimens"].split(",")], dtype=np.float32)
    np.testing.assert_allclose(dims, mapping["sdk_dimensions_d1_a2_d4_d5_m"], atol=1e-7)
    args.output.mkdir(parents=True, exist_ok=False)
    source = args.output / "sdk_forward_only.cpp"
    source.write_text('''#include "arm/kinematics.h"
extern "C" void sdk_forward_only(const float* q, const float* dims, float* p, float* r) {
  const auto result=sckd::arm::Forward(q,dims);
  for(int i=0;i<3;++i) { p[i]=result.pos[i]; for(int j=0;j<3;++j) r[i*3+j]=result.rot(i,j); }
}
''')
    bridge = args.output.resolve() / "lib_sdk_forward_only.so"
    subprocess.run(["g++", "-std=c++17", "-O2", "-shared", "-fPIC", "-I" + str(sdk / "include"),
                    "-I/usr/include/eigen3", str(source), "-L" + str(sdk / "lib"),
                    "-Wl,-rpath," + str(sdk / "lib"), "-larm", "-o", str(bridge)], check=True)
    sdk_library = ctypes.CDLL(str(bridge))
    forward = sdk_library.sdk_forward_only
    forward.argtypes = [np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS")] * 4
    forward.restype = None
    model = mujoco.MjModel.from_xml_path(str(model_path))
    addresses = [int(model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT,
                                                     "Joint" + str(i))]) for i in range(1, 6)]
    shoulder = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "Link2")
    tip = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "racket_center")
    sign, offset = np.array(mapping["sign"]), np.array(mapping["offset_rad"])
    permutation = np.array(mapping["permutation"]) - 1
    tool_rotation = np.array(mapping["sdk_tool_frame_rotation"])
    home = np.array([-1.4, 35, -22.5, -88, 0.])
    observed = np.array([-1.664, 13.338, -27.066, -88.442, -2.725])
    poses = [("zero", np.zeros(5)), ("home", home), ("observed_feedback", observed)]
    for name, base in list(poses):
        for joint, direction in itertools.product(range(5), (-1, 1)):
            q = base.copy()
            q[joint] += direction * 10
            poses.append((f"{name}_J{joint + 1}_{direction:+d}0deg", q))
    rng = np.random.default_rng(20261023)
    for i in range(100):
        poses.append((f"random_{i}", rng.uniform([-90, -40, -80, -180, -80], [90, 65, 25, 180, 80])))
    rows = []
    for name, degrees in poses:
        q = np.radians(degrees)
        p, r = np.zeros(3, np.float32), np.zeros(9, np.float32)
        forward(q.astype(np.float32), dims, p, r)
        sdk_relative = p.astype(float) - [0, 0, float(dims[0])]
        data = mujoco.MjData(model)
        data.qpos[:] = model.qpos0
        data.qpos[addresses] = sign * q[permutation] + offset
        before = data.qpos.copy()
        mujoco.mj_forward(model, data)
        np.testing.assert_array_equal(data.qpos, before)
        relative = data.site_xpos[tip] - data.xpos[shoulder]
        orientation_error = float(np.max(np.abs(data.site_xmat[tip].reshape(3, 3) @ tool_rotation - r.reshape(3, 3))))
        # 纯数学链把尺寸设成SDK值，仅用于归因CAD残差；没有改MjModel或XML。
        r1 = rotation("z", q[0])
        r2 = r1 @ rotation("y", -q[1])
        r3 = r2 @ rotation("y", -q[2])
        r4 = r3 @ rotation("z", -q[3] - np.pi / 2)
        ideal = r2 @ [float(dims[1]), 0, 0] + r3 @ [0, 0, -float(dims[2])] + r4 @ [float(dims[3]), 0, 0]
        rows.append({"name": name, "sdk_q_deg": degrees.tolist(), "mujoco_q_deg": np.degrees(data.qpos[addresses]).tolist(),
                     "sdk_tip": p.tolist(), "mujoco_tip": data.site_xpos[tip].tolist(),
                     "orientation_matrix_error": orientation_error,
                     "position_error_same_shoulder_m": float(np.linalg.norm(relative - sdk_relative)),
                     "idealized_chain_position_error_m": float(np.linalg.norm(ideal - sdk_relative))})
    max_orientation = max(row["orientation_matrix_error"] for row in rows)
    max_position = max(row["position_error_same_shoulder_m"] for row in rows)
    max_idealized = max(row["idealized_chain_position_error_m"] for row in rows)
    passed = max_orientation < 1e-6 and max_position < .12 and max_idealized < 1e-6
    report = {"passed": passed, "platform": "Ubuntu", "hardware_started": False,
              "sdk_functions_called": ["sckd::arm::Forward"], "sample_count": len(rows),
              "model_sha256": digest(model_path), "sdk_libarm_sha256": digest(library),
              "max_orientation_matrix_error": max_orientation,
              "max_position_error_same_shoulder_m": max_position,
              "max_idealized_chain_position_error_m": max_idealized,
              "full_position_alignment_validated": False, "physical_calibration_complete": False,
              "samples": rows}
    (args.output / "verification.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({key: value for key, value in report.items() if key != "samples"}, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
