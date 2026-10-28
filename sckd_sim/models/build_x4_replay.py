#!/usr/bin/env python3
"""Build the X4_950 reference-only ball replay scene.

This deliberately does not run a simulation step: the replay application writes
the ball pose/velocity every frame.  The existing X4 scene remains untouched.
"""
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco


ROOT = Path(__file__).resolve().parents[1]
GENERATED = ROOT / "models" / "generated"
SOURCE = GENERATED / "x4_950_scene.mjcf"
OUT = GENERATED / "x4_950_replay.mjcf"
OUT_XML = GENERATED / "x4_950_replay.xml"
MANIFEST = GENERATED / "x4_950_replay_manifest.json"


def main() -> None:
    tree = ET.parse(SOURCE)
    root = tree.getroot()
    world = root.find("worldbody")
    if world is None:
        raise RuntimeError("x4_950_scene.mjcf has no worldbody")

    ball = ET.SubElement(world, "body", {
        "name": "ball",
        "gravcomp": "1",
    })
    ET.SubElement(ball, "freejoint", {"name": "ball_free"})
    ET.SubElement(ball, "geom", {
        "name": "ball_geom",
        "type": "sphere",
        "size": "0.015",
        "mass": "0.005",
        "contype": "0",
        "conaffinity": "0",
        "rgba": "0.95 0.95 0.95 1",
    })

    keyframe = root.find("keyframe")
    if keyframe is None:
        raise RuntimeError("x4_950_scene.mjcf has no keyframe")
    home = keyframe.find("key[@name='home']")
    if home is None:
        raise RuntimeError("x4_950_scene.mjcf has no home keyframe")
    robot_qpos = [float(x) for x in home.get("qpos", "").split()]
    if len(robot_qpos) != 24:
        raise RuntimeError(f"expected 24 robot qpos values, got {len(robot_qpos)}")
    # Ball state is appended after the 24 robot qpos values.  It is a harmless
    # reference pose; replay code overwrites it before consuming observations.
    ball_qpos = [0.0, 0.0, 1.5, 1.0, 0.0, 0.0, 0.0]
    home.set("qpos", " ".join(str(x) for x in robot_qpos + ball_qpos))
    home.set("qvel", " ".join("0.0" for _ in range(29)))

    ET.indent(tree, space="  ")
    text = ET.tostring(root, encoding="unicode")
    OUT.write_text("<?xml version='1.0' encoding='utf-8'?>\n" + text)
    OUT_XML.write_text(OUT.read_text())

    model = mujoco.MjModel.from_xml_path(str(OUT))
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)
    ball_joint = model.joint("ball_free").id
    ball_geom = model.geom("ball_geom").id
    manifest = {
        "variant": "x4_950_replay",
        "purpose": "reference_only_ball_trajectory_replay",
        "source_scene": str(SOURCE.relative_to(ROOT)),
        "ball": {
            "body": "ball",
            "joint": "ball_free",
            "geom": "ball_geom",
            "shape": "sphere_proxy",
            "radius_m": 0.015,
            "mass_kg_placeholder": 0.005,
            "collision_enabled": False,
            "gravity_compensation": True,
            "note": "Replay app writes qpos/qvel each frame; mass is not a flight-physics claim.",
        },
        "dimensions": {
            "nq": int(model.nq),
            "nv": int(model.nv),
            "nu": int(model.nu),
            "robot_qpos": 24,
            "ball_qpos": 7,
            "robot_qvel": 23,
            "ball_qvel": 6,
        },
        "static_check": {
            "reset_keyframe": 0,
            "forward_called": True,
            "mj_step_called": False,
            "ball_joint_id": int(ball_joint),
            "ball_geom_id": int(ball_geom),
        },
    }
    MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(manifest, ensure_ascii=False))


if __name__ == "__main__":
    main()
