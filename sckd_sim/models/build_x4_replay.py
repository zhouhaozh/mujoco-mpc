#!/usr/bin/env python3
"""Build an isolated X4_950 reference-only ball replay scene."""
import argparse
import json
import os
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco


ROOT = Path(__file__).resolve().parents[1]
GENERATED = ROOT / "models" / "generated"
SOURCE = GENERATED / "x4_950_scene.mjcf"
OUT = GENERATED / "x4_950_replay.mjcf"


def _manifest_path(output: Path) -> Path:
    return output.with_name(f"{output.stem}_manifest.json")


def build_replay(source: Path, output: Path) -> dict:
    """Copy one robot scene to ``output`` and add the measured replay ball."""
    source = Path(source).resolve()
    output = Path(output).resolve()
    if source == output:
        raise ValueError("replay output must differ from its source scene")
    if not source.is_file():
        raise FileNotFoundError(f"replay source scene not found: {source}")
    output.parent.mkdir(parents=True, exist_ok=True)

    tree = ET.parse(source)
    root = tree.getroot()
    compiler = root.find("compiler")
    if compiler is not None and compiler.get("meshdir"):
        meshdir = (source.parent / compiler.get("meshdir")).resolve()
        compiler.set("meshdir", os.path.relpath(meshdir, output.parent))

    world = root.find("worldbody")
    if world is None:
        raise RuntimeError(f"{source} has no worldbody")
    joints = root.findall(".//freejoint[@name='ball_free']")
    geoms = root.findall(".//geom[@name='ball_geom']")
    if len(joints) != len(geoms) or len(joints) > 1:
        raise RuntimeError("replay scene must contain either one complete ball or no ball")
    already_replay = bool(joints)
    if already_replay:
        ball_geom = geoms[0]
        ball_geom.set("contype", "0")
        ball_geom.set("conaffinity", "0")
    else:
        ball = ET.SubElement(world, "body", {"name": "ball", "gravcomp": "1"})
        ET.SubElement(ball, "freejoint", {"name": "ball_free"})
        ET.SubElement(ball, "geom", {
            "name": "ball_geom", "type": "sphere", "size": "0.015", "mass": "0.005",
            "contype": "0", "conaffinity": "0", "rgba": "0.95 0.95 0.95 1",
        })

    racket = root.find(".//geom[@name='racket_face']")
    if racket is None:
        raise RuntimeError(f"{source} has no racket_face geom")
    rgba = [float(value) for value in racket.get("rgba", "0.5 0.5 0.5 1").split()]
    if len(rgba) != 4:
        raise RuntimeError("racket_face rgba must have four values")
    rgba[3] = 0.0
    racket.set("rgba", " ".join(f"{value:g}" for value in rgba))

    keyframe = root.find("keyframe")
    home = keyframe.find("key[@name='home']") if keyframe is not None else None
    if home is None:
        raise RuntimeError(f"{source} has no home keyframe")
    qpos = [float(value) for value in home.get("qpos", "").split()]
    if already_replay:
        if len(qpos) != 31:
            raise RuntimeError(f"existing replay must have 31 home qpos values, got {len(qpos)}")
    else:
        if len(qpos) != 24:
            raise RuntimeError(f"robot scene must have 24 home qpos values, got {len(qpos)}")
        qpos += [0.0, 0.0, 1.5, 1.0, 0.0, 0.0, 0.0]
        home.set("qpos", " ".join(str(value) for value in qpos))
    home.set("qvel", " ".join("0.0" for _ in range(29)))

    ET.indent(tree, space="  ")
    output.write_text("<?xml version='1.0' encoding='utf-8'?>\n" +
                      ET.tostring(root, encoding="unicode") + "\n")
    xml_output = output.with_suffix(".xml") if output.suffix == ".mjcf" else None
    if xml_output is not None:
        xml_output.write_text(output.read_text())

    model = mujoco.MjModel.from_xml_path(str(output))
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)
    ball_joint = model.joint("ball_free").id
    ball_geom = model.geom("ball_geom").id
    if model.geom_contype[ball_geom] or model.geom_conaffinity[ball_geom]:
        raise RuntimeError("measured replay ball collision must be disabled")
    manifest = {
        "variant": "x4_950_replay",
        "purpose": "reference_only_ball_trajectory_replay",
        "source_scene": str(source), "scene": str(output),
        "xml_scene": str(xml_output) if xml_output is not None else None,
        "ball": {
            "body": "ball", "joint": "ball_free", "geom": "ball_geom",
            "shape": "sphere_proxy", "radius_m": 0.015,
            "mass_kg_placeholder": 0.005, "collision_enabled": False,
            "gravity_compensation": True,
            "note": "Replay app writes qpos/qvel each frame; mass is not a flight-physics claim.",
        },
        "dimensions": {
            "nq": int(model.nq), "nv": int(model.nv), "nu": int(model.nu),
            "robot_qpos": 24, "ball_qpos": 7, "robot_qvel": 23, "ball_qvel": 6,
        },
        "static_check": {
            "reset_keyframe": 0, "forward_called": True, "mj_step_called": False,
            "ball_joint_id": int(ball_joint), "ball_geom_id": int(ball_geom),
            "input_already_replay": already_replay,
        },
    }
    _manifest_path(output).write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--output", type=Path, default=OUT)
    args = parser.parse_args()
    print(json.dumps(build_replay(args.source, args.output), ensure_ascii=False))


if __name__ == "__main__":
    main()
