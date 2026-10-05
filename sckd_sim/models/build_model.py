#!/usr/bin/env python3
"""x4_tennis 模型管线:URDF(+STL) -> MJCF + provenance manifest。

用法(conda sim 环境):
  /home/ubuntu/miniconda3/envs/sim/bin/python models/build_model.py [--variant x4_tennis]

管线(2026-10-05 v2):from_xml_path(URDF) -> mj_saveLastXML 保网格基线 -> XML 级编辑。
  (v1 的 mjSpec.to_xml 会丢 URDF 来源网格 geom —— 实测 nmesh=0,已弃用)

产物(models/generated/):
  x4_tennis_scene.mjcf      编译验证过的场景(机器人+地面+灯+执行器+keyframe)
  x4_tennis_manifest.json   变更清单(provenance):来源哈希、每项编辑及理由、关节/执行器映射
"""
import argparse
import hashlib
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco

MODELS = Path(__file__).resolve().parents[1] / "models"
GENERATED = MODELS / "generated"
CONTRACT_JOINTS = [  # 契约轴序(§4.2),与 cpp/core/types.h kAxisJointName 一致
    "FL_steer_joint", "FL_wheel_left_joint", "FL_wheel_right_joint",
    "FR_steer_joint", "FR_wheel_left_joint", "FR_wheel_right_joint",
    "RL_steer_joint", "RL_wheel_left_joint", "RL_wheel_right_joint",
    "RR_steer_joint", "RR_wheel_left_joint", "RR_wheel_right_joint",
    "Joint1", "Joint2", "Joint3", "Joint4", "Joint5",
]
WHEEL_BODIES = [
    ("FL_wheel_left_link", "left"), ("FL_wheel_right_link", "right"),
    ("FR_wheel_left_link", "left"), ("FR_wheel_right_link", "right"),
    ("RL_wheel_left_link", "left"), ("RL_wheel_right_link", "right"),
    ("RR_wheel_left_link", "left"), ("RR_wheel_right_link", "right"),
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def build(variant: str) -> int:
    cfg = json.loads((MODELS / "config" / f"{variant}.json").read_text())
    urdf = (MODELS / cfg["source_urdf"]).resolve()
    if not urdf.exists():
        print(f"[ERR] URDF not found: {urdf}", file=sys.stderr)
        return 1
    GENERATED.mkdir(parents=True, exist_ok=True)
    edits = []

    # --- 1) 预处理:注入 <mujoco><compiler .../>(与 STL 同目录,便于相对路径) ---
    tree = ET.parse(urdf)
    root = tree.getroot()
    if root.find("mujoco") is None:
        mje = ET.SubElement(root, "mujoco")
        comp = ET.SubElement(mje, "compiler")
        comp.set("angle", "radian")
        comp.set("discardvisual", "false")  # URDF 导入默认丢 visual geom,需显式关闭
        comp.set("fusestatic", "false")     # 否则 base_link(无关节)被融合进 worldbody
        edits.append({"edit": "inject_mujoco_compiler",
                      "reason": "angle=radian + discardvisual=false;不改源 URDF;临时副本与 STL 同目录"})
    pre = urdf.parent / f"_{variant}_pre.urdf"
    tree.write(pre, encoding="unicode", xml_declaration=True)

    # --- 2) 经典路径保网格:from_xml_path -> mj_saveLastXML ---
    mujoco.MjModel.from_xml_path(str(pre))
    baseline = GENERATED / f"_{variant}_baseline.mjcf"
    mujoco.mj_saveLastXML(str(baseline), mujoco.MjModel.from_xml_path(str(pre)))
    pre.unlink(missing_ok=True)
    edits.append({"edit": "urdf_to_mjcf_saveLastXML",
                  "reason": "网格 geom 保真(mjSpec.to_xml 实测丢 URDF 网格);inertial 原样保留(§3.2.1)"})

    # --- 3) XML 级编辑 ---
    tree = ET.parse(baseline)
    root = tree.getroot()

    # 3a) base_link 加 freejoint(浮动底盘)
    bl = None
    for body in root.iter("body"):
        if body.get("name") == "base_link":
            bl = body
            break
    if bl is None:
        print("[ERR] base_link not found", file=sys.stderr)
        return 1
    ET.SubElement(bl, "freejoint").set("name", "root")
    bl.set("pos", f"0 0 {cfg['keyframe_home']['base_z_m']}")  # qpos0=站立位(simulate 免 keyframe)
    edits.append({"edit": "add_freejoint_base_link", "reason": "URDF 无世界连接;底盘需 6 DOF 浮动"})

    # 3b) 网格 geom 仅视觉(contype=0);轮子加基元圆柱碰撞
    wc = cfg["wheel_collision"]
    n_mesh_geom = 0
    for g in root.iter("geom"):
        if g.get("mesh") is not None:
            g.set("contype", "0")
            g.set("conaffinity", "0")
            if g.get("group") == "0":      # URDF 碰撞代理(默认金黄)→ 藏到 group 3
                g.set("group", "3")
                g.set("rgba", "0.5 0.5 0.5 0")
            n_mesh_geom += 1
    bodies = {b.get("name"): b for b in root.iter("body")}
    for name, side in WHEEL_BODIES:
        g = ET.SubElement(bodies[name], "geom")
        g.set("name", f"col_{name}")
        g.set("type", "cylinder")
        g.set("size", f"{wc['radius_m']} {wc['half_width_m']}")
        g.set("pos", " ".join(str(v) for v in wc["center_in_wheel_frame"][side]))
        g.set("quat", "0.7071067811865476 0.7071067811865476 0 0")  # 圆柱 z -> 轮轴 y
        g.set("friction", " ".join(str(v) for v in wc["friction"]))
        g.set("contype", "1")
        g.set("conaffinity", "1")
        g.set("mass", "0")
    edits.append({"edit": "mesh_geoms_visual_only_and_wheel_collider",
                  "reason": f"{n_mesh_geom} 个网格 geom 仅视觉;轮用圆柱 r={wc['radius_m']} "
                            f"half={wc['half_width_m']}(STL bbox/径向 p99,§3.2.1);质量 0 不改 URDF 惯量"})

    # 3c) 地面 + 灯
    wb = root.find("worldbody")
    fl = ET.SubElement(wb, "geom")
    fl.set("name", "floor")
    fl.set("type", "plane")
    fl.set("size", "30 30 0.1")
    fl.set("friction", " ".join(str(v) for v in wc["friction"]))
    fl.set("contype", "1")
    fl.set("conaffinity", "1")
    lt = ET.SubElement(wb, "light")
    lt.set("pos", "0 0 4")
    lt.set("dir", "0 0 -1")
    lt.set("directional", "true")
    edits.append({"edit": "floor_plane_and_light",
                  "reason": "M0 平地;μ=1 名义值(F1),F3 轮地标定后替换(§9-3)"})

    # 3d) 执行器:17 电机,与关节同名,ctrlrange=±机构限矩
    jc = {j["joint"]: j for j in cfg["joints"]}
    asec = ET.SubElement(root, "actuator")
    for jn in CONTRACT_JOINTS:
        a = ET.SubElement(asec, "motor")
        a.set("name", jn)
        a.set("joint", jn)
        a.set("gear", "1")
        a.set("ctrllimited", "true")
        lim = jc[jn]["mechanism_limit_nm"]
        a.set("ctrlrange", f"{-lim} {lim}")
    edits.append({"edit": "add_actuators_17",
                  "reason": "gear=1(减速器输出端口径,设计 §3.3);ctrlrange=±机构限矩为二级保护,一级在电机模型限幅链"})

    # 3e) keyframe home + 步长
    base_z = cfg["keyframe_home"]["base_z_m"]
    qpos = [0.0] * (7 + 17)
    qpos[2] = base_z
    ksec = ET.SubElement(root, "keyframe")
    k = ET.SubElement(ksec, "key")
    k.set("name", "home")
    k.set("qpos", " ".join(str(v) for v in qpos))
    opt = root.find("option")
    if opt is None:
        opt = ET.SubElement(root, "option")
    opt.set("timestep", "0.001")
    edits.append({"edit": "keyframe_home_and_timestep",
                  "reason": f"base_z={base_z}(0.043−0.1689+轮半径);dt=1 ms(§1.2);关节全零=臂水平"})

    # 3f) meshdir 指向 source(产物自包含)并保存
    comp = root.find("compiler")
    if comp is None:
        comp = ET.SubElement(root, "compiler")
    comp.set("meshdir", f"../source/{urdf.parent.name}")
    out_xml = GENERATED / f"{variant}_scene.mjcf"
    tree.write(out_xml, encoding="unicode", xml_declaration=True)
    # simulate GUI 命令行不识别 .mjcf 扩展(按资源解码失败->空场景);输出同名 .xml 供 GUI
    xml_gui = GENERATED / f"{variant}_scene.xml"
    xml_gui.write_text(out_xml.read_text())
    baseline.unlink(missing_ok=True)

    # --- 4) 从产物重载自检 ---
    model = mujoco.MjModel.from_xml_path(str(out_xml))
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    for _ in range(100):
        mujoco.mj_step(model, data)

    ok = (model.nu == 17 and model.nq == 24 and model.nv == 23 and model.nmesh > 0)
    manifest = {
        "variant": variant,
        "generated_from": {
            "urdf": str(cfg["source_urdf"]), "urdf_sha256": sha256(urdf),
            "config": f"config/{variant}.json", "config_sha256": sha256(MODELS / "config" / f"{variant}.json"),
            "motors": "config/motors.json", "motors_sha256": sha256(MODELS / "config" / "motors.json"),
            "mujoco_version": mujoco.__version__,
        },
        "edits": edits,
        "joint_map": [
            {"contract_axis": i, "joint": jn,
             "joint_id": mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jn),
             "actuator_id": mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, jn),
             "motor": jc[jn]["motor"], "mechanism_limit_nm": jc[jn]["mechanism_limit_nm"]}
            for i, jn in enumerate(CONTRACT_JOINTS)
        ],
        "compile_check": {"nq": model.nq, "nv": model.nv, "nu": model.nu,
                          "nmesh": model.nmesh, "ngeom": model.ngeom,
                          "steps_smoke": 100, "ok": ok,
                          "base_z_after_100_steps": float(data.qpos[2])},
        "known_gaps": [
            "连杆惯量未辨识(URDF 原样)", "T-N 包络读图估值 calibrated=false",
            "地面摩擦名义值", "自碰撞未启用(臂碰撞体后续里程碑补充)",
            "J5 滚转轴等效惯量 ~0.0011 kg·m²(拍绕杆轴);armature=0 时 1 kHz 离散 kd<~2.2 N·m·s/rad 才稳定(M0 实测发现,转子反射惯量待 F3 辨识)",
        ],
    }
    (GENERATED / f"{variant}_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))

    print(f"[OK] {out_xml}")
    print(f"     nq={model.nq} nv={model.nv} nu={model.nu} nmesh={model.nmesh} ngeom={model.ngeom} "
          f"smoke 100 steps, base_z={data.qpos[2]:.4f}")
    print(f"[OK] manifest -> {GENERATED / (variant + '_manifest.json')}")
    return 0 if ok else 2


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="x4_tennis")
    sys.exit(build(ap.parse_args().variant))
