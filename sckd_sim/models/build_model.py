#!/usr/bin/env python3
"""x4_tennis 模型管线:URDF(+STL) -> MJCF + provenance manifest。

用法(conda sim 环境):
  /home/ubuntu/miniconda3/envs/sim/bin/python models/build_model.py [--variant x4_tennis]

产物(models/generated/):
  x4_tennis_scene.mjcf      编译验证过的场景(机器人+地面+灯+执行器+keyframe)
  x4_tennis_manifest.json   变更清单(provenance):来源哈希、每项编辑及理由、关节/执行器映射

设计依据: docs/SIM_SYSTEM_DESIGN.md §3.2/§3.2.1(管线与事实表)。
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

    # --- 1) 预处理:注入 <mujoco><compiler angle="radian"/></mujoco>,写到源目录(STL 同基) ---
    tree = ET.parse(urdf)
    root = tree.getroot()
    if root.find("mujoco") is None:
        mje = ET.SubElement(root, "mujoco")
        comp = ET.SubElement(mje, "compiler")
        comp.set("angle", "radian")
        edits.append({"edit": "inject_mujoco_compiler",
                      "reason": "角度统一 radian;不改源 URDF;临时副本与 STL 同目录以便解析 meshes/ 相对路径"})
    pre = urdf.parent / f"_{variant}_pre.urdf"
    tree.write(pre, encoding="unicode", xml_declaration=True)

    # --- 2) 载入 mjSpec ---
    spec = mujoco.MjSpec.from_file(str(pre))
    edits.append({"edit": "preserve_source_inertials",
                  "reason": "URDF <inertial> 原样保留(质量/COM/惯量未辨识,事实表 §3.2.1)"})

    # --- 3) base_link 加 freejoint(浮动底盘) ---
    base = spec.body("base_link")
    base.add_freejoint(name="root")
    edits.append({"edit": "add_freejoint_base_link", "reason": "URDF 无世界连接;底盘需 6 DOF 浮动"})

    # --- 4) 碰撞策略:原 STL 网格仅作视觉;轮子加基元圆柱碰撞 ---
    wc = cfg["wheel_collision"]
    for body in spec.bodies:
        for g in body.geoms:
            g.contype = 0
            g.conaffinity = 0
    edits.append({"edit": "mesh_geoms_visual_only",
                  "reason": "30k/6k 三角网格参与接触代价高且基元更可控(设计 §3.2 碰撞体替换,清单可审计)"})
    for name, side in WHEEL_BODIES:
        b = spec.body(name)
        pos = wc["center_in_wheel_frame"][side]
        b.add_geom(
            type=mujoco.mjtGeom.mjGEOM_CYLINDER,
            name=f"col_{name}",
            size=[wc["radius_m"], wc["half_width_m"], 0.0],
            pos=pos,
            quat=[0.7071067811865476, 0.7071067811865476, 0.0, 0.0],  # 圆柱 z 轴 -> 轮轴 y
            friction=wc["friction"],
            contype=1, conaffinity=1, mass=0.0,
        )
    edits.append({"edit": "wheel_primitive_collider",
                  "reason": f"圆柱 r={wc['radius_m']} half={wc['half_width_m']} m,几何取自 STL bbox/径向 p99(§3.2.1);质量 0 不改变 URDF 惯量"})

    # --- 5) 地面 + 灯 ---
    spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_PLANE, name="floor",
                            size=[30.0, 30.0, 0.1], friction=wc["friction"],
                            contype=1, conaffinity=1)
    spec.worldbody.add_light(pos=[0, 0, 4], dir=[0, 0, -1],
                             type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL)
    edits.append({"edit": "floor_plane", "reason": "M0 平地;μ=1 名义值(F1),F3 轮地标定后替换(§9-3)"})

    # --- 6) 执行器:17 电机,与关节同名,ctrlrange=±机构限矩 ---
    jc = {j["joint"]: j for j in cfg["joints"]}
    for jn in CONTRACT_JOINTS:
        spec.add_actuator(name=jn, trntype=mujoco.mjtTrn.mjTRN_JOINT, target=jn,
                          gear=[1.0, 0, 0, 0, 0, 0],
                          ctrllimited=True,
                          ctrlrange=[-jc[jn]["mechanism_limit_nm"], jc[jn]["mechanism_limit_nm"]])
    edits.append({"edit": "add_actuators_17",
                  "reason": "gear=1(减速器输出端口径,设计 §3.3);ctrlrange=±机构限矩为二级保护,一级在电机模型限幅链"})

    # --- 7) keyframe:home(轮接地) ---
    base_z = cfg["keyframe_home"]["base_z_m"]
    qpos = [0.0] * (7 + 17)
    qpos[2] = base_z
    spec.add_key(name="home", qpos=qpos)
    edits.append({"edit": "keyframe_home",
                  "reason": f"base_z={base_z}(steer z 0.043 + 轮关节 z -0.1689 + 轮半径 {wc['radius_m']});关节全零=臂水平伸展"})

    # --- 8) 步长 ---
    try:
        spec.option.timestep = 0.001
    except AttributeError:
        pass  # 保存后由编译模型侧设置
    edits.append({"edit": "timestep_1ms", "reason": "设计 §1.2:物理 1 ms;0.5 ms 用于接触校核场景"})

    # --- 9) 编译 + 保存(改写 meshdir 为相对 generated/ 的路径)+ 从产物重载自检 ---
    model = spec.compile()
    out_xml = GENERATED / f"{variant}_scene.mjcf"
    out_xml.write_text(spec.to_xml())
    pre.unlink(missing_ok=True)

    # 保存的 MJCF 位于 generated/,网格在 ../source/<variant>/;改写使其自包含可加载
    stree = ET.parse(out_xml)
    sroot = stree.getroot()
    scomp = sroot.find("compiler")
    if scomp is None:
        scomp = ET.SubElement(sroot, "compiler")
    scomp.set("meshdir", f"../source/{urdf.parent.name}")
    stree.write(out_xml, encoding="unicode", xml_declaration=True)

    model = mujoco.MjModel.from_xml_path(str(out_xml))   # 从产物重载,验证自包含
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    for _ in range(100):
        mujoco.mj_step(model, data)

    ok = (model.nu == 17 and model.nq == 24 and model.nv == 23)
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
        "compile_check": {"nq": model.nq, "nv": model.nv, "nu": model.nu, "steps_smoke": 100, "ok": ok,
                          "base_z_after_100_steps": float(data.qpos[2])},
        "known_gaps": [
            "连杆惯量未辨识(URDF 原样)", "T-N 包络读图估值 calibrated=false",
            "地面摩擦名义值", "自碰撞未启用(臂碰撞体后续里程碑补充)",
            "J5 滚转轴等效惯量 ~0.0011 kg·m²(拍绕杆轴);armature=0 时 1 kHz 离散 kd<~2.2 N·m·s/rad 才稳定(M0 实测发现,转子反射惯量待 F3 辨识)",
        ],
    }
    (GENERATED / f"{variant}_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))

    print(f"[OK] {out_xml}")
    print(f"     nq={model.nq} nv={model.nv} nu={model.nu} smoke 100 steps, base_z={data.qpos[2]:.4f}")
    print(f"[OK] manifest -> {GENERATED / (variant + '_manifest.json')}")
    return 0 if ok else 2


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="x4_tennis")
    sys.exit(build(ap.parse_args().variant))
