#!/usr/bin/env python3
"""比较同一命令序列的两次记录；不把模拟器一致性称为真机精度验收。"""
import argparse
import hashlib
import json
import math
from pathlib import Path

KNOWN_SOURCES = {"simulation_truth", "measured", "estimated", "command_integrated"}


def number(value, label, nullable=False):
    if value is None and nullable:
        return
    try:
        valid = type(value) in (int, float) and math.isfinite(value)
    except OverflowError:
        valid = False
    if not valid:
        raise ValueError(f"{label}: 必须为有限数值")


def integer(value, label, positive=False):
    if type(value) is not int or not (int(positive) <= value <= 2**64-1):
        raise ValueError(f"{label}: 必须为合法无符号整数")


def boolean(value, label):
    if type(value) is not bool:
        raise ValueError(f"{label}: 必须为布尔值")


def vector(value, size, label, nullable=False):
    if not isinstance(value, list) or len(value) != size:
        raise ValueError(f"{label}: 维度必须为 {size}")
    for item in value:
        number(item, label, nullable)


def timestamp(value, label, expected_clock=None):
    if not isinstance(value, dict) or value.get("clock") not in ("simulation", "monotonic"):
        raise ValueError(f"{label}: 未知时钟")
    number(value["seconds"], label)
    if expected_clock is not None and value["clock"] != expected_clock:
        raise ValueError(f"{label}: 时钟不一致")


def sampled_timestamp(known, stamp, clock, label):
    boolean(known, label+"_known")
    if known:
        timestamp(stamp, label, clock)
    elif stamp is not None:
        raise ValueError(f"{label}: 未知采样时间必须为 null")


def source(value, label):
    # unknown/unavailable 保留在报告中；不升级为已测量或真值。
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label}: 来源必须为非空字符串")


def validate_sample(row, previous=None):
    c, o = row["command"], row["observation"]
    integer(c["sequence"], "command.sequence", positive=True)
    integer(c["plan_id"], "command.plan_id")
    timestamp(c["issued"], "command.issued")
    clock = c["issued"]["clock"]
    timestamp(c["expires"], "command.expires", clock)
    if c["expires"]["seconds"] <= c["issued"]["seconds"]:
        raise ValueError("命令有效期必须晚于下发时间")
    if not isinstance(c["frame_id"], str) or not c["frame_id"]:
        raise ValueError("命令缺少坐标系")
    for field in ("body_twist", "q", "dq", "kp", "kd", "tau_ff"):
        vector(c[field], 3 if field == "body_twist" else 5, "command."+field)
    if any(v < 0 for v in c["kp"]+c["kd"]):
        raise ValueError("命令增益不能为负")
    integer(o["sequence"], "observation.sequence", positive=True)
    timestamp(o["received"], "observation.received", clock)
    if o["frame_id"] != c["frame_id"]:
        raise ValueError("反馈与命令坐标系不同，禁止直接计算误差")
    for field in ("base_available", "ready", "arm_degraded", "manual_takeover", "coherent_snapshot"):
        boolean(o[field], "observation."+field)
    vector(o["base_pose"], 3, "observation.base_pose", nullable=not o["base_available"])
    vector(o["body_twist"], 3, "observation.body_twist", nullable=not o["base_available"])
    sampled_timestamp(o["base_stamp_known"], o["base_stamp"], clock, "observation.base_stamp")
    source(o["base_source"], "observation.base_source")
    if not isinstance(o["joints"], list) or len(o["joints"]) != 5:
        raise ValueError("反馈关节维度必须为 5")
    for j in o["joints"]:
        boolean(j["available"], "joint.available")
        for field in ("q", "dq"):
            number(j[field], "joint."+field, nullable=not j["available"])
        # 温度等没有对应模型时可显式为null；它们不参与位置/速度统计。
        for field in ("torque", "temperature"):
            number(j[field], "joint."+field, nullable=True)
        integer(j["fault_bits"], "joint.fault_bits")
        sampled_timestamp(j["stamp_known"], j["stamp"], clock, "joint.stamp")
        source(j["position_source"], "joint.position_source")
        source(j["velocity_source"], "joint.velocity_source")
    boolean(row["result"]["submitted"], "result.submitted")
    if previous:
        p, po = previous["command"], previous["observation"]
        if c["sequence"] <= p["sequence"] or o["sequence"] <= po["sequence"]:
            raise ValueError("命令或反馈序号不严格递增")
        if clock != p["issued"]["clock"] or c["frame_id"] != p["frame_id"]:
            raise ValueError("同一记录中混用了时钟或坐标系")
        if c["issued"]["seconds"] <= p["issued"]["seconds"]:
            raise ValueError("命令时间不严格递增")
        if o["received"]["seconds"] < po["received"]["seconds"]:
            raise ValueError("反馈接收时间倒退")


def validate_mapping(mapping):
    if not isinstance(mapping, dict):
        raise ValueError("坐标映射必须为对象")
    for key in ("from_frame", "to_frame", "version"):
        if not isinstance(mapping.get(key), str) or not mapping[key]:
            raise ValueError("坐标映射缺少 "+key)
    vector(mapping["sign"], 5, "joint_map.sign")
    vector(mapping["offset_rad"], 5, "joint_map.offset_rad")
    if any(s not in (-1, 1) for s in mapping["sign"]):
        raise ValueError("坐标映射方向必须为 +1/-1")
    boolean(mapping["calibrated"], "joint_map.calibrated")


def load(path):
    raw = path.read_bytes()
    rows = []
    for index, line in enumerate(raw.splitlines(), start=1):
        try:
            row = json.loads(line)
        except (json.JSONDecodeError, UnicodeDecodeError) as error:
            raise ValueError(f"{path}: 第{index}行不是合法JSON: {error}") from error
        if not isinstance(row, dict):
            raise ValueError(f"{path}: 第{index}行必须为对象")
        rows.append(row)
    if not rows or rows[0].get("type") != "manifest":
        raise ValueError("缺少 manifest")
    if any(row.get("schema") != "X4_ROBOT_IO/1" for row in rows):
        raise ValueError("不支持的日志版本")
    samples = rows[1:]
    if not samples or any(row.get("type") != "sample" for row in samples):
        raise ValueError("无样本或存在未知记录")
    metadata = rows[0].get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError("manifest metadata 必须为对象")
    if "joint_map" in metadata:
        try:
            validate_mapping(metadata["joint_map"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"{path}: manifest坐标映射无效: {error}") from error
    previous = None
    for index, row in enumerate(samples, start=2):
        try:
            validate_sample(row, previous)
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"{path}: 第{index}行无效: {error}") from error
        previous = row
    return metadata, samples, hashlib.sha256(raw).hexdigest()


def statistics(values):
    if not values:
        return {"count": 0, "rmse": None, "max_abs": None}
    if any(not math.isfinite(x) for x in values):
        raise ValueError("指标中存在非有限值")
    return {"count": len(values), "rmse": math.sqrt(sum(x*x for x in values)/len(values)),
            "max_abs": max(abs(x) for x in values)}


def tracking(samples):
    result = {"q_rad": [], "dq_rad_s": []}
    for field, key in (("q", "q_rad"), ("dq", "dq_rad_s")):
        for axis in range(5):
            result[key].append(statistics([
                row["observation"]["joints"][axis][field] - row["command"][field][axis]
                for row in samples if usable(row["observation"]["joints"][axis], field)
            ]))
    result["rejected"] = sum(not row["result"]["submitted"] for row in samples)
    result["base_timestamp_known"] = sum(row["observation"]["base_stamp_known"] for row in samples)
    result["base_timestamp_unknown"] = len(samples)-result["base_timestamp_known"]
    result["base_sources"] = sorted({row["observation"]["base_source"] for row in samples})
    result["base_source_unknown"] = sum(row["observation"]["base_source"] not in KNOWN_SOURCES for row in samples)
    result["joint_status"] = []
    for axis in range(5):
        joints = [row["observation"]["joints"][axis] for row in samples]
        result["joint_status"].append({
            "position_sources": sorted({j["position_source"] for j in joints}),
            "velocity_sources": sorted({j["velocity_source"] for j in joints}),
            "position_source_unknown": sum(j["position_source"] not in KNOWN_SOURCES for j in joints),
            "velocity_source_unknown": sum(j["velocity_source"] not in KNOWN_SOURCES for j in joints),
            "stamp_unknown": sum(not j["stamp_known"] for j in joints),
            "unavailable": sum(not j["available"] for j in joints),
            "q_excluded": sum(not usable(j, "q") for j in joints),
            "dq_excluded": sum(not usable(j, "dq") for j in joints)})
    result["note"] = "逐条下发命令与当次返回反馈比较，未补偿延迟；未知时间戳保留并计数，无效来源不参与统计"
    return result


def usable(joint, field):
    return joint["available"] and joint["position_source" if field == "q" else "velocity_source"] in KNOWN_SOURCES


def compare(left_path, right_path):
    lm, left, lh = load(left_path)
    rm, right, rh = load(right_path)
    if [r["command"]["sequence"] for r in left] != [r["command"]["sequence"] for r in right]:
        raise ValueError("样本序号集合不同，不能截取交集冒充完整配对")
    mapping = rm.get("joint_map")
    errors = []
    time_errors = []
    l0 = left[0]["command"]["issued"]["seconds"]
    r0 = right[0]["command"]["issued"]["seconds"]
    for lrow, rrow in zip(left, right):
        a, b = lrow["command"], rrow["command"]
        different = a["frame_id"] != b["frame_id"]
        if different and (not mapping or mapping["from_frame"] != a["frame_id"] or mapping["to_frame"] != b["frame_id"]):
            raise ValueError("模型坐标不同且没有显式映射记录")
        for field in ("body_twist", "q", "dq", "kp", "kd", "tau_ff"):
            av = list(a[field])
            if different and field in ("q", "dq", "tau_ff"):
                av = [mapping["sign"][j]*v + (mapping["offset_rad"][j] if field == "q" else 0)
                      for j, v in enumerate(av)]
            if len(av) != len(b[field]):
                raise ValueError("命令维度不一致")
            errors.extend(x-y for x, y in zip(av, b[field]))
        time_errors.append((a["issued"]["seconds"]-l0) - (b["issued"]["seconds"]-r0))
    same_frame = left[0]["command"]["frame_id"] == right[0]["command"]["frame_id"]
    state_delta = None
    if same_frame:
        state_delta = {"q_rad": [], "dq_rad_s": []}
        for field, key in (("q", "q_rad"), ("dq", "dq_rad_s")):
            for axis in range(5):
                state_delta[key].append(statistics([
                    b["observation"]["joints"][axis][field]-a["observation"]["joints"][axis][field]
                    for a, b in zip(left, right)
                    if usable(a["observation"]["joints"][axis], field) and usable(b["observation"]["joints"][axis], field)
                ]))
    return {"schema": "X4_ROBOT_IO_COMPARISON/1", "samples": len(left),
            "left": {"path": str(left_path.resolve()), "sha256": lh, "metadata": lm, "tracking": tracking(left)},
            "right": {"path": str(right_path.resolve()), "sha256": rh, "metadata": rm, "tracking": tracking(right)},
            "command_difference": statistics(errors), "relative_schedule_difference_s": statistics(time_errors),
            "joint_state_difference_same_frame": state_delta,
            "physical_alignment_validated": False,
            "limitations": ["下发成功不代表驱动确认", "跨模型坐标未标定时不输出跨模型状态误差",
                            "模型、初态、外力及碰撞差异会影响状态，须结合manifest解释",
                            "状态差按命令序号配对，不代表反馈在同一时刻采样；调度差只比较命令计划时间"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left", type=Path, required=True)
    parser.add_argument("--right", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = compare(args.left, args.right)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as f:
        json.dump(result, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write("\n")
    print(json.dumps({"samples": result["samples"], "command_difference": result["command_difference"],
                      "output": str(args.output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
