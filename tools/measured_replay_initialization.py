"""Apply frozen measured-replay initialization by source-content identity.

The settings contain effective controller boundary states estimated from a
prefix of each recorded trajectory.  They are replay inputs, not identified
hardware state or shared physical parameters.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SETTINGS = ROOT / "sckd_sim/models/config/measured_replay_initialization.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_settings(path: Path) -> dict:
    settings = json.loads(path.read_text())
    if settings.get("schema_version") != 1:
        raise ValueError("measured replay initialization schema_version must be 1")
    policy = settings.get("arm_start_policy")
    if policy not in ("immediate", "hold_until_fresh_command"):
        raise ValueError("unsupported measured replay arm_start_policy")
    records = settings.get("records_by_source_sha256")
    if not isinstance(records, dict):
        raise ValueError("records_by_source_sha256 must be an object")
    for digest, record in records.items():
        if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError("record key must be a lowercase SHA-256")
        state = record.get("chassis_servo_state")
        if state is not None and not isinstance(state, dict):
            raise ValueError("chassis_servo_state must be null or an object")
    return settings


def apply_initialization(payload, source_path, *, settings_path=DEFAULT_SETTINGS):
    """Return a copied replay payload and provenance for the applied policy.

    Cached chassis states match only by exact source-file SHA-256.  An unknown
    source uses a cold chassis-controller start while retaining the configured
    arm start policy.  No command or measured initial state is mutated in place.
    """
    settings_path = Path(settings_path)
    source_path = Path(source_path)
    settings = _load_settings(settings_path)
    result = copy.deepcopy(payload)
    source_sha256 = _sha256(source_path)
    metadata = {
        "status": "disabled" if not settings.get("enabled", False) else "not_cached",
        "enabled": bool(settings.get("enabled", False)),
        "settings_path": str(settings_path.resolve()),
        "settings_sha256": _sha256(settings_path),
        "source_path": str(source_path.resolve()),
        "source_sha256": source_sha256,
        "arm_start_policy": None,
        "chassis_servo_state": None,
        "prefix_end_s": None,
        "identification_source": settings.get("identification_source"),
    }
    if not metadata["enabled"]:
        return result, metadata

    policy = settings["arm_start_policy"]
    result["arm_start_policy"] = policy
    metadata["arm_start_policy"] = policy
    record = settings["records_by_source_sha256"].get(source_sha256)
    initial = result.setdefault("initial", {})
    if record is None:
        # Do not preserve a state supplied for some other recording.
        initial.pop("chassis_servo_state", None)
        return result, metadata

    state = copy.deepcopy(record.get("chassis_servo_state"))
    if state is None:
        initial.pop("chassis_servo_state", None)
    else:
        initial["chassis_servo_state"] = state
    metadata.update(status="matched", chassis_servo_state=state,
                    prefix_end_s=record.get("prefix_end_s"),
                    source_file=record.get("source_file"), label=record.get("label"))
    return result, metadata
