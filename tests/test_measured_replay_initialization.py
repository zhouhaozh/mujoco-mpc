import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from measured_replay_initialization import DEFAULT_SETTINGS, apply_initialization


SOURCE = ROOT / "outputs/multitrajectory_sysid_20261030/activation_model"


def payload():
    return {
        "initial": {"q_rad": [1, 2, 3, 4, 5], "chassis_servo_state": {"wrong": True}},
        "commands": [{"t_s": 0.0, "q_rad": [0] * 5, "dq_rad_s": [0] * 5,
                      "base_twist": [0, 0, 0]}],
        "arm_start_policy": "immediate",
    }


def digest(data):
    return hashlib.sha256(data).hexdigest()


def write_fixture(directory, *, enabled=True):
    directory = Path(directory)
    null_source = directory / "null.csv"
    empty_source = directory / "empty.csv"
    null_source.write_bytes(b"timestamp,value\n0,1\n")
    empty_source.write_bytes(b"timestamp,value\n0,2\n")
    settings = {
        "schema_version": 1,
        "enabled": enabled,
        "arm_start_policy": "hold_until_fresh_command",
        "identification_source": "synthetic unit-test fixture",
        "records_by_source_sha256": {
            digest(null_source.read_bytes()): {
                "label": "NULL", "source_file": "null.csv",
                "chassis_servo_state": None, "prefix_end_s": 0.048,
            },
            digest(empty_source.read_bytes()): {
                "label": "EMPTY", "source_file": "empty.csv",
                "chassis_servo_state": {}, "prefix_end_s": 0.049,
            },
        },
    }
    settings_path = directory / "settings.json"
    settings_path.write_text(json.dumps(settings))
    return settings_path, null_source, empty_source


class MeasuredReplayInitializationTests(unittest.TestCase):
    def test_default_configuration_is_portable_and_complete(self):
        settings = json.loads(DEFAULT_SETTINGS.read_text())
        self.assertTrue(settings["enabled"])
        self.assertEqual(settings["arm_start_policy"], "hold_until_fresh_command")
        self.assertEqual(len(settings["records_by_source_sha256"]), 10)
        self.assertEqual(len(set(settings["records_by_source_sha256"])), 10)
        for source_sha256, record in settings["records_by_source_sha256"].items():
            self.assertEqual(len(source_sha256), 64)
            int(source_sha256, 16)
            self.assertIn("source_file", record)
            self.assertIn("chassis_servo_state", record)
            self.assertIn("prefix_end_s", record)

    @unittest.skipUnless((SOURCE / "initial_states.json").is_file() and
                         (SOURCE / "baseline/manifest.json").is_file(),
                         "identification evidence is not present in this checkout")
    def test_frozen_records_match_identification_outputs_when_available(self):
        settings = json.loads(DEFAULT_SETTINGS.read_text())
        states = json.loads((SOURCE / "initial_states.json").read_text())
        manifest = json.loads((SOURCE / "baseline/manifest.json").read_text())
        for group in range(1, 11):
            source = manifest["sources"][str(group)]
            record = settings["records_by_source_sha256"][source["sha256"]]
            self.assertEqual(record["chassis_servo_state"], states[str(group)]["state"])
            self.assertEqual(record["prefix_end_s"], states[str(group)]["prefix_end_s"])
            self.assertEqual(record["label"], source["label"])

    def test_content_hash_matches_after_rename_and_preserves_payload(self):
        with tempfile.TemporaryDirectory() as directory:
            settings, _, source = write_fixture(directory)
            renamed = Path(directory) / "renamed.csv"
            renamed.write_bytes(source.read_bytes())
            original = payload()
            commands = copy.deepcopy(original["commands"])
            applied, metadata = apply_initialization(
                original, renamed, settings_path=settings)
        self.assertEqual(metadata["status"], "matched")
        self.assertEqual(metadata["label"], "EMPTY")
        self.assertEqual(applied["arm_start_policy"], "hold_until_fresh_command")
        self.assertEqual(applied["initial"]["chassis_servo_state"], {})
        self.assertEqual(applied["commands"], commands)
        self.assertEqual(original, payload())

    def test_changed_content_is_not_misapplied_and_uses_cold_chassis(self):
        with tempfile.TemporaryDirectory() as directory:
            settings, source, _ = write_fixture(directory)
            changed = Path(directory) / "changed.csv"
            changed.write_bytes(source.read_bytes() + b"\n")
            applied, metadata = apply_initialization(
                payload(), changed, settings_path=settings)
        self.assertEqual(metadata["status"], "not_cached")
        self.assertNotIn("chassis_servo_state", applied["initial"])
        self.assertEqual(applied["arm_start_policy"], "hold_until_fresh_command")

    def test_null_and_empty_state_keep_distinct_meanings(self):
        with tempfile.TemporaryDirectory() as directory:
            settings, null_source, empty_source = write_fixture(directory)
            cold, cold_meta = apply_initialization(
                payload(), null_source, settings_path=settings)
            initialized, initialized_meta = apply_initialization(
                payload(), empty_source, settings_path=settings)
        self.assertNotIn("chassis_servo_state", cold["initial"])
        self.assertIsNone(cold_meta["chassis_servo_state"])
        self.assertEqual(initialized["initial"]["chassis_servo_state"], {})
        self.assertEqual(initialized_meta["chassis_servo_state"], {})

    def test_disabled_configuration_returns_an_unmodified_copy(self):
        with tempfile.TemporaryDirectory() as directory:
            settings, source, _ = write_fixture(directory, enabled=False)
            original = payload()
            applied, metadata = apply_initialization(
                original, source, settings_path=settings)
        self.assertEqual(metadata["status"], "disabled")
        self.assertEqual(applied, original)
        self.assertIsNot(applied, original)
        self.assertEqual(digest(json.dumps(applied, sort_keys=True).encode()),
                         digest(json.dumps(original, sort_keys=True).encode()))


if __name__ == "__main__":
    unittest.main()
