import importlib.util
import copy
import json
from pathlib import Path
import tempfile
import unittest

SPEC = importlib.util.spec_from_file_location("compare_robot_io", Path(__file__).parents[1] / "tools/compare_robot_io.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class ComparisonTest(unittest.TestCase):
    def write(self, path, frame="mpc_model", seq=1, metadata=None, q=.2):
        stamp = {"seconds": .001, "clock": "simulation"}
        command = {"sequence": seq, "plan_id": 0, "frame_id": frame, "issued": stamp,
                   "expires": {"seconds": .101, "clock": "simulation"},
                   "body_twist": [0., 0., 0.], "q": [.1]*5, "dq": [0.]*5,
                   "kp": [10.]*5, "kd": [1.]*5, "tau_ff": [0.]*5}
        observation = {"sequence": seq, "frame_id": frame, "received": stamp,
                       "joints": [{"q": q, "dq": 0., "torque": 0., "temperature": None,
                                   "available": True, "stamp_known": True, "stamp": stamp,
                                   "position_source": "measured", "velocity_source": "measured",
                                   "fault_bits": 0} for _ in range(5)],
                       "base_available": True, "base_pose": [0., 0., 0.], "body_twist": [0., 0., 0.],
                       "base_stamp_known": False, "base_stamp": None, "base_source": "command_integrated",
                       "ready": True, "arm_degraded": False, "manual_takeover": False,
                       "coherent_snapshot": False}
        rows = [{"type": "manifest", "metadata": metadata or {}},
                {"type": "sample", "command": command, "observation": observation,
                 "result": {"submitted": True}}]
        path.write_text("\n".join(json.dumps({"schema": "X4_ROBOT_IO/1", **row}) for row in rows)+"\n")

    def change(self, path, edit, second_sample=False):
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        if second_sample:
            row = copy.deepcopy(rows[-1])
            row["command"]["sequence"] += 1
            row["command"]["issued"]["seconds"] += .001
            row["command"]["expires"]["seconds"] += .001
            row["observation"]["sequence"] += 1
            row["observation"]["received"]["seconds"] += .001
            rows.append(row)
        edit(rows[-1])
        path.write_text("\n".join(json.dumps(row) for row in rows)+"\n")

    def test_independent_command_and_tracking_metrics(self):
        with tempfile.TemporaryDirectory() as folder:
            a, b = Path(folder)/"a", Path(folder)/"b"
            self.write(a, q=.2); self.write(b, q=.4)
            result = MODULE.compare(a, b)
            self.assertEqual(result["command_difference"]["max_abs"], 0)
            self.assertAlmostEqual(result["left"]["tracking"]["q_rad"][0]["rmse"], .1)
            self.assertAlmostEqual(result["joint_state_difference_same_frame"]["q_rad"][0]["rmse"], .2)
            self.assertEqual(result["left"]["tracking"]["base_timestamp_known"], 0)
            self.assertFalse(result["physical_alignment_validated"])

    def test_no_intersection_only_comparison(self):
        with tempfile.TemporaryDirectory() as folder:
            a, b = Path(folder)/"a", Path(folder)/"b"
            self.write(a); self.write(b, seq=2)
            with self.assertRaises(ValueError): MODULE.compare(a, b)

    def test_cross_frame_requires_explicit_mapping(self):
        with tempfile.TemporaryDirectory() as folder:
            a, b = Path(folder)/"a", Path(folder)/"b"
            self.write(a); self.write(b, frame="sdk_model")
            with self.assertRaises(ValueError): MODULE.compare(a, b)
            self.write(b, frame="sdk_model", metadata={"joint_map": {
                "from_frame": "mpc_model", "to_frame": "sdk_model",
                "sign": [1]*5, "offset_rad": [0]*5, "calibrated": False, "version": "test/1"}})
            result = MODULE.compare(a, b)
            self.assertIsNone(result["joint_state_difference_same_frame"])
            self.assertEqual(result["command_difference"]["max_abs"], 0)

    def test_mixed_clock_stream_is_rejected_even_when_numeric_times_match(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/"mixed"
            self.write(path)
            def switch_clock(row):
                for field in ("issued", "expires"):
                    row["command"][field]["clock"] = "monotonic"
                row["observation"]["received"]["clock"] = "monotonic"
                for joint in row["observation"]["joints"]:
                    joint["stamp"]["clock"] = "monotonic"
            self.change(path, switch_clock, second_sample=True)
            with self.assertRaisesRegex(ValueError, "混用了时钟"):
                MODULE.compare(path, path)

    def test_observation_frame_must_match_command_before_metrics(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/"frame"
            self.write(path)
            self.change(path, lambda r: r["observation"].update(frame_id="different_model"))
            with self.assertRaisesRegex(ValueError, "反馈与命令坐标系不同"):
                MODULE.compare(path, path)

    def test_unavailable_and_unknown_sources_never_produce_zero_error_metrics(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/"unknown"
            self.write(path, q=.1)
            def unknown(row):
                row["observation"]["joints"][0]["position_source"] = "unavailable"
                row["observation"]["joints"][1]["velocity_source"] = "unrecognized_sensor"
            self.change(path, unknown)
            result = MODULE.compare(path, path)
            track = result["left"]["tracking"]
            self.assertEqual(track["q_rad"][0], {"count": 0, "rmse": None, "max_abs": None})
            self.assertEqual(track["dq_rad_s"][1]["count"], 0)
            self.assertEqual(track["dq_rad_s"][0]["count"], 1)
            self.assertEqual(track["joint_status"][0]["position_source_unknown"], 1)
            self.assertEqual(track["joint_status"][1]["velocity_sources"], ["unrecognized_sensor"])
            self.assertIsNone(result["joint_state_difference_same_frame"]["q_rad"][0]["rmse"])

    def test_unknown_timestamps_remain_unknown_and_are_counted(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/"nostamp"
            self.write(path)
            self.change(path, lambda r: r["observation"]["joints"][0].update(stamp_known=False, stamp=None))
            result = MODULE.compare(path, path)
            track = result["left"]["tracking"]
            self.assertEqual(track["joint_status"][0]["stamp_unknown"], 1)
            self.assertEqual(track["base_timestamp_unknown"], 1)
            self.assertEqual(track["q_rad"][0]["count"], 1)

    def test_invalid_sequence_dimensions_values_and_timestamps_are_rejected(self):
        edits = {
            "zero sequence": lambda r: r["command"].update(sequence=0),
            "boolean sequence": lambda r: r["command"].update(sequence=True),
            "wrong twist dimension": lambda r: r["command"].update(body_twist=[0., 0.]),
            "nonfinite command": lambda r: r["command"]["q"].__setitem__(0, float("nan")),
            "nonfinite available observation": lambda r: r["observation"]["joints"][0].update(q=float("inf")),
            "invalid lifetime": lambda r: r["command"]["expires"].update(seconds=0.),
            "expiration clock": lambda r: r["command"]["expires"].update(clock="monotonic"),
            "sample clock": lambda r: r["observation"]["joints"][0]["stamp"].update(clock="monotonic"),
            "unknown stamp with value": lambda r: r["observation"]["joints"][0].update(stamp_known=False),
        }
        with tempfile.TemporaryDirectory() as folder:
            for label, edit in edits.items():
                with self.subTest(label=label):
                    path = Path(folder)/"bad"
                    self.write(path)
                    self.change(path, edit)
                    with self.assertRaises(ValueError):
                        MODULE.compare(path, path)

    def test_invalid_frame_mapping_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/"mapping"
            self.write(path, metadata={"joint_map": {
                "from_frame": "mpc_model", "to_frame": "sdk_model", "version": "test/1",
                "sign": [0]*5, "offset_rad": [0]*5, "calibrated": False}})
            with self.assertRaisesRegex(ValueError, "方向必须"):
                MODULE.compare(path, path)

    def test_stream_frame_changes_and_observation_sequence_reuse_are_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/"stream"
            self.write(path)
            def switch_frame(row):
                row["command"]["frame_id"] = "sdk_model"
                row["observation"]["frame_id"] = "sdk_model"
            self.change(path, switch_frame, second_sample=True)
            with self.assertRaisesRegex(ValueError, "混用了时钟或坐标系"):
                MODULE.compare(path, path)
            self.write(path)
            self.change(path, lambda r: r["observation"].update(sequence=1), second_sample=True)
            with self.assertRaisesRegex(ValueError, "序号不严格递增"):
                MODULE.compare(path, path)


if __name__ == "__main__":
    unittest.main()
