import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET

import mujoco
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "sckd_sim/models"))
from build_x4_replay import build_replay


class X4ReplayBuilderTests(unittest.TestCase):
    profiles = ROOT / "sckd_sim/models/config/measured_dynamics"

    def test_profiles_build_isolated_loadable_replays_without_source_changes(self):
        for name in ("baseline", "fitted"):
            with self.subTest(profile=name), tempfile.TemporaryDirectory() as directory:
                source = self.profiles / name / "scene.mjcf"
                before = hashlib.sha256(source.read_bytes()).hexdigest()
                output = Path(directory) / name / "scene.mjcf"
                manifest = build_replay(source, output)

                self.assertEqual(hashlib.sha256(source.read_bytes()).hexdigest(), before)
                self.assertEqual(manifest["dimensions"], {
                    "nq": 31, "nv": 29, "nu": 17, "robot_qpos": 24,
                    "ball_qpos": 7, "robot_qvel": 23, "ball_qvel": 6,
                })
                self.assertTrue(output.with_suffix(".xml").is_file())
                self.assertEqual(
                    json.loads((output.parent / "scene_manifest.json").read_text()), manifest)

                original = mujoco.MjModel.from_xml_path(str(source))
                replay = mujoco.MjModel.from_xml_path(str(output))
                np.testing.assert_array_equal(replay.qpos0[:original.nq], original.qpos0)
                for field in ("body_mass", "body_inertia", "body_pos", "body_quat",
                              "jnt_type", "jnt_pos", "jnt_axis", "jnt_range",
                              "dof_armature", "dof_damping", "dof_frictionloss",
                              "geom_type", "geom_size", "geom_pos", "geom_quat",
                              "geom_contype", "geom_conaffinity", "geom_friction",
                              "geom_solref", "geom_solimp", "geom_margin", "geom_gap",
                              "actuator_trnid", "actuator_ctrlrange", "actuator_gear"):
                    original_values = getattr(original, field)
                    replay_values = getattr(replay, field)
                    np.testing.assert_array_equal(
                        replay_values[:len(original_values)], original_values, err_msg=field)
                for joint_name in ("Joint1", "Joint2", "Joint3", "Joint4", "Joint5"):
                    original_dof = original.jnt_dofadr[original.joint(joint_name).id]
                    replay_dof = replay.jnt_dofadr[replay.joint(joint_name).id]
                    for field in ("armature", "damping", "frictionloss"):
                        self.assertEqual(getattr(replay, f"dof_{field}")[replay_dof],
                                         getattr(original, f"dof_{field}")[original_dof])
                face = replay.geom("racket_face").id
                ball = replay.geom("ball_geom").id
                self.assertEqual(replay.geom_rgba[face, 3], 0)
                self.assertEqual(replay.geom_contype[ball], 0)
                self.assertEqual(replay.geom_conaffinity[ball], 0)
                meshdir = ET.parse(output).getroot().find("compiler").get("meshdir")
                self.assertTrue(Path(meshdir).is_absolute())

    def test_existing_replay_is_idempotent_and_does_not_duplicate_ball(self):
        source = ROOT / "sckd_sim/models/generated/x4_950_replay.mjcf"
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "first.mjcf"
            first = build_replay(source, output)
            second_output = Path(directory) / "second.mjcf"
            second = build_replay(output, second_output)

            self.assertTrue(first["static_check"]["input_already_replay"])
            self.assertTrue(second["static_check"]["input_already_replay"])
            tree = ET.parse(second_output)
            self.assertEqual(len(tree.findall(".//freejoint[@name='ball_free']")), 1)
            self.assertEqual(len(tree.findall(".//geom[@name='ball_geom']")), 1)
            model = mujoco.MjModel.from_xml_path(str(second_output))
            self.assertEqual((model.nq, model.nv, model.nu), (31, 29, 17))

    def test_output_must_not_overwrite_source(self):
        source = self.profiles / "baseline" / "scene.mjcf"
        with self.assertRaisesRegex(ValueError, "must differ"):
            build_replay(source, source)


if __name__ == "__main__":
    unittest.main()
