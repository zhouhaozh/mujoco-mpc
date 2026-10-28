import sys
import shutil
import subprocess
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "sckd_sim/models"))
from badminton_court import add_badminton_court


class BadmintonCourtTests(unittest.TestCase):
    def setUp(self):
        self.root = ET.fromstring('<mujoco><worldbody/></mujoco>')
        add_badminton_court(self.root)
        self.court = self.root.find("worldbody/body[@name='court']")

    def bounds(self, name):
        geom = self.court.find(f"geom[@name='{name}']")
        pos = np.fromstring(geom.get("pos"), sep=" ")
        size = np.fromstring(geom.get("size"), sep=" ")
        return np.array([pos[0] - size[0], pos[0] + size[0],
                         pos[1] - size[1], pos[1] + size[1]])

    def test_playing_boundaries_include_lines(self):
        np.testing.assert_allclose(self.bounds("ct_court"), [0, 13.4, -3.05, 3.05])
        np.testing.assert_allclose(self.bounds("ct_side_s_l")[2:], [-2.59, -2.55])
        np.testing.assert_allclose(self.bounds("ct_side_s_r")[2:], [2.55, 2.59])
        expected = {
            "ct_base_near": [0, .04], "ct_base_far": [13.36, 13.40],
            "ct_svc_near": [4.68, 4.72], "ct_svc_far": [8.68, 8.72],
            "ct_long_near": [.76, .80], "ct_long_far": [12.60, 12.64],
            "ct_mid_near": [0, 4.72], "ct_mid_far": [8.68, 13.40],
        }
        for name, bounds in expected.items():
            np.testing.assert_allclose(self.bounds(name)[:2], bounds, atol=1e-12)
        for geom in self.court.findall("geom"):
            if geom.get("rgba") == "0.98 0.98 0.95 1" and geom.get("type") == "box":
                size = np.fromstring(geom.get("size"), sep=" ")
                self.assertAlmostEqual(min(size[:2]) * 2, .04)

    def test_net_height_depth_and_clearance(self):
        for side, edge in (("l", -3.05), ("r", 3.05)):
            mesh = self.root.find(f"asset/mesh[@name='ct_net_{side}']")
            vertices = np.fromstring(mesh.get("vertex"), sep=" ").reshape(-1, 3)
            centre = vertices[vertices[:, 1] == 0, 2]
            end = vertices[vertices[:, 1] == edge, 2]
            np.testing.assert_allclose([centre.min(), centre.max()], [.764, 1.524])
            np.testing.assert_allclose([end.min(), end.max()], [.790, 1.550])
            post = self.court.find(f"geom[@name='ct_post_{side}']")
            np.testing.assert_allclose(np.fromstring(post.get("pos"), sep=" "), [6.7, edge, .775])

    def test_replacement_is_idempotent_and_collision_is_explicit(self):
        original = ET.tostring(self.root)
        add_badminton_court(self.root)
        self.assertEqual(ET.tostring(self.root), original)
        model = mujoco.MjModel.from_xml_string(ET.tostring(self.root, encoding="unicode"))
        self.assertEqual(model.nq, 0)
        self.assertFalse(model.geom_contype.any())
        add_badminton_court(self.root, collision_enabled=True)
        model = mujoco.MjModel.from_xml_string(ET.tostring(self.root, encoding="unicode"))
        self.assertEqual(int(np.count_nonzero(model.geom_contype)), 4)
        for name in ("ct_net_l", "ct_net_r", "ct_post_l", "ct_post_r"):
            self.assertEqual(model.geom(name).contype, 1)

    def test_generated_scenes_load_without_changing_state_dimensions(self):
        for name, dimensions in (("x4_950_scene", (24, 23, 17)),
                                 ("x4_950_replay", (31, 29, 17)),
                                 ("x4_tennis_demo", (31, 29, 17))):
            path = ROOT / "sckd_sim/models/generated" / name
            self.assertEqual(path.with_suffix(".xml").read_bytes(), path.with_suffix(".mjcf").read_bytes())
            model = mujoco.MjModel.from_xml_path(str(path.with_suffix(".mjcf")))
            self.assertEqual((model.nq, model.nv, model.nu), dimensions)
            data = mujoco.MjData(model)
            mujoco.mj_resetDataKeyframe(model, data, 0)
            mujoco.mj_forward(model, data)
            self.assertTrue(np.isfinite(data.qpos).all())
            np.testing.assert_allclose(model.geom("ct_court").size[:2], [6.7, 3.05])

    def test_demo_patch_does_not_accumulate_contact_sections(self):
        models = ROOT / "sckd_sim/models"
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            (work / "generated").mkdir()
            for name in ("patch_demo.py", "badminton_court.py"):
                shutil.copy(models / name, work / name)
            paths = [work / "generated" / f"x4_tennis_demo.{suffix}"
                     for suffix in ("mjcf", "xml")]
            for path in paths:
                shutil.copy(models / "generated" / path.name, path)
            command = [sys.executable, str(work / "patch_demo.py")]
            subprocess.run(command, check=True, capture_output=True)
            before = [path.read_bytes() for path in paths]
            subprocess.run(command, check=True, capture_output=True)
            self.assertEqual(before, [path.read_bytes() for path in paths])
            root = ET.parse(paths[0]).getroot()
            self.assertEqual(len(root.findall("contact")), 1)
            self.assertEqual(len(root.findall("contact/exclude")), 2)


if __name__ == "__main__":
    unittest.main()
