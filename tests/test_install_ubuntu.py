"""验证安装入口的提权边界、失败中止和跨目录运行，不执行真实 apt。"""

import importlib.util
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]


class InstallUbuntuTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="mpc-install-test-")
        self.addCleanup(self.temporary.cleanup)
        self.base = Path(self.temporary.name)
        self.project = self.base / "project with spaces"
        (self.project / "tools").mkdir(parents=True)
        (self.project / "sckd_sim/build").mkdir(parents=True)
        self.script = self.project / "tools/install_ubuntu.sh"
        shutil.copyfile(ROOT / "tools/install_ubuntu.sh", self.script)
        self.bin = self.base / "bin"
        self.bin.mkdir()
        self.log = self.base / "commands.log"
        self.env = dict(os.environ, PATH=f"{self.bin}:{os.environ['PATH']}",
                        MOCK_LOG=str(self.log), MOCK_MISSING="0", MOCK_ARCH="x86_64",
                        MOCK_VIEW_FILE=str(self.base / "view_environment.txt"),
                        MOCK_APT_STATUS="0", MOCK_PYTHON_STATUS="0", MOCK_SMOKE_STATUS="0")
        self.write_command("uname", 'if [[ "$1" == -m ]]; then echo "$MOCK_ARCH"; else echo Linux; fi')
        self.write_command("dpkg-query", '''if [[ "$MOCK_MISSING" == 1 ]]; then exit 1; fi
printf 'install ok installed' ''')
        self.write_command("sudo", 'exec "$@"')
        self.write_command("apt-get", 'exit "$MOCK_APT_STATUS"')
        self.write_command("python3", 'exit "$MOCK_PYTHON_STATUS"')
        self.write_command("smoke", 'printf "%s" "${M2_VIEW+set}" > "$MOCK_VIEW_FILE"\n'
                           'exit "$MOCK_SMOKE_STATUS"',
                           self.project / "sckd_sim/build/m2_intercept", record_cwd=True)

    def write_command(self, name, body, path=None, record_cwd=False):
        path = path or self.bin / name
        arguments = '"$PWD" "$@"' if record_cwd else '"$@"'
        path.write_text('#!/bin/bash\n'
                        f'printf "%s\\t" {name} {arguments} >> "$MOCK_LOG"\n'
                        'printf "\\n" >> "$MOCK_LOG"\n' + body + '\n')
        path.chmod(0o755)

    def invoke(self, *args, **environment):
        return subprocess.run(["/bin/bash", str(self.script), *args], cwd=self.base,
                              env=dict(self.env, **environment), capture_output=True,
                              text=True, timeout=10)

    def calls(self, name):
        if not self.log.exists():
            return []
        lines = [line.rstrip("\t").split("\t") for line in self.log.read_text().splitlines()]
        return [line[1:] for line in lines if line[0] == name]

    def test_installed_packages_do_not_require_apt(self):
        result = self.invoke()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertFalse(self.calls("sudo"))
        self.assertFalse(self.calls("apt-get"))
        self.assertEqual(self.calls("smoke")[0][0], str(self.project / "sckd_sim"))

    def test_skip_system_deps_and_arguments_work_outside_repository(self):
        result = self.invoke("--skip-system-deps", "--jobs", "3", "--index-url",
                             "https://mirror.example/simple")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertFalse(self.calls("dpkg-query"))
        self.assertFalse(self.calls("apt-get"))
        command = self.calls("python3")[-1]
        self.assertEqual(command[0], str(self.project / "tools/setup_predictor_ubuntu.py"))
        self.assertEqual(command[command.index("--jobs") + 1], "3")
        self.assertEqual(command[command.index("--index-url") + 1], "https://mirror.example/simple")
        self.assertEqual(command[command.index("--build-dir") + 1], "sckd_sim/build")
        self.assertEqual(self.calls("smoke")[0][1:], ["models/generated/x4_tennis_demo.mjcf",
                                                     "3", "qp", "0.25", "2026", "pfly"])

    def test_missing_packages_are_installed_before_python_without_elevating_python(self):
        result = self.invoke(MOCK_MISSING="1")
        self.assertEqual(result.returncode, 0, result.stderr)
        apt = self.calls("apt-get")
        self.assertEqual(apt[0], ["update"])
        self.assertEqual(apt[1][:2], ["install", "-y"])
        self.assertIn("python3-venv", apt[1])
        self.assertIn("libeigen3-dev", apt[1])
        self.assertTrue(all(command[0] == "apt-get" for command in self.calls("sudo")))

    def test_apt_failure_stops_before_environment_or_simulation(self):
        result = self.invoke(MOCK_MISSING="1", MOCK_APT_STATUS="7")
        self.assertEqual(result.returncode, 7, result.stderr)
        self.assertEqual(self.calls("apt-get"), [["update"]])
        self.assertFalse(self.calls("python3"))
        self.assertFalse(self.calls("smoke"))

    def test_python_failure_stops_before_simulation(self):
        result = self.invoke("--skip-system-deps", MOCK_PYTHON_STATUS="9")
        self.assertEqual(result.returncode, 9, result.stderr)
        self.assertFalse(self.calls("smoke"))

    def test_simulation_failure_is_not_reported_as_success(self):
        result = self.invoke("--skip-system-deps", MOCK_SMOKE_STATUS="1")
        self.assertEqual(result.returncode, 1, result.stderr)

    def test_smoke_does_not_inherit_viewer_state_writer(self):
        for value in ("1", "0"):
            with self.subTest(inherited_m2_view=value):
                result = self.invoke("--skip-system-deps", M2_VIEW=value)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(Path(self.env["MOCK_VIEW_FILE"]).read_text(), "")

    def test_unsupported_architecture_stops_before_package_changes(self):
        result = self.invoke(MOCK_ARCH="aarch64")
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse(self.calls("dpkg-query"))
        self.assertFalse(self.calls("apt-get"))
        self.assertFalse(self.calls("python3"))

    def test_invalid_jobs_stop_before_package_changes(self):
        for value in ("0", "-1", "abc"):
            with self.subTest(jobs=value):
                result = self.invoke("--jobs", value)
                self.assertEqual(result.returncode, 2, result.stderr)
        self.assertFalse(self.calls("apt-get"))
        self.assertFalse(self.calls("python3"))

    def test_help_is_available_without_system_checks_or_installation(self):
        result = self.invoke("--help", MOCK_ARCH="aarch64", MOCK_MISSING="1")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("用法", result.stdout)
        self.assertFalse(self.calls("dpkg-query"))
        self.assertFalse(self.calls("apt-get"))
        self.assertFalse(self.calls("python3"))


class SetupPreflightTests(unittest.TestCase):
    def test_missing_or_corrupt_osqp_stops_before_installing_packages(self):
        spec = importlib.util.spec_from_file_location("setup_under_test", ROOT / "tools/setup_predictor_ubuntu.py")
        setup = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(setup)
        with tempfile.TemporaryDirectory(prefix="mpc-setup-preflight-") as directory:
            root = Path(directory)
            lib = root / "sckd_sim/third_party/osqp/libosqp.a"
            for content in (None, b"corrupt archive"):
                with self.subTest(content=content):
                    if content is not None:
                        lib.parent.mkdir(parents=True)
                        lib.write_bytes(content)
                    with mock.patch.object(setup, "ROOT", root), \
                            mock.patch.object(setup, "run") as run, \
                            mock.patch.object(sys, "argv", ["setup_predictor_ubuntu.py"]):
                        with self.assertRaises((SystemExit, RuntimeError)):
                            setup.main()
                        run.assert_not_called()
                    self.assertFalse((root / ".venv").exists())


if __name__ == "__main__":
    unittest.main()
