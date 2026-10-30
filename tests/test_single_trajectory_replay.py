import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'tools'))
import test_measured_dynamics
from run_measured_dynamics import BINARY, group_directories

SCRIPT = ROOT/'tools/run_measured_dynamics.py'
CONFIG_DIR = ROOT/'sckd_sim/models/config/measured_dynamics/current'


class SingleTrajectoryReplayTests(unittest.TestCase):
    def invoke(self, *arguments):
        return subprocess.run([sys.executable, str(SCRIPT), *map(str, arguments)],
                              capture_output=True, text=True, timeout=30)

    @unittest.skipUnless(BINARY.exists(), 'build x4_command_replay first')
    def test_single_file_does_not_read_sibling_csv_and_keeps_current_settings(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = test_measured_dynamics.DynamicsReplayTests().write_log(root)
            # This sibling would fail parsing if single-file selection fell back to the directory.
            (root/'other.csv').write_text('invalid sibling recording\n')
            output = root/'replay'
            result = self.invoke('--config-dir', CONFIG_DIR, '--trajectory-file', source, '--no-build', '--output', output)
            self.assertEqual(result.returncode, 0, result.stderr or result.stdout)
            self.assertEqual([p.name for p in group_directories(output)], ['group_1'])
            self.assertFalse((output/'group_2_input.json').exists())
            manifest = json.loads((output/'manifest.json').read_text())
            self.assertEqual(list(manifest['experiment_sha256']), ['1'])
            self.assertEqual(manifest['initialization_mode'], 'directory_settings')
            self.assertEqual(manifest['initialization_by_group']['1']['source_path'], str(source.resolve()))
            native = json.loads((output/'group_1/summary.json').read_text())
            self.assertEqual(native['arm_start_policy'], 'hold_until_fresh_command')
            self.assertTrue(native['finite'])
            self.assertTrue(native['time_advancement_sane'])
            self.assertFalse(any(native['warnings'].values()))

    def test_single_file_rejects_missing_file_and_directory_before_writing_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for source in (root/'missing.csv', root):
                output = root/'replay'
                result = self.invoke('--config-dir', CONFIG_DIR, '--trajectory-file', source, '--no-build', '--output', output)
                self.assertEqual(result.returncode, 2)
                self.assertIn('existing CSV file', result.stderr)
                self.assertFalse(output.exists())

    def test_single_file_cannot_be_combined_with_directory_or_saved_replay(self):
        result = self.invoke('--trajectory-file', 'one.csv', '--data-dir', 'recordings')
        self.assertEqual(result.returncode, 2)
        self.assertIn('not allowed with argument', result.stderr)
        result = self.invoke('--trajectory-file', 'one.csv', '--view-run', 'previous')
        self.assertEqual(result.returncode, 2)
        self.assertIn('--view-run cannot be combined', result.stderr)


if __name__ == '__main__':
    unittest.main()
