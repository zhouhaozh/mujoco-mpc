"""Ubuntu纯stdlib fixture：实际调用参数、已解析身份与0命中负结果保留。"""
from contextlib import redirect_stdout
import io
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import check_time_windows_synthetic as confirmation


class SyntheticConfirmationFixtureTests(unittest.TestCase):
    def test_zero_hit_return_one_is_completed_and_requested_arguments_are_consumed(self):
        with tempfile.TemporaryDirectory(prefix="synthetic-time-fixture-") as directory:
            root = Path(directory)
            team, prediction, source = root / "team", root / "prediction", root / "source"
            for name in ("sckd_sim/models/generated", "sckd_sim/cpp/runtime",
                         "sckd_sim/third_party/nmpc6", "sckd_sim/build-independent"):
                (team / name).mkdir(parents=True)
            (team / "sckd_sim/models/generated/x4_tennis_demo.mjcf").write_text("<mujoco/>")
            (team / "sckd_sim/cpp/runtime/fixture.h").write_text("// fixed source\n")
            (team / "sckd_sim/CMakeLists.txt").write_text("# fixed fixture\n")
            (team / "sckd_sim/third_party/nmpc6/nlp.so").write_text("unused default fixture solver")
            nlp = root / "independent_nlp.so"
            nlp.write_text("independent stage-dt fixture solver")
            binary = team / "sckd_sim/build-independent/m2_intercept"
            binary.write_text(f"#!{sys.executable}\n" + r'''
import json,os,sys
from pathlib import Path
policy_path=Path(os.environ['M2_TIME_WINDOWS_PATH'])
policy=json.loads(policy_path.read_text())
values={'scene_path':sys.argv[1],'trials':int(sys.argv[2]),'swing_mode':sys.argv[3],
 'init_delay_s':float(sys.argv[4]),'seed':int(sys.argv[5]),'base_mode':sys.argv[6],
 'serve_family':sys.argv[7],'fly_mode':sys.argv[8],
 'control_variant':os.environ['M2_CONTROL_VARIANT'],'steer_rate_mode':os.environ['M2_STEER_RATE_MODE'],
 'time_windows_path':str(policy_path),'time_windows':policy,
 'nmpc6_nlp_path':os.environ['M2_NMPC6_NLP'],'predictor_socket_path':os.environ['BALL_PREDICTOR_SOCKET'],
 'measured_replay_path':None}
print('[resolved config] '+json.dumps({'values':values,'sources':{'time_windows':'file:'+str(policy_path)}}))
calls=Path(__file__).resolve().parents[2]/'calls.jsonl'
with calls.open('a') as stream:
 stream.write(json.dumps({'argv':sys.argv,'values':values,'EXP_POS':os.environ.get('EXP_POS')})+'\n')
print('命中 0/'+sys.argv[2]+'(0%) 接近率 0% 平均击球点偏差 n/am')
print('旧机NMPC: 0 解 7 拒收 平均 0ms 最大单解 1ms 回退 3 trial')
raise SystemExit(1)
''')
            binary.chmod(0o755)
            (prediction / ".venv-tennis/bin").mkdir(parents=True)
            (prediction / ".venv-tennis/bin/python").symlink_to(sys.executable)
            (source / "tennis_calibration").mkdir(parents=True)
            service = source / "tennis_calibration/fixture_service.py"
            service.write_text('''import json,socket,sys,time
from pathlib import Path
Path(sys.argv[3]).write_text(Path(sys.argv[2]).read_text())
sock=socket.socket(socket.AF_UNIX);sock.bind(sys.argv[1]);sock.listen(1)
time.sleep(60)
''')
            policy_path = root / "selected_policy.json"
            policy = {"schema": "SCKD_TIME_WINDOWS/1", "pfly_init_delay_s": .13,
                      "nmpc6": {"dt_ratios": [.5, .3, .2], "allow_nonuniform_dt": True}}
            policy_path.write_text(json.dumps(policy))
            bundle_path = root / "synthetic_bundle.json"
            bundle = {"schema": "TENNIS_MODEL_BUNDLE/1", "fingerprint": "a" * 64,
                      "resources": {}, "settings": {"frame": {"mode": "synthetic_world"}}}
            bundle_path.write_text(json.dumps(bundle))
            settings_path = root / "settings.json"
            settings_path.write_text(json.dumps([{"id": "fixed_finalist", "mode": "nmpc6",
                "policy": policy_path.name, "bundle": bundle_path.name, "nlp_path": nlp.name}]))
            args = SimpleNamespace(team=team, prediction=prediction, prediction_source=source,
                binary=binary, out=root / "out", settings=settings_path, families="fixed", seeds="31",
                trials=3, timeout=5, startup_timeout=3, retry_incomplete=False)

            def inspect(_prediction, **kwargs):
                return json.loads(Path(kwargs["model_bundle"]).read_text())

            def command(_prediction, _selection, dest, **kwargs):
                return [sys.executable, str(service), "{socket}", str(kwargs["model_bundle"]),
                        str(dest / "predictor_resolved_manifest.json")]

            with patch.object(confirmation, "inspect_predictor_bundle", side_effect=inspect), \
                 patch.object(confirmation, "predictor_service_command", side_effect=command), \
                 patch.dict(os.environ, {"EXP_POS": "9", "M2_CONTROL_VARIANT": "wrong"}), \
                 redirect_stdout(io.StringIO()):
                runner = confirmation.SyntheticConfirmation(args)
                first = runner.run()
                second = runner.run()
            self.assertTrue(first["complete"], first["technical_failures"])
            self.assertEqual(first, second)
            self.assertEqual(first["completed_balls"], 3)
            self.assertEqual(first["settings"][0]["hits"], 0)
            self.assertEqual(first["settings"][0]["hit_rate"], 0)
            self.assertFalse(first["selection_changed"])
            group = first["groups"][0]
            self.assertEqual(group["returncode"], 1)
            self.assertEqual(group["denominator"], 3)
            self.assertEqual(group["experiment"]["status"], "completed")
            self.assertEqual(group["experiment"]["spec"]["resources"]["loaded_nlp"]["path"], str(nlp))
            self.assertEqual(group["predictor_bundle_fingerprint"], bundle["fingerprint"])
            calls = [json.loads(line) for line in (team / "calls.jsonl").read_text().splitlines()]
            self.assertEqual(len(calls), 1)  # 完整同身份重跑复用，0命中也不丢弃。
            self.assertEqual(calls[0]["argv"][2:], ["3", "qp", "0.13", "31", "nmpc6", "fixed", "v2"])
            self.assertEqual(calls[0]["values"]["steer_rate_mode"], "both_unlimited")
            self.assertIsNone(calls[0]["EXP_POS"])


if __name__ == "__main__":
    unittest.main()
