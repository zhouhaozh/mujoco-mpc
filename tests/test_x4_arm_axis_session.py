#!/usr/bin/env python3
"""仅SDK模拟器：批量输入拒绝、人工记录、回程切轴、途中停止、HUP退出。"""
import argparse
import csv
import json
import os
import pathlib
import selectors
import signal
import subprocess
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", required=True, type=pathlib.Path)
    parser.add_argument("--config", required=True, type=pathlib.Path)
    parser.add_argument("--output", required=True, type=pathlib.Path)
    args = parser.parse_args()
    output = args.output.resolve()
    assert not output.exists(), "use a new output directory"
    p = subprocess.Popen([str(args.binary.resolve()), "--config", str(args.config.resolve()),
                          "--backend", "sim", "--step-deg", "1", "--output", str(output)],
                         stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    selector = selectors.DefaultSelector()
    selector.register(p.stdout, selectors.EVENT_READ)
    transcript = bytearray()
    pending = bytearray()

    def wait_for(text, timeout=12):
        target = text.encode()
        deadline = time.monotonic() + timeout
        while target not in pending:
            assert p.poll() is None, transcript.decode(errors="replace")
            ready = selector.select(max(0, deadline - time.monotonic()))
            assert ready, ("timeout", text, transcript.decode(errors="replace"))
            chunk = os.read(p.stdout.fileno(), 4096)
            assert chunk, transcript.decode(errors="replace")
            transcript.extend(chunk)
            pending.extend(chunk)
        end = pending.index(target) + len(target)
        del pending[:end]

    def send(text):
        p.stdin.write(text.encode())
        p.stdin.flush()

    try:
        wait_for("已稳定")
        # 旧逐行/拍实现会把第二个+留到2.4秒后；当前必须整批拒绝。
        send("+\n" + "\n" * 600 + "+\n")
        wait_for("拒绝整批命令")
        send("+\n")
        wait_for("已稳定")
        send("record 2 opposite simulated_observation_only\n")
        time.sleep(.10)
        send("-\n")
        wait_for("已稳定")
        send("next\n")
        wait_for("J2 holding")
        send("+\n")
        time.sleep(.35)
        send("stop\n")
        wait_for("本轮点动已终止")
        send("+\n")
        wait_for("本轮已停止")
        # 检查终端断开信号也进入受控冻结/Stop路径。
        p.send_signal(signal.SIGHUP)
        assert p.wait(timeout=5) == 130
        transcript.extend(p.stdout.read())
    finally:
        if p.poll() is None:
            p.terminate()
            p.wait(timeout=5)
        selector.close()
    output.joinpath("session_transcript.txt").write_bytes(transcript)
    events = [json.loads(line) for line in output.joinpath("events.jsonl").read_text().splitlines()]
    assert sum(e["type"] == "batch_rejected" for e in events) == 1
    assert [e["detail"] for e in events if e["type"] == "move"] == ["J1+", "J1-", "J2+"]
    observed = json.loads(output.joinpath("direction_observations.json").read_text())
    assert observed["observations"] == [{"command_joint": 1, "observed_physical_joint": 2,
                                          "direction": "opposite", "note": " simulated_observation_only"}]
    result = json.loads(output.joinpath("result.json").read_text())
    assert result["exit_code"] == 130 and result["reason"] == "signal"
    stop_time = next(e["t"] for e in events if e["type"] == "software_stop")
    rows = list(csv.DictReader(output.joinpath("samples.csv").open()))
    after = [r for r in rows if float(r["monotonic_s"]) > stop_time]
    assert after
    for j in range(1, 6):
        assert all(float(r[f"j{j}_target_dq"]) == 0 for r in after)
        assert max(float(r[f"j{j}_target_q"]) for r in after) == min(float(r[f"j{j}_target_q"]) for r in after)
    print("interactive simulation checks passed: batch/record/return/select/stop/HUP")


if __name__ == "__main__":
    main()
