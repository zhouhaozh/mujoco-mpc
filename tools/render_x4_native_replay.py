#!/usr/bin/env python3
"""Render recorded X4 replay frames without advancing MuJoCo physics."""
import argparse
import json
import os
import subprocess
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
import mujoco
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, type=Path)
    ap.add_argument("--output", type=Path)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--speed", type=float, default=0.25)
    a = ap.parse_args()
    if a.fps <= 0 or a.speed <= 0:
        ap.error("--fps and --speed must be positive")
    run = a.run.resolve(); out = (a.output or run).resolve(); out.mkdir(parents=True, exist_ok=True)
    summary = json.loads((run / "summary.json").read_text())
    scene = Path(summary["scene"]).resolve()
    frames = [json.loads(x) for x in (run / "frames.jsonl").read_text().splitlines() if x.strip()]
    if not frames: raise RuntimeError("frames.jsonl is empty")
    model = mujoco.MjModel.from_xml_path(str(scene)); data = mujoco.MjData(model)
    model.vis.global_.offwidth = 1280
    model.vis.global_.offheight = 720
    renderer = mujoco.Renderer(model, height=720, width=1280)
    mp4 = out / "preview.mp4"; png = out / "preview.png"
    source_t = np.asarray([float(f.get("t", i)) for i, f in enumerate(frames)])
    duration = max(0.0, float(source_t[-1] - source_t[0]))
    output_count = max(1, int(np.ceil(duration / a.speed * a.fps)))
    output_sources = source_t[0] + np.arange(output_count) / a.fps * a.speed
    ff = subprocess.Popen(["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "1280x720", "-r", str(a.fps), "-i", "-", "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(mp4)], stdin=subprocess.PIPE)
    srt = []
    try:
        for i, src_t in enumerate(output_sources):
            fr = frames[min(len(frames) - 1, int(np.searchsorted(source_t, src_t, side="right") - 1))]
            data.qpos[:] = np.asarray(fr["qpos"], dtype=float)
            data.qvel[:] = np.asarray(fr["qvel"], dtype=float)
            mujoco.mj_forward(model, data)
            cam = mujoco.MjvCamera(); mujoco.mjv_defaultFreeCamera(model, cam)
            ball = data.qpos[-7:-4] if len(data.qpos) >= 7 else np.array([3., 0., 1.])
            cam.lookat[:] = [float((ball[0] + data.qpos[0]) * .5), float((ball[1] + data.qpos[1]) * .5), 1.0]
            cam.distance = max(5.0, min(14.0, 5.0 + abs(float(ball[0] - data.qpos[0])) * .35))
            cam.azimuth = 90; cam.elevation = -12
            renderer.update_scene(data, cam); rgb = renderer.render()
            ff.stdin.write(np.asarray(rgb, dtype=np.uint8).tobytes())
            target = fr.get("target", {}); valid = target.get("valid", False)
            text = (f"X4_950 replay | measured reference / no physical impact | t={fr.get('t',0):.3f}s "
                    f"phase={fr.get('phase','?')} solver={fr.get('executing',False)} "
                    f"target={'valid' if valid else 'invalid'}")
            start = i / a.fps; end = (i + 1) / a.fps
            srt.extend([str(i + 1), f"{int(start//3600):02d}:{int(start%3600//60):02d}:{start%60:06.3f} --> {int(end//3600):02d}:{int(end%3600//60):02d}:{end%60:06.3f}", text, ""])
            if i == 0: first = rgb
    finally:
        ff.stdin.close()
        if ff.wait() != 0:
            raise RuntimeError("ffmpeg failed while encoding preview.mp4")
        renderer.close()
    (out / "preview.srt").write_text("\n".join(srt))
    subprocess.run(["ffmpeg", "-y", "-i", str(mp4), "-vf", f"subtitles={out/'preview.srt'}", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(mp4.with_suffix('.overlay.mp4'))], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    mp4.with_suffix('.overlay.mp4').replace(mp4)
    subprocess.run(["ffmpeg", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "1280x720", "-i", "-", "-frames:v", "1", str(png)], input=np.asarray(first, dtype=np.uint8).tobytes(), check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print(json.dumps({"scene": str(scene), "frames": len(frames), "preview": str(mp4), "png": str(png), "mj_step_called": False}))

if __name__ == "__main__": main()
